"""Answer generation module using a Qwen VL model for Question sets."""

import base64
import json
import logging
import re
from pathlib import Path

from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from mergeeda.models.builder import build_model
from mergeeda.utils import IMAGE_SUFFIXES, MATERIAL_TAG_PATTERN

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generate model answers for Question sets and save unified preds.json.

    Reads all JSON Question files from an input directory, queries the Qwen VL model
    for each question (loading material images/tables where applicable), and
    writes a single preds.json to the output path.
    """

    def __init__(
        self,
        model_cfg: DictConfig,
    ) -> None:
        """Initialize AnswerGenerator by building the model from config."""
        self._model = build_model(model_cfg)
        logger.info(f"AnswerGenerator initialized with model: {model_cfg.name}")

    def generate(
        self,
        questions_dir: str | Path,
        materials_dir: str | Path,
        output_path: str | Path,
        chunks_dir: str | Path | None = None,
        include_specification: bool = False,
    ) -> None:
        """Generate answers for all Question JSON files in questions_dir and save preds.json.

        Iterates over all .json files in questions_dir sorted by filename, assigns
        sequential IDs across files (ascending filename order, then original
        order within each file), queries the model, and writes preds.json.

        When include_specification is True, the content of the source chunk
        markdown file (resolved from chunks_dir using the item's source_chunk
        field) is prepended to the question as "Specification: ...".
        """
        questions_dir = Path(questions_dir)
        materials_dir = Path(materials_dir)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if include_specification and chunks_dir is None:
            raise ValueError(
                "chunks_dir must be provided when include_specification is True"
            )
        chunks_path: Path | None = (
            Path(chunks_dir) if chunks_dir is not None else None
        )

        json_files = sorted(questions_dir.glob("*.json"), key=lambda p: p.name)
        if not json_files:
            logger.warning(f"No .json files found in: {questions_dir}")
            return

        logger.info(
            f"Processing {len(json_files)} Question files from: {questions_dir}"
        )

        all_items: list[dict] = []
        for json_file in json_files:
            items = self._load_question_file(json_file)
            all_items.extend(items)
            logger.info(f"Loaded {len(items)} questions from: {json_file.name}")

        results: list[dict] = []
        for idx, item in enumerate(
            tqdm(all_items, desc="Generating answers"), start=1
        ):
            answer = self._query_model(
                item,
                materials_dir,
                chunks_dir=chunks_path,
                include_specification=include_specification,
            )
            result = {k: v for k, v in item.items()}
            result["id"] = idx
            result["answer"] = answer
            # reorder: id first
            ordered = {"id": result.pop("id")}
            ordered.update(result)
            ordered["answer"] = ordered.pop("answer")
            results.append(ordered)

        output_file = output_path / "preds.json"
        output_file.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Saved {len(results)} predictions -> {output_file}")

    def _load_question_file(self, json_file: Path) -> list[dict]:
        """Load and return items from a single Question JSON file."""
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                logger.warning(
                    f"Unexpected format in {json_file.name}, skipping"
                )
                return []
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load {json_file.name}: {e}")
            return []

    def _query_model(
        self,
        item: dict,
        materials_dir: Path,
        chunks_dir: Path | None = None,
        include_specification: bool = False,
    ) -> str:
        """Query the model with a question and optional material images.

        When include_specification is False, material-type questions load the
        referenced image file (if it is an image) to pass as visual context,
        and table (.txt) materials are appended as text to the question.

        When include_specification is True, the source chunk markdown
        referenced by item["source_chunk"] is prepended to the question as
        "Specification: ...", with any <material:filename> tags inside the
        chunk resolved via _resolve_chunk_materials. In this mode the
        item["material"] field is ignored because the chunk already embeds
        the relevant material at its original location.
        """
        question = item.get("question", "")
        material_filename: str | None = item.get("material")

        imgs: list[Image.Image] = []
        extra_text = ""
        spec_prefix = ""

        if include_specification and chunks_dir is not None:
            source_chunk: str | None = item.get("source_chunk")
            if not source_chunk:
                logger.warning(
                    "include_specification is enabled but item has no 'source_chunk' field"
                )
            else:
                chunk_path = chunks_dir / source_chunk
                if not chunk_path.exists():
                    logger.warning(f"Source chunk file not found: {chunk_path}")
                else:
                    try:
                        chunk_text = chunk_path.read_text(encoding="utf-8")
                        resolved_text, chunk_imgs = (
                            self._resolve_chunk_materials(
                                chunk_text, materials_dir
                            )
                        )
                        spec_prefix = f"Specification: {resolved_text}\n\n"
                        imgs.extend(chunk_imgs)
                    except OSError as e:
                        logger.warning(
                            f"Failed to read chunk {chunk_path}: {e}"
                        )

        if material_filename and not include_specification:
            material_path = materials_dir / material_filename
            if not material_path.exists():
                logger.warning(f"Material file not found: {material_path}")
            else:
                suffix = material_path.suffix.lower()
                if suffix in IMAGE_SUFFIXES:
                    try:
                        imgs.append(Image.open(material_path).convert("RGB"))
                    except OSError as e:
                        logger.warning(
                            f"Failed to open image {material_path}: {e}"
                        )
                elif suffix == ".txt":
                    try:
                        table_text = material_path.read_text(encoding="utf-8")
                        extra_text = (
                            f"\n\n[Table: {material_filename}]\n{table_text}"
                        )
                    except OSError as e:
                        logger.warning(
                            f"Failed to read table {material_path}: {e}"
                        )
                else:
                    logger.warning(
                        f"Unsupported material type, skipping: {material_filename}"
                    )

        full_question = spec_prefix + question + extra_text

        try:
            answer = self._model(full_question, imgs if imgs else None)
        except Exception as e:
            logger.error(
                f"Model inference failed for question: {question[:60]}...: {e}"
            )
            answer = ""

        return answer

    def _resolve_chunk_materials(
        self,
        chunk_text: str,
        materials_dir: Path,
    ) -> tuple[str, list[Image.Image]]:
        """Resolve <material:filename> tags inside a chunk.

        Table (.txt) tags are replaced in place with the file contents.
        Image tags are removed from the text and the images are returned
        separately to be passed to the vision model.
        """
        imgs: list[Image.Image] = []

        def replace(match: re.Match[str]) -> str:
            filename = match.group(1).strip()
            material_path = materials_dir / filename
            if not material_path.exists():
                logger.warning(f"Chunk material not found: {material_path}")
                return ""

            suffix = material_path.suffix.lower()
            if suffix == ".txt":
                try:
                    return material_path.read_text(encoding="utf-8")
                except OSError as e:
                    logger.warning(
                        f"Failed to read chunk table {material_path}: {e}"
                    )
                    return ""
            if suffix in IMAGE_SUFFIXES:
                try:
                    imgs.append(Image.open(material_path).convert("RGB"))
                except OSError as e:
                    logger.warning(
                        f"Failed to open chunk image {material_path}: {e}"
                    )
                return ""
            logger.warning(
                f"Unsupported chunk material type, skipping: {filename}"
            )
            return ""

        resolved = MATERIAL_TAG_PATTERN.sub(replace, chunk_text)
        return resolved, imgs
