"""Answer generation module using a Qwen VL model for SFT test split files."""

import json
import logging
from pathlib import Path

from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from mergeeda.models.builder import build_model
from mergeeda.utils import IMAGE_SUFFIXES

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generate model answers for an SFT test split file and save preds.json.

    Reads a single SFT-format JSON file (final_dataset_test.json) produced by
    TrainDataGenerator, queries the Qwen VL model for each item, and writes
    preds.json to the output directory.

    Each SFT item has the shape:
        {
            "conversations": [
                {"from": "human", "value": "<image>\\nQuestion text"},
                {"from": "gpt",   "value": "Gold answer text"}
            ],
            "source_chunk": "chunk.md",
            "type": "L1",
            "image": ["dataset/materials/img.jpg"],   # optional
            "material": "img.jpg"                     # optional
        }

    Material handling:
    - Image material: PIL image loaded from materials_dir and passed via imgs;
      the leading "<image>\\n" token is stripped from the question string
      because QwenVLModel inserts image tokens itself via apply_chat_template.
    - Table material: the "[Table: ...]\n...\n\n" block is already embedded in
      conversations[0]["value"] and is kept as-is so the model reads it as text.
    """

    def __init__(self, model_cfg: DictConfig) -> None:
        """Initialize AnswerGenerator by building the model from config."""
        self._model = build_model(model_cfg)
        logger.info(f"AnswerGenerator initialized with model: {model_cfg.name}")

    def generate(
        self,
        sft_test_file: str | Path,
        materials_dir: str | Path,
        output_path: str | Path,
    ) -> None:
        """Generate answers for every item in sft_test_file and save preds.json.

        Reads the SFT test split JSON, queries the model for each item, and
        writes preds.json containing each item's original fields plus the
        model's answer (in 'answer') and the GPT gold answer (in 'gold_answer').
        """
        sft_test_file = Path(sft_test_file)
        materials_dir = Path(materials_dir)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if not sft_test_file.exists():
            raise FileNotFoundError(f"SFT test file not found: {sft_test_file}")

        items = self._load_sft_file(sft_test_file)
        if not items:
            logger.warning(f"No items loaded from: {sft_test_file}")
            return

        logger.info(f"Loaded {len(items)} items from: {sft_test_file}")

        results: list[dict] = []
        for idx, item in enumerate(
            tqdm(items, desc="Generating answers"), start=1
        ):
            question, imgs = self._prepare_input(item, materials_dir)
            gold_answer = item.get("conversations", [{}, {}])[1].get(
                "value", ""
            )

            try:
                answer = self._model(question, imgs if imgs else None)
            except Exception as e:
                logger.error(f"Model inference failed for item {idx}: {e}")
                answer = ""

            result: dict = {
                "id": idx,
                "type": item.get("type"),
                "source_chunk": item.get("source_chunk"),
                "material": item.get("material"),
                "question": question,
                "gold_answer": gold_answer,
                "answer": answer,
            }
            # omit None-valued keys to keep output clean
            result = {k: v for k, v in result.items() if v is not None}
            results.append(result)

        output_file = output_path / "preds.json"
        output_file.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Saved {len(results)} predictions -> {output_file}")

    def _load_sft_file(self, path: Path) -> list[dict]:
        """Load and return items from an SFT JSON file."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                logger.error(
                    f"Unexpected format in {path.name}, expected a list"
                )
                return []
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load {path.name}: {e}")
            return []

    def _prepare_input(
        self,
        item: dict,
        materials_dir: Path,
    ) -> tuple[str, list[Image.Image]]:
        """Extract the question string and images from an SFT item.

        For image material items, strips the leading "<image>\\n" from the
        human turn value (QwenVLModel handles image token insertion itself)
        and loads the image file from materials_dir.

        For table material items, the human turn value already contains the
        "[Table: filename]\\n<content>\\n\\n" block and is passed unchanged.
        """
        human_value: str = item.get("conversations", [{}])[0].get("value", "")
        material_filename: str | None = item.get("material")
        imgs: list[Image.Image] = []

        if material_filename:
            suffix = Path(material_filename).suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                # Strip the <image>\n prefix — QwenVLModel inserts image tokens
                # via apply_chat_template when PIL images are passed in imgs.
                human_value = human_value.removeprefix("<image>\n")
                material_path = materials_dir / material_filename
                if material_path.exists():
                    try:
                        imgs.append(Image.open(material_path).convert("RGB"))
                    except OSError as e:
                        logger.warning(
                            f"Failed to open image {material_path}: {e}"
                        )
                else:
                    logger.warning(f"Material image not found: {material_path}")
            # Table (.txt): human_value already contains [Table: ...]\n<content>\n\n
            # so no extra handling is needed.

        return human_value, imgs
