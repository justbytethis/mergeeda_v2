"""SFT training data generation module using GPT to answer given questions to make train/test Question-Answer pairs."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from mergeeda.utils import (
    IMAGE_SUFFIXES,
    MATERIAL_TAG_PATTERN,
    build_material_content_blocks,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert on the AMBA (Advanced Microcontroller Bus Architecture) specifications.

You are given:
- A text chunk extracted from an AMBA specification document.
- Optional materials referenced by the chunk: tables provided as labeled text blocks ([Table: FILENAME]) and figures provided as inline images, in the order they appear in the text.
- A question about the content of that chunk.

Write a single, high-quality answer to the question.

Rules:
1. Ground the answer strictly in the provided text chunk and materials. Do not invent facts, signals, or behaviors that are not supported by what you are shown.
2. The answer must be SELF-CONTAINED and authoritative. Do NOT reference "the passage", "the chunk", "the provided text", "the figure above", section numbers, or the source in any form. Write as if you are stating the specification directly.
3. Be technically precise: use the exact signal names, field names, and terminology from the AMBA specification as they appear in the text / materials.
4. Keep the answer concise but complete. Cover every point needed to fully answer the question; do not add unrelated background.
5. For questions that reference a figure or table, interpret the relevant material to produce the answer, but still do not mention the figure/table identifier in your response.
6. Output ONLY the answer as plain text. No JSON, no preamble, no markdown code fences, no lists unless genuinely warranted by the content."""


class TrainDataGenerator:
    """Generate Qwen-VL SFT training data by answering Questions via GPT.

    Resolves each question's source chunk from the processed dataset directory,
    calls the GPT model with the chunk text and any referenced materials, and
    saves results in the Qwen-VL finetune schema consumed by data_processor.py.
    API calls are parallelised across items using a thread pool.
    """

    def __init__(
        self,
        model: str = "gpt-5.1",
        api_key: str | None = None,
        max_workers: int = 20,
    ) -> None:
        """Initialize TrainDataGenerator with an OpenAI client."""
        self._model = model
        self._client = OpenAI(api_key=api_key)
        self._max_workers = max_workers
        logger.info(
            f"TrainDataGenerator initialized with model={model}, max_workers={max_workers}"
        )

    def generate(
        self,
        train_file: str | Path,
        dataset_name: str,
        chunks_dir: str | Path,
        materials_dir: str | Path,
        output_file: str | Path,
    ) -> None:
        """Generate SFT items for a single train Question JSON file and save results.

        Reads the train_file (list of Question dicts), resolves each source chunk,
        calls GPT in parallel, and writes all results to output_file once done.
        """
        train_file = Path(train_file)
        chunks_dir = Path(chunks_dir)
        materials_dir = Path(materials_dir)
        output_file = Path(output_file)

        with train_file.open("r", encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list):
            logger.warning(f"Unexpected format (not a list): {train_file}")
            return

        chunk_cache: dict[str, str] = {}
        for item in items:
            source_chunk = item.get("source_chunk", "")
            if source_chunk and source_chunk not in chunk_cache:
                chunk_path = chunks_dir / source_chunk
                if chunk_path.exists():
                    chunk_cache[source_chunk] = chunk_path.read_text(
                        encoding="utf-8"
                    )
                else:
                    logger.warning(f"Chunk not found: {chunk_path}")

        logger.info(f"{train_file.name}: {len(items)} items to process")

        desc = f"{train_file.parent.name}/{train_file.name}"
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            results = list(
                tqdm(
                    executor.map(
                        self._process_item,
                        items,
                        repeat(chunk_cache),
                        repeat(materials_dir),
                        repeat(dataset_name),
                    ),
                    total=len(items),
                    desc=desc,
                    leave=False,
                )
            )

        results = [r for r in results if r is not None]
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            f.write("\n")
        logger.info(f"Wrote {len(results)} items -> {output_file}")

    def _process_item(
        self,
        item: dict,
        chunk_cache: dict[str, str],
        materials_dir: Path,
        dataset_name: str,
    ) -> dict | None:
        """Call GPT for a single item and return the SFT result, or None if skipped."""
        question = item.get("question", "")
        source_chunk = item.get("source_chunk", "")
        if not question or not source_chunk:
            logger.warning(
                f"Skipping item missing question/source_chunk: {item}"
            )
            return None
        chunk_text = chunk_cache.get(source_chunk)
        if chunk_text is None:
            return None

        try:
            answer = self._call_api(question, chunk_text, materials_dir)
        except Exception as e:
            logger.error(f"GPT call failed for {source_chunk}: {e}")
            return None

        return self._build_sft_item(item, answer, dataset_name, materials_dir)

    def _call_api(
        self,
        question: str,
        chunk_text: str,
        materials_dir: Path,
    ) -> str:
        """Call GPT with a chunk + question and return the answer text."""
        messages = self._build_messages(question, chunk_text, materials_dir)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        return (response.choices[0].message.content or "").strip()

    def _build_messages(
        self,
        question: str,
        chunk_text: str,
        materials_dir: Path,
    ) -> list[dict]:
        """Assemble the system + user message list for the GPT call."""
        material_filenames = MATERIAL_TAG_PATTERN.findall(chunk_text)
        user_content: list[dict] = build_material_content_blocks(
            material_filenames, materials_dir
        )
        user_content.append(
            {
                "type": "text",
                "text": (
                    "<text_chunk>\n"
                    f"{chunk_text}\n"
                    "</text_chunk>\n\n"
                    f"Question: {question}\n\n"
                    "Write the answer."
                ),
            }
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _build_sft_item(
        self,
        item: dict,
        answer: str,
        dataset_name: str,
        materials_dir: Path,
    ) -> dict:
        """Wrap a (question, GPT answer) pair into the Qwen-VL SFT schema."""
        question = item.get("question", "")
        material_filename = item.get("material")

        human_text = question
        image_paths: list[str] = []

        if material_filename:
            suffix = Path(material_filename).suffix.lower()
            material_path = materials_dir / material_filename
            if suffix in IMAGE_SUFFIXES:
                if material_path.exists():
                    image_paths.append(
                        f"{dataset_name}/materials/{material_filename}"
                    )
                    human_text = f"<image>\n{question}"
                else:
                    logger.warning(
                        f"Material image not found, omitting: {material_path}"
                    )
            elif suffix == ".txt":
                if material_path.exists():
                    table_text = material_path.read_text(encoding="utf-8")
                    human_text = f"[Table: {material_filename}]\n{table_text}\n\n{question}"
                else:
                    logger.warning(
                        f"Material table not found, omitting: {material_path}"
                    )
            else:
                logger.warning(
                    f"Unsupported material type, omitting: {material_filename}"
                )

        sft_item: dict = {
            "conversations": [
                {"from": "human", "value": human_text},
                {"from": "gpt", "value": answer},
            ],
            "source_chunk": item.get("source_chunk"),
            "type": item.get("type"),
        }
        if image_paths:
            sft_item["image"] = image_paths
        if material_filename:
            sft_item["material"] = material_filename
        return sft_item
