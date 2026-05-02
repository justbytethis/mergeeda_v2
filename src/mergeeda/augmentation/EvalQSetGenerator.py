"""Question Set generation module for all chunks in a dataset."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from .QGenerator import QGenerator

logger = logging.getLogger(__name__)


class EvalQSetGenerator:
    """Generate and save a Question Set from a directory of text chunks."""

    def __init__(
        self,
        model: str = "gpt-5.1",
        api_key: str | None = None,
        max_workers: int = 20,
    ) -> None:
        """Initialize EvalQSetGenerator with an underlying QGenerator."""
        self._question_generator = QGenerator(model=model, api_key=api_key)
        self._max_workers = max_workers
        logger.info(
            f"EvalQSetGenerator initialized with model={model}, max_workers={max_workers}"
        )

    def generate(
        self,
        chunks_dir: str | Path,
        materials_dir: str | Path,
        output_path: str | Path,
        output_name: str,
    ) -> None:
        """Generate Questions for all chunks in parallel and save to a single JSON file.

        Spawns up to max_workers threads, each calling QGenerator for one chunk.
        All results are collected into a single list and written to:
            {output_path}/{output_name}.json
        """
        chunks_dir = Path(chunks_dir)
        materials_dir = Path(materials_dir)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        chunk_files = sorted(
            chunks_dir.glob("*.md"), key=lambda p: self._sort_key(p)
        )
        if not chunk_files:
            logger.warning(f"No .md files found in: {chunks_dir}")
            return

        logger.info(
            f"Processing {len(chunk_files)} chunks from: {chunks_dir} "
            f"with max_workers={self._max_workers}"
        )

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            results_per_chunk = list(
                tqdm(
                    executor.map(
                        self._process_chunk,
                        chunk_files,
                        [materials_dir] * len(chunk_files),
                    ),
                    total=len(chunk_files),
                    desc="Generating Question Set",
                )
            )

        all_questions: list[dict] = [q for qs in results_per_chunk for q in qs]
        logger.info(f"Total questions generated: {len(all_questions)}")

        output_file = output_path / f"{output_name}.json"
        output_file.write_text(
            json.dumps(all_questions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Question Set saved to: {output_file}")

    def _process_chunk(
        self,
        chunk_file: Path,
        materials_dir: Path,
    ) -> list[dict]:
        """Generate Question Set for a single chunk and return results."""
        try:
            questions = self._question_generator.generate(
                chunk_path=chunk_file,
                materials_dir=materials_dir,
            )
        except Exception as e:
            logger.error(
                f"Failed to generate questions for {chunk_file.name}: {e}"
            )
            return []
        return questions or []

    def _sort_key(self, path: Path) -> int:
        """Sort chunk files numerically by stem (e.g., '33.md' -> 33)."""
        try:
            return int(path.stem)
        except ValueError:
            return 0
