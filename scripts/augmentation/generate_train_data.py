"""Script to generate SFT training data for AMBA train Question files using GPT."""

import json
import logging
import os
import random
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from mergeeda.augmentation import TrainDataGenerator

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../configs/augmentation",
    config_name="generate_train_data",
)
def main(cfg: DictConfig) -> None:
    """Generate GPT answers for each train Question file, save as SFT JSON, and merge."""
    logger.info("Starting SFT training data generation")
    logger.info(f"Processed dir: {cfg.processed_dir}")
    logger.info(f"Output dir: {cfg.output_dir}")
    logger.info(f"Model: {cfg.model.name}")
    logger.info(f"Train files: {list(cfg.train_files)}")

    original_cwd = Path(get_original_cwd())
    processed_dir = original_cwd / cfg.processed_dir
    output_dir = original_cwd / cfg.output_dir

    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed dir not found: {processed_dir}")

    api_key: str | None = cfg.model.api_key or os.environ.get("OPENAI_API_KEY")

    generator = TrainDataGenerator(
        model=cfg.model.name,
        api_key=api_key,
        max_workers=cfg.model.max_workers,
    )

    for train_file_rel in cfg.train_files:
        train_file = (original_cwd / train_file_rel).resolve()
        if not train_file.exists():
            logger.warning(f"Train file not found, skipping: {train_file}")
            continue

        # Expected layout: .../question_set/<dataset_name>/<filename>_train.json
        dataset_name = train_file.parent.name
        chunks_dir = processed_dir / dataset_name / "chunks"
        materials_dir = processed_dir / dataset_name / "materials"

        if not chunks_dir.exists():
            logger.warning(f"Chunks dir missing, skipping: {chunks_dir}")
            continue
        if not materials_dir.exists():
            logger.warning(f"Materials dir missing, skipping: {materials_dir}")
            continue

        output_file = output_dir / f"{dataset_name}.json"
        logger.info(f"Processing {train_file}")

        generator.generate(
            train_file=train_file,
            dataset_name=dataset_name,
            chunks_dir=chunks_dir,
            materials_dir=materials_dir,
            output_file=output_file,
        )

    logger.info("SFT training data generation completed")

    logger.info(f"Merging SFT data into {output_dir}")

    final_train_path = output_dir / "final_dataset_train.json"
    final_test_path = output_dir / "final_dataset_test.json"
    json_files = sorted(
        p
        for p in output_dir.rglob("*.json")
        if p not in (final_train_path, final_test_path)
    )
    if not json_files:
        logger.warning(f"No JSON files found under {output_dir}")
        return

    merged: list[dict] = []
    for path in json_files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning(f"Skipping non-list JSON: {path}")
            continue
        merged.extend(data)
        logger.info(f"{path.relative_to(output_dir)}: {len(data)} items")

    logger.info(f"Total merged items: {len(merged)}")

    train_ratio: float = float(cfg.get("train_ratio", 0.7))
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(
            f"train_ratio must be in (0, 1), got {train_ratio}"
        )

    seed: int | None = cfg.get("seed", None)
    if seed is not None:
        random.seed(seed)
    random.shuffle(merged)
    split_idx = int(len(merged) * train_ratio)
    train_data = merged[:split_idx]
    test_data = merged[split_idx:]

    output_dir.mkdir(parents=True, exist_ok=True)

    for path, data in (
        (final_train_path, train_data),
        (final_test_path, test_data),
    ):
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")

    logger.info(f"Train: {len(train_data)} items -> {final_train_path}")
    logger.info(f"Test:  {len(test_data)} items -> {final_test_path}")


if __name__ == "__main__":
    main()
