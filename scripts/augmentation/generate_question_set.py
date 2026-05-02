"""Script to generate Question sets from AMBA document chunks."""

import logging
import os
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from mergeeda.augmentation import EvalQSetGenerator

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../configs/augmentation",
    config_name="generate_question_set",
)
def main(cfg: DictConfig) -> None:
    """Generate Question sets from AMBA document text chunks."""
    logger.info("Starting AMBA Question set generation")
    logger.info(f"Chunks directory: {cfg.chunks_dir}")
    logger.info(f"Materials directory: {cfg.materials_dir}")
    logger.info(f"Output directory: {cfg.output_dir}")
    logger.info(f"Output name: {cfg.output_name}")
    logger.info(f"Model: {cfg.model.name}")

    original_cwd = Path(get_original_cwd())
    chunks_path = original_cwd / cfg.chunks_dir
    materials_path = original_cwd / cfg.materials_dir
    output_path = original_cwd / cfg.output_dir

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_path}")
    if not materials_path.exists():
        raise FileNotFoundError(
            f"Materials directory not found: {materials_path}"
        )

    # Resolve API key: config takes precedence, then environment variable
    api_key: str | None = cfg.model.api_key or os.environ.get("OPENAI_API_KEY")

    generator = EvalQSetGenerator(
        model=cfg.model.name,
        api_key=api_key,
        max_workers=cfg.model.max_workers,
    )

    generator.generate(
        chunks_dir=chunks_path,
        materials_dir=materials_path,
        output_path=output_path,
        output_name=cfg.output_name,
    )

    logger.info("AMBA Question set generation completed successfully")
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
