"""Script to evaluate a model on an AMBA evaluation Question set."""

import logging
import os
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from mergeeda.evaluation import AnswerGenerator, LLMJudgeEvaluator

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../configs/evaluation",
    config_name="eval_model",
)
def main(cfg: DictConfig) -> None:
    """Run answer generation and LLM-judge evaluation on an AMBA Question set."""
    logger.info("Starting model evaluation")
    logger.info(f"Question directory: {cfg.questions_dir}")
    logger.info(f"Materials directory: {cfg.materials_dir}")
    logger.info(f"Chunks directory: {cfg.chunks_dir}")
    logger.info(f"Output directory: {cfg.output_dir}")
    logger.info(f"Model: {cfg.model.name}")
    logger.info(f"Judge model: {cfg.judge.name}")
    logger.info(f"Include specification in prompt: {cfg.include_specification}")

    original_cwd = Path(get_original_cwd())
    question_path = original_cwd / cfg.questions_dir
    materials_path = original_cwd / cfg.materials_dir
    chunks_path = original_cwd / cfg.chunks_dir
    output_path = original_cwd / cfg.output_dir

    if not question_path.exists():
        raise FileNotFoundError(
            f"Question directory not found: {question_path}"
        )
    if not materials_path.exists():
        raise FileNotFoundError(
            f"Materials directory not found: {materials_path}"
        )
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_path}")

    # Step 1: Generate answers with the target model
    logger.info("Step 1/2: Generating model answers")
    generator = AnswerGenerator(model_cfg=cfg.model)
    generator.generate(
        questions_dir=question_path,
        materials_dir=materials_path,
        output_path=output_path,
        chunks_dir=chunks_path,
        include_specification=cfg.include_specification,
    )
    logger.info("Answer generation completed")

    # Step 2: Evaluate answers with the LLM judge
    logger.info("Step 2/2: Running LLM judge evaluation")
    api_key: str | None = cfg.judge.api_key or os.environ.get("OPENAI_API_KEY")
    evaluator = LLMJudgeEvaluator(
        model=cfg.judge.name,
        api_key=api_key,
        max_workers=cfg.judge.max_workers,
    )
    preds_path = output_path / "preds.json"
    evaluator.evaluate(
        preds_path=preds_path,
        chunks_dir=chunks_path,
        materials_dir=materials_path,
        output_path=output_path,
    )
    logger.info("LLM judge evaluation completed")

    logger.info("Model evaluation completed successfully")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"  - Predictions: {output_path / 'preds.json'}")
    logger.info(f"  - Scores:      {output_path / 'scores.json'}")


if __name__ == "__main__":
    main()
