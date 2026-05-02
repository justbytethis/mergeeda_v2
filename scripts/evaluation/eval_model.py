"""Script to evaluate a model on an SFT test split file."""

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
    """Run answer generation and LLM-judge evaluation on an SFT test split file."""
    logger.info("Starting model evaluation")
    logger.info("SFT test file: %s", cfg.sft_test_file)
    logger.info("Processed directory: %s", cfg.processed_dir)
    logger.info("Output directory: %s", cfg.output_dir)
    logger.info("Model: %s", cfg.model.name)
    logger.info("Judge model: %s", cfg.judge.name)
    logger.info("Use gold answer: %s", cfg.use_gold_answer)

    original_cwd = Path(get_original_cwd())
    sft_test_file = original_cwd / cfg.sft_test_file
    processed_path = original_cwd / cfg.processed_dir
    output_path = original_cwd / cfg.output_dir

    if not sft_test_file.exists():
        raise FileNotFoundError(f"SFT test file not found: {sft_test_file}")
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed directory not found: {processed_path}"
        )

    # Step 1: Generate answers with the target model
    logger.info("Step 1/2: Generating model answers")
    generator = AnswerGenerator(model_cfg=cfg.model)
    generator.generate(
        sft_test_file=sft_test_file,
        processed_dir=processed_path,
        output_path=output_path,
    )
    logger.info("Answer generation completed")

    # Step 2: Evaluate answers with the LLM judge
    logger.info("Step 2/2: Running LLM judge evaluation")
    api_key: str | None = cfg.judge.api_key or os.environ.get("OPENAI_API_KEY")
    evaluator = LLMJudgeEvaluator(
        model=cfg.judge.name,
        api_key=api_key,
        max_workers=cfg.judge.max_workers,
        use_gold_answer=cfg.use_gold_answer,
    )
    preds_path = output_path / "preds.json"
    evaluator.evaluate(
        preds_path=preds_path,
        processed_dir=processed_path,
        output_path=output_path,
    )
    logger.info("LLM judge evaluation completed")

    logger.info("Model evaluation completed successfully")
    logger.info("Results saved to: %s", output_path)
    logger.info("  - Predictions: %s", output_path / "preds.json")
    logger.info("  - Scores:      %s", output_path / "scores.json")


if __name__ == "__main__":
    main()
