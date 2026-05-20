"""One-off script: build the CAT alpha-training mix dataset.

Mixes an equal number of samples from two skills into a single ShareGPT-format
JSON file consumed by CATConversationDataset:

  - AMBA QA   : local final_dataset_train.json (already ShareGPT format).
  - PyraNet   : bnadimi/PyraNet-Verilog from the HF Hub, filtered by quality
                (rank > threshold, complexity in {Basic, Expert},
                compile_status == "No error!") then converted to ShareGPT.

Usage:
  python scripts/merge/build_cat_mix.py \
      --amba-path data/datasets/amba_document/sft_data/final_dataset_train.json \
      --output data/datasets/cat_mix_train.json \
      --per-skill 500
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("build_cat_mix")

# PyraNet dataset on the HF Hub.
_PYRANET_REPO = "bnadimi/PyraNet-Verilog"
# compile_status value indicating a clean compile (note the exact spelling).
_PYRANET_OK_STATUS = "No error!"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build the CAT mix dataset.")
    parser.add_argument(
        "--amba-path",
        default="data/datasets/amba_document/sft_data/final_dataset_train.json",
        help="Path to the AMBA ShareGPT training JSON.",
    )
    parser.add_argument(
        "--output",
        default="data/datasets/cat_mix_train.json",
        help="Output path for the mixed dataset.",
    )
    parser.add_argument(
        "--per-skill",
        type=int,
        default=500,
        help="Number of samples drawn from each skill (equal counts).",
    )
    parser.add_argument(
        "--pyranet-rank-min",
        type=int,
        default=15,
        help="Keep PyraNet samples with rank strictly greater than this.",
    )
    parser.add_argument(
        "--pyranet-complexity",
        nargs="+",
        default=["Basic", "Expert"],
        help="PyraNet complexity levels to keep.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for sampling.",
    )
    return parser.parse_args()


def load_amba_samples(
    amba_path: Path, count: int, rng: random.Random
) -> list[dict]:
    """Load and sample AMBA QA conversations (already ShareGPT format)."""
    if not amba_path.is_file():
        sys.exit(f"AMBA dataset not found: {amba_path}")

    with amba_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        sys.exit(f"AMBA dataset must be a JSON list, got {type(data).__name__}")

    usable = [item for item in data if item.get("conversations")]
    logger.info("AMBA: %d usable conversations", len(usable))
    if len(usable) < count:
        sys.exit(
            f"AMBA has only {len(usable)} samples, need {count}"
        )

    sampled = rng.sample(usable, count)
    # Keep only the conversations field; tag the skill for traceability.
    return [
        {"conversations": item["conversations"], "skill": "amba_qa"}
        for item in sampled
    ]


def _parse_description(raw: object) -> dict | None:
    """Parse the PyraNet 'description' cell into a dict.

    The CSV stores this column as a JSON string (dtype=string), so it must be
    json.loads-ed before its keys (rank, complexity, ...) can be read.
    """
    if isinstance(raw, dict):
        return raw  # already parsed (defensive)
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _pyranet_to_sharegpt(code: object, desc: dict) -> dict | None:
    """Convert a PyraNet (code, parsed-description) pair to a ShareGPT item."""
    instruction = desc.get("description")
    if not code or not instruction:
        return None
    return {
        "conversations": [
            {"from": "human", "value": str(instruction)},
            {"from": "gpt", "value": str(code)},
        ],
        "skill": "pyranet",
    }


def _pyranet_keep(desc: dict, rank_min: int, complexity: set[str]) -> bool:
    """Return True if a PyraNet row passes the quality filters."""
    if desc.get("compile_status") != _PYRANET_OK_STATUS:
        return False
    if desc.get("complexity") not in complexity:
        return False
    # rank is stored as a string in the dataset.
    try:
        rank = int(desc.get("rank", -1))
    except (TypeError, ValueError):
        return False
    return rank > rank_min


def load_pyranet_samples(
    count: int,
    rank_min: int,
    complexity: list[str],
    rng: random.Random,
) -> list[dict]:
    """Download, filter, and sample PyraNet Verilog conversations."""
    complexity_set = set(complexity)
    logger.info(
        "Loading PyraNet from HF Hub: %s (this downloads ~GBs on first run)",
        _PYRANET_REPO,
    )
    dataset = load_dataset(_PYRANET_REPO, split="train")

    filtered: list[dict] = []
    parse_failures = 0
    for row in dataset:
        desc = _parse_description(row.get("description"))
        if desc is None:
            parse_failures += 1
            continue
        if not _pyranet_keep(desc, rank_min, complexity_set):
            continue
        sample = _pyranet_to_sharegpt(row.get("code"), desc)
        if sample is not None:
            filtered.append(sample)

    if parse_failures:
        logger.warning(
            "PyraNet: %d rows had an unparseable 'description' field",
            parse_failures,
        )
    logger.info(
        "PyraNet: %d samples pass filters (rank>%d, complexity=%s, status=%r)",
        len(filtered),
        rank_min,
        sorted(complexity_set),
        _PYRANET_OK_STATUS,
    )
    if len(filtered) < count:
        sys.exit(
            f"PyraNet has only {len(filtered)} filtered samples, need {count}"
        )

    return rng.sample(filtered, count)


def main() -> None:
    """Build and write the CAT mix dataset."""
    args = parse_args()
    rng = random.Random(args.seed)

    amba = load_amba_samples(Path(args.amba_path), args.per_skill, rng)
    pyranet = load_pyranet_samples(
        args.per_skill,
        args.pyranet_rank_min,
        args.pyranet_complexity,
        rng,
    )

    mix = amba + pyranet
    rng.shuffle(mix)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(mix, f, ensure_ascii=False, indent=2)

    logger.info(
        "Wrote %d samples (%d amba_qa + %d pyranet) to %s",
        len(mix),
        len(amba),
        len(pyranet),
        output,
    )


if __name__ == "__main__":
    main()
