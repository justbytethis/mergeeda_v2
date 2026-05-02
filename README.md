# MergeEDA v2

Multimodal VLM pipeline for fine-tuning Qwen3-VL on AMBA specification documents.

## Project Structure

```
MergeEDA_v2/
├── configs/
│   ├── augmentation/
│   │   ├── generate_question_set.yaml
│   │   └── generate_train_data.yaml
│   ├── evaluation/
│   │   ├── eval_model.yaml
│   │   └── model/
│   │       ├── qwen_vl_model.yaml
│   │       └── qwen_vl_finetuned_model.yaml
│   └── preprocess/
│       └── amba_document.yaml
├── data/
│   ├── datasets/amba_document/
│   │   ├── raw/                        # Input AMBA PDF files
│   │   ├── processed/<spec>/
│   │   │   ├── chunks/                 # Markdown chunks (1.md, 2.md, ...)
│   │   │   └── materials/              # Extracted images (.jpg) and tables (.txt)
│   │   ├── question_set/<spec>/        # Question JSON files
│   │   └── sft_data/                   # SFT training JSON files
│   ├── evaluation/                     # preds.json, scores.json
│   └── outputs/                        # Hydra logs
├── scripts/
│   ├── preprocess/amba_document.py
│   ├── augmentation/
│   │   ├── generate_question_set.py
│   │   └── generate_train_data.py
│   ├── evaluation/eval_model.py
│   └── train/train.sh
├── src/mergeeda/
│   ├── preprocess/OCRParser.py
│   ├── augmentation/
│   │   ├── QGenerator.py
│   │   ├── EvalQSetGenerator.py
│   │   └── TrainDataGenerator.py
│   ├── evaluation/
│   │   ├── AnswerGenerator.py
│   │   └── LLMJudgeEvaluator.py
│   ├── models/
│   │   ├── builder.py
│   │   ├── qwen_vl_model.py
│   │   └── qwen_vl_finetuned_model.py
│   └── utils/utils.py
├── qwen-vl-finetune/                   # git submodule
└── pyproject.toml
```

## Setup

```bash
pip install -e .
export OPENAI_API_KEY="sk-..."
```

## Pipeline

### 1. Parse PDF

Uses `OCRParser` (backed by DeepSeek-OCR via vLLM) to convert each PDF page to markdown, extract figures and tables as separate material files, and chunk the result by heading level.

```bash
python scripts/preprocess/amba_document.py \
  input_pdf="data/datasets/amba_document/raw/<file>.pdf" \
  output_dir="data/datasets/amba_document/processed/<spec>"
```

Key options:
- `model.name` — OCR model (default: `deepseek-ai/DeepSeek-OCR`)
- `model.dpi` — PDF-to-image resolution (default: `300`)
- `chunking.level` — heading depth to split on (default: `3`, e.g. `1.2.3`)
- `chunking.level_patterns` — custom regex list (default: auto-detect numeric headings)

Output:
- `processed/<spec>/chunks/` — markdown chunks (`1.md`, `2.md`, ...)
- `processed/<spec>/materials/` — extracted images (`.jpg`) and tables (`.txt`)

Each chunk embeds `<material:FILENAME>` tags where figures or tables were found.

---

### 2. Generate Question Set

Uses `EvalQSetGenerator` (GPT) to produce questions for every chunk in parallel. Questions are tagged by cognitive level (L1–L6) and may reference a single material file. The output is consumed by step 3 as the source of questions for SFT data generation.

```bash
python scripts/augmentation/generate_question_set.py \
  chunks_dir="data/datasets/amba_document/processed/<spec>/chunks" \
  materials_dir="data/datasets/amba_document/processed/<spec>/materials" \
  output_dir="data/datasets/amba_document/question_set/<spec>" \
  output_name="<name>"
```

Key options:
- `model.name` — GPT model (default: `gpt-5.1`)
- `model.max_workers` — parallel API threads (default: `20`)

Output: `question_set/<spec>/<name>.json`

Each item has fields: `question`, `type` (L?), `source_chunk`, and optionally `material`.

---

### 3. Generate SFT Training Data

Uses `TrainDataGenerator` (GPT) to answer each Question item using the source chunk as context, producing Qwen-VL SFT-format JSON.

```bash
python scripts/augmentation/generate_train_data.py \
  'train_files=[data/datasets/amba_document/question_set/<spec>/questions.json]'
```

Key options:
- `processed_dir` — root directory containing per-dataset `chunks/` and `materials/` subdirs (default: `data/datasets/amba_document/processed`)
- `output_dir` — root directory for saving all generated SFT JSON files (default: `data/datasets/amba_document/sft_data`)
- `model.max_workers=20` — parallel API workers
- `train_ratio` — fraction of merged data for training split (default: `0.7`)

Output:
- `sft_data/<spec>.json` — per-dataset SFT JSON
- `sft_data/final_dataset_train.json` — merged training split
- `sft_data/final_dataset_test.json` — merged test split

Each SFT item follows the Qwen-VL finetune schema:
```json
{
  "conversations": [
    {"from": "human", "value": "<question>"},
    {"from": "gpt",   "value": "<answer>"}
  ],
  "image": ["<dataset_name>/materials/<filename>"],  // only for image-type questions
  "source_chunk": "42.md",
  "type": "L3",
  "material": "42_7.jpg"
}
```

---

### 4. Fine-tune (LoRA)

Launches LoRA SFT via the `qwen-vl-finetune` submodule. The `amba_sft` dataset token must be registered in `qwen-vl-finetune/qwenvl/data/__init__.py`. Multi-GPU training uses `torchrun` automatically when `CUDA_VISIBLE_DEVICES` exposes more than one GPU.

```bash
bash scripts/train/train.sh Qwen/Qwen3-VL-4B-Instruct [lora_rank]
```

Default `lora_rank` is `128`. The script auto-resumes from the newest checkpoint under `outputs/` that matches the model slug and rank. Pass `FRESH=1` to force a new run instead.

Key training hyperparameters (hardcoded in the script):
- Epochs: `5`
- Learning rate: `2e-5` with cosine scheduler, warmup ratio `0.03`
- Batch: `1` per device × `16` gradient accumulation steps
- LoRA alpha: `128`, dropout: `0.05`
- Frozen: vision encoder + projector; trained: LLM

Output: `outputs/<slug>-lora-<rank>-amba-<timestamp>/`

---

### 5. Evaluate

`AnswerGenerator` queries the model for each item in `final_dataset_test.json` (the SFT test split produced in step 3), then `LLMJudgeEvaluator` scores each answer (0–1, GPT judge). Both steps run in a single script.

The eval input is the SFT test split, not a question set. Each item already carries the GPT-generated gold answer (`conversations[1]["value"]`), which the judge can optionally use for comparison-based scoring via `use_gold_answer=true`.

**Base model:**

```bash
python scripts/evaluation/eval_model.py \
  sft_test_file="data/datasets/amba_document/sft_data/final_dataset_test.json" \
  materials_dir="data/datasets/amba_document/processed/<spec>/materials" \
  chunks_dir="data/datasets/amba_document/processed/<spec>/chunks" \
  output_dir="data/evaluation/base_model/<spec>"
```

**Fine-tuned model:**

```bash
python scripts/evaluation/eval_model.py \
  model=qwen_vl_finetuned_model \
  model.params.finetune_path="outputs/<slug>-lora-<rank>-amba-<timestamp>" \
  sft_test_file="data/datasets/amba_document/sft_data/final_dataset_test.json" \
  materials_dir="data/datasets/amba_document/processed/<spec>/materials" \
  chunks_dir="data/datasets/amba_document/processed/<spec>/chunks" \
  output_dir="data/evaluation/finetuned_model/<spec>"
```

Key options:
- `use_gold_answer=true` — pass the GPT gold answer to the judge for comparison-based scoring; defaults to `false` (judge scores against context only)
- `model.params.model_name` — change model size, e.g. `Qwen/Qwen3-VL-2B-Instruct` (2B/4B/8B/32B)
- `judge.name` — GPT judge model (default: `gpt-5.1`)
- `judge.max_workers` — parallel judge threads (default: `20`)

Output:
- `<output_dir>/preds.json` — model answers with `id`, `type`, `question`, `gold_answer`, `answer`, `source_chunk`, etc.
- `<output_dir>/scores.json` — judge scores with `reason` and `score` (0.0–1.0) added to each item

---

## Hydra Overrides

All scripts support inline config overrides:

```bash
# Single value
python scripts/evaluation/eval_model.py model.params.torch_dtype=float16

# List value (quotes required)
python scripts/augmentation/generate_train_data.py \
  'train_files=[data/datasets/amba_document/question_set/<spec>/a.json,data/datasets/amba_document/question_set/<spec>/b.json]'

# Config group switch
python scripts/evaluation/eval_model.py model=qwen_vl_finetuned_model
```

Hydra logs and merged configs are saved under `data/outputs/YYYY-MM-DD/HH-MM-SS/`.
