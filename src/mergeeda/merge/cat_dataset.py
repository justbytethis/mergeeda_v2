"""Dataset and collator for Learnable CAT (LoRA Soups) alpha-coefficient training.

The training data is a JSON list of objects with at least ``instruction`` and
``reference_code`` fields. Each example is rendered with the model chat template
as a single user/assistant turn, and the loss is masked so that only the
assistant response (``reference_code``) tokens contribute to the loss.
"""

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Label value ignored by the cross-entropy loss.
IGNORE_INDEX: int = -100


class CATInstructionDataset(Dataset):
    """Instruction-tuning dataset that masks the loss to the assistant response.

    Each item yields ``input_ids`` and ``labels`` of equal length. Tokens that
    belong to the prompt (system/user turn and chat-template scaffolding) are
    set to ``IGNORE_INDEX`` in ``labels`` so only the response is trained on.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        instruction_key: str = "instruction",
        response_key: str = "reference_code",
    ) -> None:
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._instruction_key = instruction_key
        self._response_key = response_key

        data_path = Path(data_path)
        if not data_path.is_file():
            raise FileNotFoundError(f"CAT training data not found: {data_path}")

        with data_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError(
                f"CAT training data must be a JSON list, got {type(raw).__name__}"
            )

        self._examples: list[dict] = []
        for idx, item in enumerate(raw):
            if instruction_key not in item or response_key not in item:
                raise ValueError(
                    f"Example {idx} is missing '{instruction_key}' or "
                    f"'{response_key}' field"
                )
            self._examples.append(item)

        logger.info(
            "Loaded %d CAT training examples from %s",
            len(self._examples),
            data_path,
        )

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self._examples[index]
        instruction = str(example[self._instruction_key])
        response = str(example[self._response_key])

        # Render the prompt up to (but excluding) the assistant response so we
        # know exactly how many tokens to mask. add_generation_prompt=True
        # appends the assistant-turn opening tokens.
        prompt_text = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        # Full text = prompt + response + EOS.
        full_text = prompt_text + response + self._tokenizer.eos_token

        prompt_ids = self._tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]
        full_ids = self._tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self._max_length,
        )["input_ids"]

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()

        # Mask the prompt portion; clamp in case truncation cut into the prompt.
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels[:prompt_len] = IGNORE_INDEX

        return {"input_ids": input_ids, "labels": labels}


class CATDataCollator:
    """Pad a batch of ``input_ids``/``labels`` and build the attention mask."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError(
                "Tokenizer has neither pad_token_id nor eos_token_id; "
                "cannot pad CAT training batches"
            )
        self._pad_id = pad_id

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(f["input_ids"].size(0) for f in features)

        input_ids_batch: list[torch.Tensor] = []
        labels_batch: list[torch.Tensor] = []
        attention_batch: list[torch.Tensor] = []

        for f in features:
            ids = f["input_ids"]
            labels = f["labels"]
            pad_len = max_len - ids.size(0)

            input_ids_batch.append(
                torch.cat([ids, torch.full((pad_len,), self._pad_id, dtype=torch.long)])
            )
            labels_batch.append(
                torch.cat([labels, torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)])
            )
            attention_batch.append(
                torch.cat([
                    torch.ones(ids.size(0), dtype=torch.long),
                    torch.zeros(pad_len, dtype=torch.long),
                ])
            )

        return {
            "input_ids": torch.stack(input_ids_batch),
            "labels": torch.stack(labels_batch),
            "attention_mask": torch.stack(attention_batch),
        }
