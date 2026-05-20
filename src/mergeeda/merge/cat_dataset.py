"""Dataset and collator for Learnable CAT (LoRA Soups) alpha-coefficient training.

The training data is a JSON list of objects in ShareGPT conversation format::

    {"conversations": [{"from": "human", "value": "..."},
                       {"from": "gpt",   "value": "..."}]}

Each conversation is rendered with the model chat template, and the loss is
masked so that only the assistant ("gpt") turns contribute to the loss.
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

# ShareGPT role -> chat-template role mapping.
_ROLE_MAP: dict[str, str] = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}
# Roles whose tokens are trained on (loss is computed).
_RESPONSE_ROLES: frozenset[str] = frozenset({"assistant"})


class CATConversationDataset(Dataset):
    """ShareGPT-format dataset that masks the loss to the assistant turns.

    Each item yields ``input_ids`` and ``labels`` of equal length. Tokens that
    belong to non-assistant turns (and chat-template scaffolding) are set to
    ``IGNORE_INDEX`` in ``labels`` so only the responses are trained on.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        conversations_key: str = "conversations",
    ) -> None:
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._conversations_key = conversations_key

        data_path = Path(data_path)
        if not data_path.is_file():
            raise FileNotFoundError(f"CAT training data not found: {data_path}")

        with data_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError(
                f"CAT training data must be a JSON list, got {type(raw).__name__}"
            )

        self._examples: list[list[dict[str, str]]] = []
        for idx, item in enumerate(raw):
            turns = self._normalize_conversation(item, idx)
            if turns:
                self._examples.append(turns)

        if not self._examples:
            raise ValueError(f"No usable conversations found in {data_path}")

        logger.info(
            "Loaded %d CAT training conversations from %s",
            len(self._examples),
            data_path,
        )

    def _normalize_conversation(
        self, item: dict, idx: int
    ) -> list[dict[str, str]]:
        """Convert one raw item into a list of {role, content} chat turns."""
        if self._conversations_key not in item:
            raise ValueError(
                f"Example {idx} is missing '{self._conversations_key}' field"
            )
        raw_turns = item[self._conversations_key]
        if not isinstance(raw_turns, list) or not raw_turns:
            raise ValueError(f"Example {idx} has an empty conversation")

        turns: list[dict[str, str]] = []
        for turn in raw_turns:
            role_raw = str(turn.get("from", "")).lower()
            role = _ROLE_MAP.get(role_raw)
            if role is None:
                raise ValueError(
                    f"Example {idx} has an unknown role '{role_raw}'"
                )
            turns.append({"role": role, "content": str(turn.get("value", ""))})
        return turns

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        turns = self._examples[index]

        input_ids: list[int] = []
        labels: list[int] = []

        # Render turn-by-turn so the assistant spans can be located exactly.
        # The prompt prefix before each turn (including prior turns) is masked;
        # only the newly added assistant tokens carry a loss.
        for i, turn in enumerate(turns):
            prefix_text = self._tokenizer.apply_chat_template(
                turns[:i],
                tokenize=False,
                add_generation_prompt=(turn["role"] == "assistant"),
            )
            upto_text = self._tokenizer.apply_chat_template(
                turns[: i + 1],
                tokenize=False,
                add_generation_prompt=False,
            )
            prefix_ids = self._tokenizer(
                prefix_text, add_special_tokens=False
            )["input_ids"]
            upto_ids = self._tokenizer(
                upto_text, add_special_tokens=False
            )["input_ids"]

            # Tokens added by this turn.
            new_ids = upto_ids[len(prefix_ids):]
            input_ids = upto_ids  # cumulative
            if turn["role"] in _RESPONSE_ROLES:
                labels = labels + new_ids
            else:
                labels = labels + [IGNORE_INDEX] * len(new_ids)

        # Truncate from the right; input_ids and labels stay aligned.
        input_ids = input_ids[: self._max_length]
        labels = labels[: self._max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


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
