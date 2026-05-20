"""Dataset and collator for Learnable CAT (LoRA Soups) alpha-coefficient training.

The training data is a JSON list of objects in ShareGPT conversation format::

    {"conversations": [{"from": "human", "value": "..."},
                       {"from": "gpt",   "value": "..."}]}

Both skills use a fixed single-turn structure, so the first user turn is taken
as the prompt and the first assistant turn as the response. Each pair is
rendered with the model chat template, and the loss is masked so that only the
response ("gpt") tokens contribute to the loss.
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


class CATConversationDataset(Dataset):
    """ShareGPT-format dataset of single prompt/response pairs.

    Each item yields ``input_ids`` and ``labels`` of equal length. The prompt
    tokens (and chat-template scaffolding) are set to ``IGNORE_INDEX`` in
    ``labels`` so only the response tokens are trained on.
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

        # Each example is a single (prompt, response) pair. Both skills use a
        # fixed 1 human + 1 gpt structure, so only the first user turn and the
        # first assistant turn are kept. Each pair is tokenized once here so
        # samples whose response is fully truncated away (which would make the
        # batch loss NaN: 0 supervised tokens) can be dropped up front.
        self._examples: list[dict[str, list[int]]] = []
        dropped = 0
        for idx, item in enumerate(raw):
            prompt, response = self._extract_pair(item, idx)
            encoded = self._encode(prompt, response)
            if not any(label != IGNORE_INDEX for label in encoded["labels"]):
                dropped += 1
                continue
            self._examples.append(encoded)

        if dropped:
            logger.warning(
                "Dropped %d/%d CAT examples whose response was fully truncated "
                "at max_length=%d",
                dropped,
                len(raw),
                self._max_length,
            )
        if not self._examples:
            raise ValueError(
                f"All CAT examples were dropped; increase max_length "
                f"(current: {self._max_length})"
            )

        logger.info(
            "Loaded %d CAT training examples from %s",
            len(self._examples),
            data_path,
        )

    def _extract_pair(self, item: dict, idx: int) -> tuple[str, str]:
        """Extract the (prompt, response) pair from one ShareGPT item."""
        if self._conversations_key not in item:
            raise ValueError(
                f"Example {idx} is missing '{self._conversations_key}' field"
            )
        raw_turns = item[self._conversations_key]
        if not isinstance(raw_turns, list) or not raw_turns:
            raise ValueError(f"Example {idx} has an empty conversation")

        prompt: str | None = None
        response: str | None = None
        for turn in raw_turns:
            role = _ROLE_MAP.get(str(turn.get("from", "")).lower())
            value = str(turn.get("value", ""))
            if role == "user" and prompt is None:
                prompt = value
            elif role == "assistant" and response is None:
                response = value

        if prompt is None or response is None:
            raise ValueError(
                f"Example {idx} must contain one user and one assistant turn"
            )
        return prompt, response

    def _encode(self, prompt: str, response: str) -> dict[str, list[int]]:
        """Tokenize one (prompt, response) pair into truncated input_ids/labels.

        The prompt span is masked with IGNORE_INDEX; only response tokens carry
        a loss. Truncation is from the right, so input_ids and labels stay
        aligned (and the response may be partially or fully cut off).
        """
        # Render the prompt (with the assistant-turn opening) and the full
        # text separately so the prompt span can be masked exactly.
        prompt_text = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = prompt_text + response + self._tokenizer.eos_token

        prompt_ids = self._tokenizer(
            prompt_text, add_special_tokens=False
        )["input_ids"]
        full_ids = self._tokenizer(
            full_text, add_special_tokens=False
        )["input_ids"]

        labels = list(full_ids)
        # Mask the prompt portion; only the response tokens carry a loss.
        prompt_len = min(len(prompt_ids), len(full_ids))
        for i in range(prompt_len):
            labels[i] = IGNORE_INDEX

        # Truncate from the right; input_ids and labels stay aligned.
        return {
            "input_ids": full_ids[: self._max_length],
            "labels": labels[: self._max_length],
        }

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self._examples[index]
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "labels": torch.tensor(encoded["labels"], dtype=torch.long),
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
