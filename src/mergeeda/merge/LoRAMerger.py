"""LoRA adapter merging using PEFT add_weighted_adapter."""

import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration

logger = logging.getLogger(__name__)

# combination_types that require each optional parameter
_NEEDS_DENSITY = {
    "ties", "ties_svd",
    "dare_ties", "dare_linear", "dare_ties_svd", "dare_linear_svd",
    "magnitude_prune", "magnitude_prune_svd",
}
_NEEDS_MAJORITY_SIGN = {"ties", "ties_svd", "dare_ties", "dare_ties_svd"}
_NEEDS_SVD = {"svd", "ties_svd", "dare_ties_svd", "dare_linear_svd", "magnitude_prune_svd"}


class LoRAMerger:
    """Merge multiple LoRA adapters into a single adapter using add_weighted_adapter."""

    def __init__(
        self,
        base_model_name: str,
        adapter_paths: list[str],
        adapter_names: list[str],
        combination_type: str,
        weights: list[float],
        output_path: str,
        density: float | None = None,
        majority_sign_method: str = "total",
        svd_rank: int | None = None,
        svd_clamp: float | None = None,
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        device_map: str = "auto",
    ) -> None:
        if len(adapter_paths) < 2:
            raise ValueError("adapter_paths must contain at least two adapter paths")
        if len(adapter_paths) != len(adapter_names):
            raise ValueError("adapter_paths and adapter_names must have the same length")
        if len(adapter_paths) != len(weights):
            raise ValueError("adapter_paths and weights must have the same length")

        self._base_model_name = base_model_name
        self._adapter_paths = adapter_paths
        self._adapter_names = adapter_names
        self._combination_type = combination_type
        self._weights = weights
        self._output_path = Path(output_path)
        self._density = density
        self._majority_sign_method = majority_sign_method
        self._svd_rank = svd_rank
        self._svd_clamp = svd_clamp

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self._torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)
        self._attn_implementation = attn_implementation
        self._device_map = device_map

    def merge(self) -> None:
        """Load adapters, merge via add_weighted_adapter, and save the result."""
        logger.info("Loading base model: %s", self._base_model_name)
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            self._base_model_name,
            torch_dtype=self._torch_dtype,
            attn_implementation=self._attn_implementation,
            device_map=self._device_map,
        )

        logger.info(
            "Loading adapter '%s' from: %s",
            self._adapter_names[0],
            self._adapter_paths[0],
        )
        model = PeftModel.from_pretrained(
            base_model,
            self._adapter_paths[0],
            adapter_name=self._adapter_names[0],
        )

        for path, name in zip(self._adapter_paths[1:], self._adapter_names[1:]):
            logger.info("Loading adapter '%s' from: %s", name, path)
            model.load_adapter(path, adapter_name=name)

        kwargs = self._build_adapter_kwargs()
        logger.info(
            "Merging adapters %s with combination_type='%s', weights=%s, kwargs=%s",
            self._adapter_names,
            self._combination_type,
            self._weights,
            kwargs,
        )
        model.add_weighted_adapter(
            adapters=self._adapter_names,
            weights=self._weights,
            adapter_name="merged",
            combination_type=self._combination_type,
            **kwargs,
        )

        model.set_adapter("merged")

        self._output_path.mkdir(parents=True, exist_ok=True)
        logger.info("Saving merged adapter to: %s", self._output_path)
        model.save_pretrained(str(self._output_path))
        logger.info("Merge complete")

    def _build_adapter_kwargs(self) -> dict:
        """Build keyword arguments for add_weighted_adapter based on combination_type."""
        kwargs: dict = {}

        if self._combination_type in _NEEDS_DENSITY:
            if self._density is None:
                raise ValueError(
                    f"density is required for combination_type='{self._combination_type}'"
                )
            kwargs["density"] = self._density

        if self._combination_type in _NEEDS_MAJORITY_SIGN:
            kwargs["majority_sign_method"] = self._majority_sign_method

        if self._combination_type in _NEEDS_SVD:
            if self._svd_rank is not None:
                kwargs["svd_rank"] = self._svd_rank
            if self._svd_clamp is not None:
                kwargs["svd_clamp"] = self._svd_clamp

        return kwargs
