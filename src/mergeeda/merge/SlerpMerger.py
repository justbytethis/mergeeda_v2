"""LoRA adapter slerp merging via mergekit.

Flow:
  1. Each LoRA adapter is merged into the base model with merge_and_unload,
     producing two full model checkpoints in a temporary directory.
  2. mergekit run_merge applies spherical linear interpolation (slerp) between
     the two full models.
  3. The output is a full merged model (not a LoRA adapter).

Note: slerp requires exactly 2 adapters.
"""

import logging
import shutil
import tempfile
from pathlib import Path

import torch
import yaml
from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

logger = logging.getLogger(__name__)


class SlerpMerger:
    """Merge two LoRA adapters into a full model using slerp via mergekit."""

    def __init__(
        self,
        base_model_name: str,
        adapter_paths: list[str],
        output_path: str,
        t: float = 0.5,
        t_per_layer: list[dict] | None = None,
        dtype: str = "bfloat16",
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        device_map: str = "auto",
        lora_merge_cache: str | None = None,
        copy_tokenizer: bool = True,
        lazy_unpickle: bool = False,
        low_cpu_memory: bool = False,
    ) -> None:
        if len(adapter_paths) != 2:
            raise ValueError(
                "slerp requires exactly 2 adapter_paths "
                f"(got {len(adapter_paths)})"
            )

        self._base_model_name = base_model_name
        self._adapter_paths = [Path(p) for p in adapter_paths]
        self._output_path = Path(output_path)

        # slerp interpolation: t_per_layer overrides scalar t when provided
        self._t = t
        self._t_per_layer = t_per_layer

        self._dtype = dtype  # mergekit output dtype

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self._torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)
        self._attn_implementation = attn_implementation
        self._device_map = device_map

        self._lora_merge_cache = lora_merge_cache
        self._copy_tokenizer = copy_tokenizer
        self._lazy_unpickle = lazy_unpickle
        self._low_cpu_memory = low_cpu_memory

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def merge(self) -> None:
        """Merge two LoRA adapters with slerp and save the full model."""
        with tempfile.TemporaryDirectory(prefix="mergeeda_slerp_") as tmp_dir:
            tmp = Path(tmp_dir)
            full_model_paths = self._unload_adapters(tmp)
            self._run_slerp(full_model_paths)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _unload_adapters(self, tmp: Path) -> list[str]:
        """Merge each LoRA adapter into the base model and save full weights."""
        logger.info("Loading base model for adapter unloading: %s", self._base_model_name)
        full_model_paths: list[str] = []

        for idx, adapter_path in enumerate(self._adapter_paths):
            logger.info(
                "Merging adapter %d/%d into base model: %s",
                idx + 1,
                len(self._adapter_paths),
                adapter_path,
            )
            base = Qwen3VLForConditionalGeneration.from_pretrained(
                self._base_model_name,
                torch_dtype=self._torch_dtype,
                attn_implementation=self._attn_implementation,
                device_map=self._device_map,
            )
            model = PeftModel.from_pretrained(base, str(adapter_path))
            merged = model.merge_and_unload()

            out = tmp / f"model_{idx}"
            logger.info("Saving merged full model to: %s", out)
            merged.save_pretrained(str(out))

            # Save processor/tokenizer so mergekit can copy it
            try:
                processor = AutoProcessor.from_pretrained(self._base_model_name)
                processor.save_pretrained(str(out))
            except Exception:
                logger.warning(
                    "Could not save processor from %s; tokenizer copy may be skipped.",
                    self._base_model_name,
                )

            full_model_paths.append(str(out))

            # Free VRAM before loading the next adapter
            del merged, model, base
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return full_model_paths

    def _build_mergekit_config(self, full_model_paths: list[str]) -> MergeConfiguration:
        """Build a mergekit MergeConfiguration for slerp."""
        # t_per_layer allows per-layer / per-filter interpolation schedules
        t_value: float | list[dict] = (
            self._t_per_layer if self._t_per_layer is not None else self._t
        )

        config_dict = {
            "merge_method": "slerp",
            "base_model": full_model_paths[0],
            "models": [{"model": full_model_paths[1]}],
            "parameters": {
                "t": t_value,
                "dtype": self._dtype,
            },
        }
        raw_yaml = yaml.dump(config_dict, allow_unicode=True)
        logger.debug("mergekit config:\n%s", raw_yaml)
        return MergeConfiguration.model_validate(yaml.safe_load(raw_yaml))

    def _run_slerp(self, full_model_paths: list[str]) -> None:
        """Invoke mergekit run_merge with slerp configuration."""
        merge_cfg = self._build_mergekit_config(full_model_paths)

        self._output_path.mkdir(parents=True, exist_ok=True)

        options = MergeOptions(
            lora_merge_cache=self._lora_merge_cache,
            cuda=torch.cuda.is_available(),
            copy_tokenizer=self._copy_tokenizer,
            lazy_unpickle=self._lazy_unpickle,
            low_cpu_memory=self._low_cpu_memory,
        )

        logger.info(
            "Running slerp merge (t=%s) -> %s",
            self._t_per_layer if self._t_per_layer is not None else self._t,
            self._output_path,
        )
        run_merge(merge_cfg, out_path=str(self._output_path), options=options)
        logger.info("Slerp merge complete: %s", self._output_path)
