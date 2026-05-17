"""Script to merge two LoRA adapters using add_weighted_adapter or slerp."""

import logging
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from mergeeda.merge import LoRAMerger, SlerpMerger

logger = logging.getLogger(__name__)

_SLERP = "slerp"


@hydra.main(
    version_base=None,
    config_path="../../configs/merge",
    config_name="merge_lora",
)
def main(cfg: DictConfig) -> None:
    """Merge LoRA adapters according to the provided configuration."""
    original_cwd = Path(get_original_cwd())

    adapter_paths = [str(original_cwd / p) for p in cfg.adapter_paths]
    output_path = str(original_cwd / cfg.output_path)

    logger.info("combination_type: %s", cfg.combination_type)
    logger.info("adapter_paths: %s", adapter_paths)
    logger.info("output_path: %s", output_path)

    if cfg.combination_type == _SLERP:
        slerp_cfg = cfg.slerp
        t_per_layer_raw = slerp_cfg.get("t_per_layer", None)
        t_per_layer = (
            OmegaConf.to_container(t_per_layer_raw, resolve=True)
            if t_per_layer_raw is not None
            else None
        )
        lora_merge_cache_raw = slerp_cfg.get("lora_merge_cache", None)
        lora_merge_cache = (
            str(original_cwd / lora_merge_cache_raw)
            if lora_merge_cache_raw is not None
            else None
        )
        merger = SlerpMerger(
            base_model_name=cfg.base_model_name,
            adapter_paths=adapter_paths,
            output_path=output_path,
            t=float(slerp_cfg.t),
            t_per_layer=t_per_layer,
            dtype=slerp_cfg.dtype,
            torch_dtype=cfg.torch_dtype,
            attn_implementation=cfg.attn_implementation,
            device_map=cfg.device_map,
            lora_merge_cache=lora_merge_cache,
            copy_tokenizer=bool(slerp_cfg.copy_tokenizer),
            lazy_unpickle=bool(slerp_cfg.lazy_unpickle),
            low_cpu_memory=bool(slerp_cfg.low_cpu_memory),
        )
    else:
        merger = LoRAMerger(
            base_model_name=cfg.base_model_name,
            adapter_paths=adapter_paths,
            adapter_names=list(cfg.adapter_names),
            combination_type=cfg.combination_type,
            weights=list(cfg.weights),
            output_path=output_path,
            density=cfg.density,
            majority_sign_method=cfg.majority_sign_method,
            svd_rank=cfg.svd_rank,
            svd_clamp=cfg.svd_clamp,
            torch_dtype=cfg.torch_dtype,
            attn_implementation=cfg.attn_implementation,
            device_map=cfg.device_map,
        )

    merger.merge()


if __name__ == "__main__":
    main()
