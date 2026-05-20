"""Learnable CAT LoRA merging (LoRA Soups, arXiv:2410.13025).

CAT composes per-skill LoRA adapters as

    delta_W = sum_k alpha_k * scaling_k * B_k @ A_k

where ``alpha_k`` are layer-wise learnable coefficients normalized with a
softmax. The individual adapter weights stay frozen; only the coefficients are
trained, on a small mixed-skill dataset, for a few epochs.

Flow:
  1. Load the base model and every skill LoRA as PEFT multi-adapters.
  2. Patch each LoRA layer's forward to combine adapters via softmax(alpha).
  3. Freeze all weights except the injected ``cat_alpha`` parameters.
  4. Train ``cat_alpha`` on the provided mixed dataset (causal LM loss).
  5. Fold the learned coefficients into the adapter weights and save a single
     rank-(r_1 + ... + r_k) concatenated LoRA adapter, plus the coefficients.
"""

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora.layer import Linear as LoraLinear
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .cat_dataset import CATConversationDataset, CATDataCollator
from .cat_layer import (
    CAT_ALPHA_ATTR,
    collect_cat_alphas,
    patch_lora_layers_for_cat,
)

logger = logging.getLogger(__name__)

# File name for the saved per-layer CAT coefficients.
_ALPHA_FILENAME = "cat_alphas.json"


class CATMerger:
    """Merge LoRA adapters via Learnable CAT and save a concatenated adapter."""

    def __init__(
        self,
        base_model_name: str,
        adapter_paths: list[str],
        adapter_names: list[str],
        output_path: str,
        train_data_path: str,
        epochs: int = 1,
        learning_rate: float = 1e-4,
        max_length: int = 4096,
        batch_size: int = 1,
        grad_accum_steps: int = 8,
        gradient_checkpointing: bool = True,
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        device_map: str = "auto",
        conversations_key: str = "conversations",
        seed: int = 42,
    ) -> None:
        if len(adapter_paths) < 2:
            raise ValueError("CAT requires at least two adapter paths")
        if len(adapter_paths) != len(adapter_names):
            raise ValueError("adapter_paths and adapter_names must have the same length")

        self._base_model_name = base_model_name
        self._adapter_paths = adapter_paths
        self._adapter_names = adapter_names
        self._output_path = Path(output_path)
        self._train_data_path = train_data_path
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._max_length = max_length
        self._batch_size = batch_size
        self._grad_accum_steps = max(1, grad_accum_steps)
        self._gradient_checkpointing = gradient_checkpointing
        self._conversations_key = conversations_key
        self._seed = seed

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self._torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)
        self._attn_implementation = attn_implementation
        self._device_map = device_map

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def merge(self) -> None:
        """Run the full Learnable CAT pipeline and save the merged adapter."""
        torch.manual_seed(self._seed)

        model, processor = self._load_model_with_adapters()
        train_device = self._resolve_train_device(model)

        alpha_params = patch_lora_layers_for_cat(
            model,
            adapter_names=self._adapter_names,
            device=train_device,
            dtype=torch.float32,
            seed=self._seed,
        )
        self._freeze_except_cat(model)

        self._train_alphas(model, processor, alpha_params, train_device)

        alphas = collect_cat_alphas(model)
        self._save_concatenated_adapter(model, processor, alphas)
        logger.info("Learnable CAT merge complete: %s", self._output_path)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _train_device_map(self) -> str | dict:
        """Resolve a device_map safe for training (single device, no offload).

        Sharded ('auto') or CPU-offloaded placements break backward through the
        frozen weights, so training always loads the whole model onto one
        device. Saving (a no-grad path) keeps the user-provided device_map.
        """
        if torch.cuda.is_available():
            return {"": 0}
        return {"": "cpu"}

    def _load_model_with_adapters(
        self,
    ) -> tuple[PeftModel, AutoProcessor]:
        """Load the base model and all skill LoRA adapters as PEFT adapters."""
        logger.info("Loading base model: %s", self._base_model_name)
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            self._base_model_name,
            torch_dtype=self._torch_dtype,
            attn_implementation=self._attn_implementation,
            device_map=self._train_device_map(),
        )

        logger.info(
            "Loading adapter '%s' from: %s",
            self._adapter_names[0],
            self._adapter_paths[0],
        )
        model: PeftModel = PeftModel.from_pretrained(
            base_model,
            self._adapter_paths[0],
            adapter_name=self._adapter_names[0],
        )
        for path, name in zip(self._adapter_paths[1:], self._adapter_names[1:]):
            logger.info("Loading adapter '%s' from: %s", name, path)
            model.load_adapter(path, adapter_name=name)

        # Activate every adapter so all of them run in the patched forward.
        model.base_model.set_adapter(self._adapter_names)

        processor = AutoProcessor.from_pretrained(self._base_model_name)
        return model, processor

    @staticmethod
    def _resolve_train_device(model: torch.nn.Module) -> torch.device:
        """Pick the device that holds the model parameters for training."""
        for param in model.parameters():
            return param.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @staticmethod
    def _freeze_except_cat(model: torch.nn.Module) -> None:
        """Freeze every parameter except the injected CAT alpha coefficients."""
        trainable = 0
        for name, param in model.named_parameters():
            if name.endswith(CAT_ALPHA_ATTR) or f".{CAT_ALPHA_ATTR}" in name:
                param.requires_grad_(True)
                trainable += param.numel()
            else:
                param.requires_grad_(False)
        logger.info("CAT trainable parameters (alpha coefficients): %d", trainable)

    def _train_alphas(
        self,
        model: torch.nn.Module,
        processor: AutoProcessor,
        alpha_params: list[torch.nn.Parameter],
        device: torch.device,
    ) -> None:
        """Train the CAT alpha coefficients with a causal LM loss."""
        tokenizer = getattr(processor, "tokenizer", processor)

        dataset = CATConversationDataset(
            data_path=self._train_data_path,
            tokenizer=tokenizer,
            max_length=self._max_length,
            conversations_key=self._conversations_key,
        )
        collator = CATDataCollator(tokenizer)
        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=True,
            collate_fn=collator,
        )

        optimizer = torch.optim.AdamW(alpha_params, lr=self._learning_rate)

        if self._gradient_checkpointing:
            # use_cache=True caches attention key/values for generation and is
            # incompatible with gradient checkpointing; disable it for training.
            model.config.use_cache = False
            # Only frozen weights exist besides alpha, so the embedding output
            # would carry no grad and checkpointed blocks could not backprop.
            # enable_input_require_grads() forces grad on the embeddings, and
            # use_reentrant=False correctly tracks the alpha closure variables.
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        model.train()
        # The base/adapter weights are frozen, but gradients must still flow
        # through them to reach the alpha parameters.
        for epoch in range(self._epochs):
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            progress = tqdm(
                loader,
                desc=f"CAT epoch {epoch + 1}/{self._epochs}",
                unit="batch",
            )
            for step, batch in enumerate(progress):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss / self._grad_accum_steps
                loss.backward()
                running_loss += outputs.loss.item()

                if (step + 1) % self._grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                progress.set_postfix(avg_loss=running_loss / (step + 1))

            # Flush a trailing partial accumulation window.
            if len(loader) % self._grad_accum_steps != 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            avg = running_loss / max(1, len(loader))
            logger.info("epoch %d finished: avg_loss=%.4f", epoch + 1, avg)

        model.eval()
        if self._gradient_checkpointing:
            model.gradient_checkpointing_disable()

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _save_concatenated_adapter(
        self,
        model: PeftModel,
        processor: AutoProcessor,
        alphas: dict[str, list[float]],
    ) -> None:
        """Fold the learned coefficients into a single concatenated LoRA adapter.

        For each LoRA layer the merged delta is

            delta_W = sum_k w_k * scaling_k * B_k @ A_k

        which is reproduced exactly by a concatenated adapter with
            A_cat = [ (w_0*scaling_0)*A_0 ; ... ; (w_{k-1}*scaling_{k-1})*A_{k-1} ]
            B_cat = [ B_0 | ... | B_{k-1} ]
        and a unit scaling (lora_alpha == r) so PEFT applies no extra factor.
        """
        per_layer = self._compute_concat_weights(model)

        merged_rank = self._validate_uniform_rank(per_layer)
        target_modules = sorted({name.split(".")[-1] for name in per_layer})

        logger.info(
            "Building concatenated adapter: rank=%d, target_modules=%s",
            merged_rank,
            target_modules,
        )

        # A fresh PEFT model on a clean base, with a single 'cat' adapter whose
        # scaling is 1.0 (lora_alpha == r). We then overwrite its A/B weights.
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            self._base_model_name,
            torch_dtype=self._torch_dtype,
            attn_implementation=self._attn_implementation,
            device_map=self._device_map,
        )
        cat_config = LoraConfig(
            r=merged_rank,
            lora_alpha=merged_rank,  # scaling = lora_alpha / r = 1.0
            lora_dropout=0.0,
            target_modules=target_modules,
            bias="none",
        )
        cat_model = get_peft_model(base_model, cat_config, adapter_name="cat")

        self._load_concat_weights(cat_model, per_layer)

        self._output_path.mkdir(parents=True, exist_ok=True)
        logger.info("Saving concatenated CAT adapter to: %s", self._output_path)
        cat_model.save_pretrained(str(self._output_path), selected_adapters=["cat"])

        # Persist the per-layer softmax coefficients for inspection / eval.
        alpha_file = self._output_path / _ALPHA_FILENAME
        with alpha_file.open("w", encoding="utf-8") as f:
            json.dump(
                {"adapter_names": self._adapter_names, "per_layer": alphas},
                f,
                indent=2,
            )
        logger.info("Saved CAT coefficients to: %s", alpha_file)

        try:
            processor.save_pretrained(str(self._output_path))
        except Exception:
            logger.warning("Could not save processor to %s", self._output_path)

    def _compute_concat_weights(
        self, model: PeftModel
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Build concatenated A_cat / B_cat tensors for every patched layer.

        Returns a mapping ``layer_name -> {"A": A_cat, "B": B_cat}`` where the
        learned softmax weight and the original per-adapter scaling are folded
        into A_cat.
        """
        per_layer: dict[str, dict[str, torch.Tensor]] = {}

        for name, module in model.named_modules():
            if not isinstance(module, LoraLinear):
                continue
            if not hasattr(module, CAT_ALPHA_ATTR):
                continue

            alpha = getattr(module, CAT_ALPHA_ATTR)
            weights = F.softmax(alpha.detach().float(), dim=0)

            a_blocks: list[torch.Tensor] = []
            b_blocks: list[torch.Tensor] = []
            for idx, adapter in enumerate(self._adapter_names):
                if adapter not in module.lora_A:
                    continue
                # lora_A: Linear(in -> r), weight shape (r, in)
                # lora_B: Linear(r -> out), weight shape (out, r)
                a_w = module.lora_A[adapter].weight.detach().float()
                b_w = module.lora_B[adapter].weight.detach().float()
                scaling = float(module.scaling[adapter])
                coeff = float(weights[idx]) * scaling
                a_blocks.append(a_w * coeff)
                b_blocks.append(b_w)

            # A_cat: (sum_r, in), B_cat: (out, sum_r)
            a_cat = torch.cat(a_blocks, dim=0)
            b_cat = torch.cat(b_blocks, dim=1)
            per_layer[name] = {"A": a_cat, "B": b_cat}

        return per_layer

    @staticmethod
    def _validate_uniform_rank(
        per_layer: dict[str, dict[str, torch.Tensor]],
    ) -> int:
        """Ensure every concatenated layer has the same rank and return it.

        PEFT's LoraConfig uses a single ``r`` for all target modules, so all
        concatenated adapters must share the same total rank. This holds when
        every skill LoRA uses the same rank across modules (the project trains
        all adapters with a single ``lora_r``).
        """
        ranks = {tensors["A"].size(0) for tensors in per_layer.values()}
        if len(ranks) != 1:
            raise ValueError(
                f"Concatenated CAT adapter has non-uniform ranks {sorted(ranks)}; "
                "all skill LoRAs must share the same rank across target modules"
            )
        return ranks.pop()

    def _load_concat_weights(
        self,
        cat_model: PeftModel,
        per_layer: dict[str, dict[str, torch.Tensor]],
    ) -> None:
        """Copy the precomputed A_cat / B_cat tensors into the fresh CAT model.

        Layer names from the source model are matched against the fresh model;
        both wrap the same base architecture so the module paths are identical.
        """
        target = dict(cat_model.named_modules())
        copied = 0

        for name, tensors in per_layer.items():
            if name not in target:
                raise RuntimeError(
                    f"Layer '{name}' from the trained model is missing in the "
                    "concatenated CAT model"
                )
            module = target[name]
            if not isinstance(module, LoraLinear) or "cat" not in module.lora_A:
                raise RuntimeError(
                    f"Layer '{name}' in the CAT model has no 'cat' LoRA adapter"
                )

            a_dst = module.lora_A["cat"].weight
            b_dst = module.lora_B["cat"].weight
            a_src = tensors["A"].to(device=a_dst.device, dtype=a_dst.dtype)
            b_src = tensors["B"].to(device=b_dst.device, dtype=b_dst.dtype)

            if a_src.shape != a_dst.shape or b_src.shape != b_dst.shape:
                raise RuntimeError(
                    f"Shape mismatch at '{name}': "
                    f"A {tuple(a_src.shape)} vs {tuple(a_dst.shape)}, "
                    f"B {tuple(b_src.shape)} vs {tuple(b_dst.shape)}"
                )

            with torch.no_grad():
                a_dst.copy_(a_src)
                b_dst.copy_(b_src)
            copied += 1

        logger.info("Loaded concatenated weights into %d CAT layers", copied)
