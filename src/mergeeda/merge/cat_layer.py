"""Learnable CAT forward patching for PEFT LoRA Linear layers.

LoRA Soups (arXiv:2410.13025) "CAT" composes several LoRA adapters as

    delta_W = sum_k alpha_k * B_k @ A_k

where ``alpha_k`` are learnable, layer-wise scalars normalized with a softmax
over the active adapters. PEFT's native multi-adapter forward simply sums the
adapter contributions with their static ``scaling``; this module replaces the
forward of each ``lora.Linear`` layer so the contributions are instead combined
with per-layer learnable softmax weights.

The original adapter weights stay frozen; only the injected ``cat_alpha``
parameters are trained.
"""

import logging
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.lora.layer import Linear as LoraLinear

logger = logging.getLogger(__name__)

# Name of the learnable parameter injected into each patched LoRA layer.
CAT_ALPHA_ATTR: str = "cat_alpha"
# Names of bookkeeping attributes injected alongside it.
CAT_ADAPTERS_ATTR: str = "cat_adapter_names"


def _cat_forward(self: LoraLinear, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """Replacement forward for a patched ``lora.Linear`` layer.

    Computes the base layer output plus the softmax-weighted sum of each active
    adapter's LoRA delta. Mirrors PEFT's dtype handling: the LoRA path runs in
    the dtype of ``lora_A`` weights and the result is cast back to ``x``'s dtype.

    Adapter merging / disabling paths are intentionally skipped: CAT never
    merges or disables adapters, so the base layer plus the weighted deltas is
    the complete output.
    """
    # Base (frozen) linear projection. The wrapped layer is a plain nn.Linear,
    # so extra args/kwargs are forwarded verbatim (normally empty).
    result = self.base_layer(x, *args, **kwargs)

    adapter_names: list[str] = getattr(self, CAT_ADAPTERS_ATTR)
    alpha: nn.Parameter = getattr(self, CAT_ALPHA_ATTR)
    # Softmax over the adapter axis -> coefficients in (0, 1) summing to 1.
    weights = F.softmax(alpha, dim=0)

    torch_result_dtype = result.dtype
    for idx, name in enumerate(adapter_names):
        if name not in self.lora_A:
            # Adapter not present on this layer (e.g. target_modules differ).
            continue
        lora_A = self.lora_A[name]
        lora_B = self.lora_B[name]
        dropout = self.lora_dropout[name]
        scaling = self.scaling[name]

        x_cast = x.to(lora_A.weight.dtype)
        delta = lora_B(lora_A(dropout(x_cast))) * scaling
        # weights[idx] is a 0-dim tensor; broadcasts over the delta.
        result = result + weights[idx].to(delta.dtype) * delta

    return result.to(torch_result_dtype)


def patch_lora_layers_for_cat(
    model: nn.Module,
    adapter_names: list[str],
    device: torch.device | str,
    dtype: torch.dtype,
) -> list[nn.Parameter]:
    """Inject learnable ``cat_alpha`` parameters and patch the forward of every
    ``lora.Linear`` layer that hosts at least one of ``adapter_names``.

    Returns the list of injected parameters so the caller can build an optimizer.
    """
    if len(adapter_names) < 2:
        raise ValueError("CAT requires at least two adapter names")

    injected: list[nn.Parameter] = []
    patched_count = 0

    for module in model.modules():
        if not isinstance(module, LoraLinear):
            continue
        # Only patch layers that actually carry the requested adapters.
        present = [n for n in adapter_names if n in module.lora_A]
        if len(present) < 2:
            continue

        # One learnable scalar per adapter, initialized to zeros so the initial
        # softmax is uniform (equal weighting, matching static CAT at start).
        alpha = nn.Parameter(
            torch.zeros(len(adapter_names), device=device, dtype=dtype),
            requires_grad=True,
        )
        module.register_parameter(CAT_ALPHA_ATTR, alpha)
        setattr(module, CAT_ADAPTERS_ATTR, list(adapter_names))

        # Bind the replacement forward to this instance only.
        module.forward = MethodType(_cat_forward, module)

        injected.append(alpha)
        patched_count += 1

    if patched_count == 0:
        raise RuntimeError(
            "No LoRA layers were patched for CAT; check that the adapters "
            f"{adapter_names} are loaded and share target modules"
        )

    logger.info(
        "Patched %d LoRA layers for Learnable CAT (%d adapters each)",
        patched_count,
        len(adapter_names),
    )
    return injected


def collect_cat_alphas(model: nn.Module) -> dict[str, list[float]]:
    """Collect the trained softmax-normalized CAT coefficients per layer.

    The returned dict maps the layer's module name to the post-softmax weights.
    """
    result: dict[str, list[float]] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoraLinear) and hasattr(module, CAT_ALPHA_ATTR):
            alpha = getattr(module, CAT_ALPHA_ATTR)
            weights = F.softmax(alpha.detach().float(), dim=0)
            result[name] = weights.cpu().tolist()
    return result
