"""
Differentiable abliteration via forward hooks.

Makes the model modification differentiable w.r.t. direction weights
so gradients flow: loss -> log_probs -> hook output -> direction -> LearnableDirectionWeights.
"""

from contextlib import suppress
from typing import Any, List

import torch

from heretic.model import Model


def _create_abliteration_hook(
    direction: torch.Tensor,
    weight: float,
):
    """Create a forward hook that projects output onto direction and subtracts."""

    def hook(module, input, output):
        # Under device_map="auto", different layers can live on different GPUs.
        # Keep the differentiable direction tensor aligned with the current
        # module output device/dtype so forward hooks work with sharded models.
        d = direction.to(device=output.device, dtype=output.dtype)
        dots = (output * d).sum(dim=-1, keepdim=True)
        projection = dots * d
        return output - weight * projection

    return hook


def _get_modules_for_hooks(model: Model) -> List[tuple]:
    """
    Get (module, layer_index) for each abliterable component.
    Mirrors heretic's get_layer_matrices structure.
    """
    modules_list = []
    layers = model.get_layers()
    for layer_index in range(len(layers)):
        layer = layers[layer_index]
        # attn.o_proj
        try:
            modules_list.append((layer.self_attn.o_proj, layer_index))
        except Exception:
            pass
        # mlp.down_proj (dense)
        with suppress(Exception):
            modules_list.append((layer.mlp.down_proj, layer_index))
        # MoE experts
        with suppress(Exception):
            for expert in layer.mlp.experts:
                modules_list.append((expert.down_proj, layer_index))
        with suppress(Exception):
            for expert in layer.block_sparse_moe.experts:
                modules_list.append((expert.w2, layer_index))
        with suppress(Exception):
            modules_list.append((layer.mlp.experts.down_proj, layer_index))
        with suppress(Exception):
            modules_list.append((layer.shared_mlp.output_linear, layer_index))
        with suppress(Exception):
            for expert in layer.moe.experts:
                modules_list.append((expert.output_linear, layer_index))
    return modules_list


def register_abliteration_hooks(
    model: Model,
    combined_direction: torch.Tensor,
    alpha: float,
    n_layers: int,
    max_weight: float,
    max_weight_position: float,
    min_weight: float,
    min_weight_distance: float,
) -> List:
    """
    Register forward hooks for differentiable abliteration.

    Args:
        model: Heretic model
        combined_direction: (n_layers+1, hidden_size) - direction per layer
        alpha: scalar coefficient scaling the abliteration
        n_layers: number of transformer layers
        max_weight, max_weight_position, min_weight, min_weight_distance: abliteration params

    Returns:
        List of hook handles for removal.
    """
    max_weight_pos_abs = max_weight_position * (n_layers - 1)
    min_weight_dist_abs = min_weight_distance * (n_layers - 1)
    handles = []
    modules_list = _get_modules_for_hooks(model)
    skipped_distance = 0
    skipped_weight = 0

    for module, layer_index in modules_list:
        distance = abs(layer_index - max_weight_pos_abs)
        if distance > min_weight_dist_abs:
            skipped_distance += 1
            continue
        layer_weight = max_weight + (distance / min_weight_dist_abs) * (min_weight - max_weight)
        effective_weight = alpha * layer_weight
        # Use magnitude so negative coefficients are not silently stripped; the
        # hook output - weight * projection is correct for either sign.
        if abs(effective_weight) < 1e-8:
            skipped_weight += 1
            continue

        direction = combined_direction[layer_index + 1]
        hook_fn = _create_abliteration_hook(direction, effective_weight)
        handle = module.register_forward_hook(hook_fn)
        handles.append(handle)

    print(f"      [hooks] {len(handles)} hooks registered out of {len(modules_list)} modules "
          f"(skipped: {skipped_distance} by distance, {skipped_weight} by weight)")
    return handles


def remove_hooks(handles: List) -> None:
    """Remove all registered hooks."""
    for h in handles:
        h.remove()
