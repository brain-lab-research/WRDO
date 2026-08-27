#!/usr/bin/env python3
"""
RDO baseline — optimization core.

Port of upstream RDO (https://github.com/wollschlager/geometry-of-refusal,
`rdo.py:refusal_cone_optimization` with `cone_dim=1`). Replaces nnsight
activation interventions with native PyTorch forward hooks. The trained
direction is later applied via the local `heretic.model.Model.abliterate`
through `baselines.rdo`.

Public exports:
- `pick_seed_direction(...)` — thin wrapper around `basic_refusal.select_direction`
- `generate_intervention_targets(...)` — one-shot target generation for three kinds
- `build_rdo_dataset(...)`, `rdo_collate(...)` — port of upstream Dataset+collate
- `train_refusal_direction(...)` — gradient-descent loop with tangent projection
- `installed_ablation_hooks`, `installed_addition_hook` — context managers
"""

from __future__ import annotations

import math
import logging
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from heretic.model import Model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Projection helper (port of upstream `projection_einops`)
# ---------------------------------------------------------------------------


def _projection(x: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Project ``x`` onto the line spanned by ``direction`` (unit norm).

    ``x`` has shape (..., hidden); ``direction`` has shape (hidden,).
    Returns a tensor with the same shape and dtype as ``x``.
    Gradient-friendly: no in-place ops, so backprop through ``direction``
    works. Inner product is accumulated in fp32 for numerical stability
    when ``x`` is bf16/fp16, then cast back to ``x.dtype``.
    """
    orig_dtype = x.dtype
    x32 = x.to(torch.float32)
    d32 = direction.to(torch.float32)
    coeff = (x32 * d32).sum(dim=-1, keepdim=True)
    return (coeff * d32).to(orig_dtype)


def _unit_dir(v: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Return v / ||v|| cast to ``dtype`` while preserving grad through v.

    Normalization is done in v's native dtype (typically fp32). The cast at
    the end is differentiable: autograd upcasts the gradient back to v's
    dtype during backprop.
    """
    return (v / v.norm()).to(dtype)


# ---------------------------------------------------------------------------
# Hook installers
# ---------------------------------------------------------------------------


def _replace_first(tup: tuple, new_first: torch.Tensor) -> tuple:
    return (new_first,) + tuple(tup[1:])


def _make_layer_ablation_pre_hook(v_param: torch.Tensor, coefficient: float = 1.0) -> Callable:
    def hook(module, args):
        if not args:
            return None
        x = args[0]
        if not torch.is_tensor(x):
            return None
        v_norm = _unit_dir(v_param, x.dtype)
        new_x = x - coefficient * _projection(x, v_norm)
        return _replace_first(args, new_x)

    return hook


def _make_submodule_ablation_post_hook(v_param: torch.Tensor, coefficient: float = 1.0) -> Callable:
    def hook(module, args, output):
        if isinstance(output, tuple):
            attn_out = output[0]
            if not torch.is_tensor(attn_out):
                return output
            v_norm = _unit_dir(v_param, attn_out.dtype)
            new_attn = attn_out - coefficient * _projection(attn_out, v_norm)
            return _replace_first(output, new_attn)
        if torch.is_tensor(output):
            v_norm = _unit_dir(v_param, output.dtype)
            return output - coefficient * _projection(output, v_norm)
        return output

    return hook


def _make_addition_pre_hook(v_param: torch.Tensor, alpha: float) -> Callable:
    def hook(module, args):
        if not args:
            return None
        x = args[0]
        if not torch.is_tensor(x):
            return None
        v_norm = _unit_dir(v_param, x.dtype)
        return _replace_first(args, x + alpha * v_norm)

    return hook


@contextmanager
def installed_ablation_hooks(
    layers: Sequence[nn.Module],
    v_param: torch.Tensor,
    coefficient: float = 1.0,
) -> Iterator[None]:
    """Install ablation hooks at three sites per layer (layer.input,
    layer.self_attn.output, layer.mlp.output). Mirrors upstream
    `intervene_with_fn_vector_ablation`.

    ``coefficient`` scales the projection that gets subtracted:
      - 1.0 (default): full projection — matches upstream RDO training.
      - >1.0: over-ablate (subtract more than the projection).
      - <0:  add the v-direction back, analogous to negative weights in the
              heretic.Model.abliterate path.
    """
    handles: list[torch.utils.hooks.RemovableHandle] = []
    try:
        for layer in layers:
            handles.append(layer.register_forward_pre_hook(
                _make_layer_ablation_pre_hook(v_param, coefficient)
            ))
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                handles.append(attn.register_forward_hook(
                    _make_submodule_ablation_post_hook(v_param, coefficient)
                ))
            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                handles.append(mlp.register_forward_hook(
                    _make_submodule_ablation_post_hook(v_param, coefficient)
                ))
        yield
    finally:
        for h in handles:
            h.remove()


def orthogonalize_embedding(model: Model, v: torch.Tensor) -> int:
    """Project ``v`` out of the input token embedding matrix in-place.

    Closes the Arditi (2024) §5.2 equivalence gap: when used together with
    `heretic.Model.abliterate` modifying every layer's `attn.o_proj` and
    `mlp.down_proj`, the resulting weight-modified model has *no* component
    along `v` written into the residual stream by any source. Hence it
    behaves identically to runtime activation-space ablation at every layer.

    Returns the vocabulary size (number of embedding rows modified). Note
    that if the model has tied input/output embeddings, this also affects
    `lm_head.weight` — which is desired (it makes the readout orthogonal
    to `v` too, consistent with "v has no influence anywhere").
    """
    emb = model.model.get_input_embeddings()
    W = emb.weight
    with torch.no_grad():
        v_unit = (v / v.norm()).to(W.device, W.dtype)
        # (vocab, hidden) @ (hidden,) -> (vocab,)
        proj_coeffs = W @ v_unit
        # outer product, broadcast back to (vocab, hidden)
        W.sub_(proj_coeffs.unsqueeze(-1) * v_unit)
    return int(W.shape[0])


@contextmanager
def installed_addition_hook(layer: nn.Module, v_param: torch.Tensor, alpha: float) -> Iterator[None]:
    """Install addition pre-hook on a single layer (the seed `best_layer`)."""
    handle = layer.register_forward_pre_hook(_make_addition_pre_hook(v_param, alpha))
    try:
        yield
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# Seed-direction selection (delegates to basic_refusal.select_direction)
# ---------------------------------------------------------------------------


def pick_seed_direction(
    model: Model,
    *,
    mean_diffs: torch.Tensor,
    harmful_val: Sequence[str],
    harmless_val: Sequence[str],
    token_positions: Sequence[int],
    token_labels: Sequence[str],
    refusal_token_ids: Sequence[int],
    selection_config: dict[str, Any],
    abliteration_params: dict[str, float],
    selection_run_dir,
) -> tuple[torch.Tensor, int, float, dict[str, Any], dict[str, str]]:
    """Pick the seed (v_seed, best_layer, alpha) via basic_refusal.select_direction.

    Returns:
        v_seed: unit-norm direction at the best (pos, layer).
        best_layer: layer index for the addition-loss intervention.
        alpha: pre-normalization norm of the mean-diff vector.
        selection_metadata: dict from select_direction.
        selection_artifacts: paths dict from select_direction.
    """
    # Late import to avoid circular import (basic_refusal imports model_utils
    # which is fine, but we keep the optim module light at module-load).
    from baselines.basic_refusal import normalize_candidate_direction, select_direction

    selection_metadata, selection_artifacts = select_direction(
        model,
        list(harmful_val),
        list(harmless_val),
        mean_diffs,
        token_positions=token_positions,
        token_labels=token_labels,
        refusal_token_ids=refusal_token_ids,
        selection_config=selection_config,
        abliteration_params=abliteration_params,
        selection_run_dir=selection_run_dir,
    )
    best_pos_idx = int(selection_metadata["selected_position_index"])
    best_layer = int(selection_metadata["selected_layer"])
    raw = mean_diffs[best_pos_idx, best_layer].to(torch.float32)
    alpha = float(raw.norm().item())
    v_seed = normalize_candidate_direction(raw)
    return v_seed, best_layer, alpha, selection_metadata, selection_artifacts


# ---------------------------------------------------------------------------
# Target generation (one-shot, three kinds)
# ---------------------------------------------------------------------------


def _batched(items: Sequence[Any], batch_size: int) -> Iterator[list[Any]]:
    batch_size = max(int(batch_size), 1)
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def _greedy_generate(
    model: Model,
    chat_prompts: Sequence[str],
    *,
    num_target_tokens: int,
    batch_size: int,
) -> list[str]:
    """Run greedy generation on ALREADY chat-formatted prompts.

    Bypasses `Model.generate` because that re-applies the chat template; we
    operate directly on the wrapped HF model so hooks installed on
    `model.model.layers[...]` take effect.
    """
    tokenizer = model.tokenizer
    hf_model = model.model
    completions: list[str] = []
    for batch in _batched(chat_prompts, batch_size):
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
            add_special_tokens=False,  # chat template already includes BOS
        ).to(hf_model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = hf_model.generate(
                **inputs,
                max_new_tokens=int(num_target_tokens),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = outputs[:, input_len:]
        completions.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return completions


def generate_intervention_targets(
    model: Model,
    chat_prompts: Sequence[str],
    *,
    kind: str,
    v_seed: torch.Tensor,
    alpha: float,
    best_layer: int,
    num_target_tokens: int,
    batch_size: int,
) -> list[str]:
    """Generate `num_target_tokens`-long completions with one of three
    interventions:
      - "ablation": activation-space ablation at every layer (3 sites).
      - "addition": add `alpha * v_seed` at `best_layer` input.
      - "retain": no intervention (baseline greedy completions).
    Returns plain strings.
    """
    if kind not in {"ablation", "addition", "retain"}:
        raise ValueError(f"unknown intervention kind {kind!r}")

    if kind == "retain":
        return _greedy_generate(
            model, chat_prompts,
            num_target_tokens=num_target_tokens, batch_size=batch_size,
        )

    layers = list(model.get_layers())
    # Use a detached buffer for generation — no autograd during inference.
    v_buf = v_seed.detach().to(model.model.device, dtype=torch.float32).clone()

    with ExitStack() as stack:
        if kind == "ablation":
            stack.enter_context(installed_ablation_hooks(layers, v_buf))
        else:  # addition
            stack.enter_context(installed_addition_hook(layers[best_layer], v_buf, alpha))
        return _greedy_generate(
            model, chat_prompts,
            num_target_tokens=num_target_tokens, batch_size=batch_size,
        )


# ---------------------------------------------------------------------------
# Dataset + collate (port of upstream CustomDataset + build_prompts_and_labels)
# ---------------------------------------------------------------------------


@dataclass
class RdoExample:
    harmful_prompt: str
    harmless_prompt: str
    ablation_prompt: str
    addition_prompt: str
    retain_prompt: str
    ablation_labels: torch.Tensor   # (seq-1,) with -100 for instruction tokens
    addition_labels: torch.Tensor   # (seq-1,) with -100 for instruction tokens


def _build_prompts_and_labels(
    tokenizer,
    harmful_chat_prompts: Sequence[str],
    harmless_chat_prompts: Sequence[str],
    ablation_targets: Sequence[str],
    addition_targets: Sequence[str],
    retain_targets: Sequence[str],
) -> list[RdoExample]:
    """Port of upstream `build_prompts_and_labels` (rdo.py:376).

    Masks the instruction tokens with -100 so CE only counts target tokens.
    """
    n = len(harmful_chat_prompts)
    if not (n == len(harmless_chat_prompts) == len(ablation_targets)
            == len(addition_targets) == len(retain_targets)):
        raise ValueError(
            "All input sequences must have the same length: "
            f"harmful={n}, harmless={len(harmless_chat_prompts)}, "
            f"abl={len(ablation_targets)}, add={len(addition_targets)}, "
            f"retain={len(retain_targets)}"
        )

    examples: list[RdoExample] = []
    for harmful, harmless, abl_t, add_t, ret_t in zip(
        harmful_chat_prompts, harmless_chat_prompts,
        ablation_targets, addition_targets, retain_targets,
    ):
        ablation_text = harmful + abl_t
        addition_text = harmless + add_t
        retain_text = harmless + ret_t

        ablation_tokens = tokenizer.encode(
            ablation_text, add_special_tokens=False, return_tensors="pt",
        )[0]
        addition_tokens = tokenizer.encode(
            addition_text, add_special_tokens=False, return_tensors="pt",
        )[0]

        # Shifted labels for next-token prediction.
        ablation_label = ablation_tokens[1:].clone()
        addition_label = addition_tokens[1:].clone()

        harmful_len = len(tokenizer.encode(harmful, add_special_tokens=False)) - 1
        harmless_len = len(tokenizer.encode(harmless, add_special_tokens=False)) - 1

        ablation_label[: max(harmful_len, 0)] = -100
        addition_label[: max(harmless_len, 0)] = -100

        examples.append(RdoExample(
            harmful_prompt=harmful,
            harmless_prompt=harmless,
            ablation_prompt=ablation_text,
            addition_prompt=addition_text,
            retain_prompt=retain_text,
            ablation_labels=ablation_label,
            addition_labels=addition_label,
        ))
    return examples


class RdoDataset(Dataset):
    def __init__(self, examples: list[RdoExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> RdoExample:
        return self.examples[idx]


def build_rdo_dataset(
    tokenizer,
    *,
    harmful_chat_prompts: Sequence[str],
    harmless_chat_prompts: Sequence[str],
    ablation_targets: Sequence[str],
    addition_targets: Sequence[str],
    retain_targets: Sequence[str],
) -> RdoDataset:
    examples = _build_prompts_and_labels(
        tokenizer,
        harmful_chat_prompts, harmless_chat_prompts,
        ablation_targets, addition_targets, retain_targets,
    )
    return RdoDataset(examples)


def rdo_collate(batch: list[RdoExample]) -> dict[str, Any]:
    """Collate function for batch_size=1 (upstream default). For larger
    batches we stack labels and let the trainer right-pad them; here we keep
    it simple."""
    return {
        "harmful_prompt": [ex.harmful_prompt for ex in batch],
        "harmless_prompt": [ex.harmless_prompt for ex in batch],
        "ablation_prompt": [ex.ablation_prompt for ex in batch],
        "addition_prompt": [ex.addition_prompt for ex in batch],
        "retain_prompt": [ex.retain_prompt for ex in batch],
        "ablation_labels": [ex.ablation_labels for ex in batch],
        "addition_labels": [ex.addition_labels for ex in batch],
    }


# ---------------------------------------------------------------------------
# CE / KL helpers
# ---------------------------------------------------------------------------


def _ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """CE loss with -100 ignore_index, flattened (B, T-1, V) -> ((B*T-1), V)."""
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1).to(flat_logits.device)
    return F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)


def _tokenize_for_forward(
    tokenizer,
    texts: Sequence[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Right-pad encoded texts for a forward pass (training, not generation)."""
    # Temporarily switch to right padding for forward; restore after.
    side_before = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        inputs = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
            add_special_tokens=False,
        )
    finally:
        tokenizer.padding_side = side_before
    return {k: v.to(device) for k, v in inputs.items()}


def _stack_labels(label_list: list[torch.Tensor], target_len: int, device: torch.device) -> torch.Tensor:
    """Right-pad labels to ``target_len`` with -100, then stack."""
    out = []
    for lab in label_list:
        if lab.shape[0] >= target_len:
            out.append(lab[:target_len])
        else:
            pad = torch.full((target_len - lab.shape[0],), -100, dtype=lab.dtype)
            out.append(torch.cat([lab, pad], dim=0))
    return torch.stack(out, dim=0).to(device)


# ---------------------------------------------------------------------------
# Training loop (port of refusal_cone_optimization, cone_dim=1, n_sample=0)
# ---------------------------------------------------------------------------


@dataclass
class RdoTrainingResult:
    best_v: torch.Tensor          # final (lowest-loss) direction, detached
    lowest_loss: float
    train_losses: list[float]     # per accumulation step
    step_logs: list[dict[str, float]]


def train_refusal_direction(
    model: Model,
    *,
    v_seed: torch.Tensor,
    alpha: float,
    best_layer: int,
    train_dataset: RdoDataset,
    cfg: dict[str, Any],
    on_step: Callable[[int, dict[str, float]], None] | None = None,
) -> RdoTrainingResult:
    device = model.model.device
    layers = list(model.get_layers())
    if not (0 <= best_layer < len(layers)):
        raise ValueError(f"best_layer={best_layer} out of range [0,{len(layers)})")
    add_layer = layers[best_layer]
    num_target_tokens = int(cfg["num_target_tokens"])
    tokenizer = model.tokenizer

    # Freeze base model — we only train ``v``.
    model.model.requires_grad_(False)
    model.model.eval()

    v = nn.Parameter(
        v_seed.detach().to(device=device, dtype=torch.float32).clone(),
        requires_grad=True,
    )
    with torch.no_grad():
        v.data.div_(v.data.norm())

    optimizer = torch.optim.AdamW(
        [v], lr=float(cfg["lr"]), betas=(0.9, 0.98),
        weight_decay=0.0, amsgrad=True,
    )

    batch_size = int(cfg["batch_size"])
    if batch_size != 1:
        # The retain-KL slice `logits[:, -num_target_tokens:, :]` assumes the
        # last positions are real tokens; with right-padding and batch_size > 1
        # some of those positions would be pad tokens. Upstream RDO uses
        # batch_size=1 exclusively. Enforce here.
        raise ValueError(
            f"RDO requires RDO_BATCH_SIZE=1; got {batch_size}. Use "
            "RDO_EFFECTIVE_BATCH_SIZE to control gradient accumulation."
        )
    effective_batch_size = int(cfg["effective_batch_size"])
    accumulation_steps = max(effective_batch_size // batch_size, 1)
    if len(train_dataset) < batch_size:
        raise ValueError(
            f"train_dataset has {len(train_dataset)} examples, less than "
            f"batch_size={batch_size}; cannot train."
        )
    loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, collate_fn=rdo_collate,
    )

    lowest_loss = math.inf
    best_v = v.detach().clone()
    train_losses: list[float] = []
    step_logs: list[dict[str, float]] = []
    patience_counter = lr_reduce_counter = step_counter = 0
    micro_counter = 0
    pending_total = 0.0

    abl_lambda = float(cfg["ablation_lambda"])
    add_lambda = float(cfg["addition_lambda"])
    ret_lambda = float(cfg["retain_lambda"])
    patience = int(cfg["patience"])
    n_lr_reduce = int(cfg["n_lr_reduce"])
    epochs = int(cfg["epochs"])

    logger.info(
        "RDO training: epochs=%d batch_size=%d eff_batch=%d acc_steps=%d "
        "alpha=%.4f best_layer=%d λ=(%.2f,%.2f,%.2f)",
        epochs, batch_size, effective_batch_size, accumulation_steps,
        alpha, best_layer, abl_lambda, add_lambda, ret_lambda,
    )

    stopped = False
    for epoch in range(epochs):
        if stopped:
            break
        for batch in loader:
            if stopped:
                break

            loss_parts: dict[str, float] = {
                "ablation": 0.0, "addition": 0.0, "retain": 0.0,
            }

            # --- ablation CE on harmful prompts ---
            if abl_lambda > 0:
                with torch.enable_grad():
                    with installed_ablation_hooks(layers, v):
                        inputs = _tokenize_for_forward(tokenizer, batch["ablation_prompt"], device)
                        target_len = inputs["input_ids"].shape[1] - 1
                        labels = _stack_labels(batch["ablation_labels"], target_len, device)
                        logits = model.model(**inputs).logits[:, :-1, :]
                        loss_abl = _ce_loss(logits, labels)
                    (abl_lambda * loss_abl).backward()
                loss_parts["ablation"] = float(loss_abl.detach().item())

            # --- addition CE on harmless prompts ---
            if add_lambda > 0:
                with torch.enable_grad():
                    with installed_addition_hook(add_layer, v, alpha):
                        inputs = _tokenize_for_forward(tokenizer, batch["addition_prompt"], device)
                        target_len = inputs["input_ids"].shape[1] - 1
                        labels = _stack_labels(batch["addition_labels"], target_len, device)
                        logits = model.model(**inputs).logits[:, :-1, :]
                        loss_add = _ce_loss(logits, labels)
                    (add_lambda * loss_add).backward()
                loss_parts["addition"] = float(loss_add.detach().item())

            # --- retain KL on harmless prompts ---
            if ret_lambda > 0:
                inputs = _tokenize_for_forward(tokenizer, batch["retain_prompt"], device)
                with torch.no_grad():
                    base_logits = model.model(**inputs).logits[:, -num_target_tokens:, :].detach()
                with torch.enable_grad():
                    with installed_ablation_hooks(layers, v):
                        new_logits = model.model(**inputs).logits[:, -num_target_tokens:, :]
                    # Match upstream RDO: KL(sample || baseline) — pulls the
                    # ablated distribution back to the base distribution.
                    # Upstream computes F.kl_div(log_softmax(base), softmax(sample));
                    # in the log-target API this is F.kl_div(log_softmax(base),
                    # log_softmax(sample), log_target=True), which propagates
                    # gradients through ``target`` (new_logp).
                    base_logp = F.log_softmax(base_logits.float(), dim=-1)
                    new_logp = F.log_softmax(new_logits.float(), dim=-1)
                    loss_ret = F.kl_div(
                        base_logp, new_logp,
                        reduction="batchmean", log_target=True,
                    )
                    (ret_lambda * loss_ret).backward()
                loss_parts["retain"] = float(loss_ret.detach().item())

            micro_total = sum(loss_parts.values())
            pending_total += micro_total
            micro_counter += 1

            if micro_counter % accumulation_steps != 0:
                continue

            # Accumulation window done — apply optimizer step.
            with torch.no_grad():
                if v.grad is not None:
                    # tangent-project grad onto the sphere of unit norm
                    v.grad.sub_(_projection(v.grad, v.data))
                    v.grad.div_(float(accumulation_steps))
                grad_norm = float(torch.nn.utils.clip_grad_norm_([v], 10.0).item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                v.data.div_(v.data.norm())

            step_counter += 1
            avg_total = pending_total / accumulation_steps
            train_losses.append(avg_total)
            step_record = {
                "step": float(step_counter),
                "total_loss": avg_total,
                "ablation_loss": loss_parts["ablation"],
                "addition_loss": loss_parts["addition"],
                "retain_loss": loss_parts["retain"],
                "grad_norm": grad_norm,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            step_logs.append(step_record)
            if on_step is not None:
                on_step(step_counter, step_record)
            pending_total = 0.0

            if avg_total < lowest_loss:
                lowest_loss = avg_total
                best_v = v.detach().clone()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if lr_reduce_counter >= n_lr_reduce:
                        logger.info(
                            "RDO: stopping at step %d (lr_reduce=%d, patience exhausted)",
                            step_counter, lr_reduce_counter,
                        )
                        stopped = True
                        break
                    lr_reduce_counter += 1
                    new_lr = optimizer.param_groups[0]["lr"] / 10.0
                    for pg in optimizer.param_groups:
                        pg["lr"] = new_lr
                    logger.info(
                        "RDO: step %d — reducing lr to %.3e (reduction %d/%d)",
                        step_counter, new_lr, lr_reduce_counter, n_lr_reduce,
                    )
                    patience_counter = 0

    return RdoTrainingResult(
        best_v=best_v.detach().cpu(),
        lowest_loss=float(lowest_loss if lowest_loss != math.inf else float("nan")),
        train_losses=train_losses,
        step_logs=step_logs,
    )


__all__ = [
    "RdoExample",
    "RdoDataset",
    "RdoTrainingResult",
    "build_rdo_dataset",
    "rdo_collate",
    "pick_seed_direction",
    "generate_intervention_targets",
    "train_refusal_direction",
    "installed_ablation_hooks",
    "installed_addition_hook",
    "orthogonalize_embedding",
]
