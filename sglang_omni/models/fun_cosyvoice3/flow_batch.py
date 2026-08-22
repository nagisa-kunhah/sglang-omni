# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
# Copyright (c) 2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
# Adapted from FunAudioLLM/CosyVoice commit
# 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc:
# cosyvoice/flow/flow.py
# cosyvoice/flow/flow_matching.py
"""True-batch adapter for the pinned CosyVoice3 PyTorch Flow decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

__all__ = [
    "FlowBatchInput",
    "infer_flow_batch",
]


@dataclass(frozen=True)
class FlowBatchInput:
    """One unpadded request for CosyVoice3 Flow inference."""

    token: torch.Tensor
    prompt_token: torch.Tensor
    prompt_feat: torch.Tensor
    embedding: torch.Tensor


@dataclass(frozen=True)
class _PackedFlowBatch:
    token: torch.Tensor
    token_mask: torch.Tensor
    combined_token_lengths: tuple[int, ...]
    prompt_token_lengths: tuple[int, ...]
    target_token_lengths: tuple[int, ...]
    prompt_mel_lengths: tuple[int, ...]
    total_mel_lengths: tuple[int, ...]
    combined_token_lengths_tensor: torch.Tensor
    total_mel_lengths_tensor: torch.Tensor
    embedding: torch.Tensor
    prompt_feat: tuple[torch.Tensor, ...]


def _flow_device_and_dtype(flow: Any) -> tuple[torch.device, torch.dtype]:
    try:
        parameter = next(flow.parameters())
    except (AttributeError, StopIteration) as exc:
        raise ValueError("Flow must expose at least one parameter") from exc
    return parameter.device, parameter.dtype


def _validate_flow_input(flow: Any, item: FlowBatchInput, index: int) -> None:
    if item.token.ndim != 2 or item.token.shape[0] != 1:
        raise ValueError(f"input {index} token must have shape [1, target_tokens]")
    if item.token.shape[1] <= 0:
        raise ValueError(f"input {index} target token sequence must not be empty")
    if item.prompt_token.ndim != 2 or item.prompt_token.shape[0] != 1:
        raise ValueError(
            f"input {index} prompt_token must have shape [1, prompt_tokens]"
        )
    if item.prompt_feat.ndim != 3 or item.prompt_feat.shape[0] != 1:
        raise ValueError(
            f"input {index} prompt_feat must have shape [1, prompt_frames, channels]"
        )
    if item.prompt_feat.shape[2] != flow.output_size:
        raise ValueError(
            f"input {index} prompt feature width must equal Flow output_size "
            f"({item.prompt_feat.shape[2]} != {flow.output_size})"
        )
    expected_prompt_frames = item.prompt_token.shape[1] * flow.token_mel_ratio
    if item.prompt_feat.shape[1] != expected_prompt_frames:
        raise ValueError(
            f"input {index} prompt feature length must equal prompt token length "
            f"times token_mel_ratio ({item.prompt_feat.shape[1]} != "
            f"{expected_prompt_frames})"
        )
    if item.embedding.ndim != 2 or item.embedding.shape[0] != 1:
        raise ValueError(f"input {index} embedding must have shape [1, speaker_dim]")

    expected_embedding_size = getattr(flow.spk_embed_affine_layer, "in_features", None)
    if (
        expected_embedding_size is not None
        and item.embedding.shape[1] != expected_embedding_size
    ):
        raise ValueError(
            f"input {index} embedding width must be {expected_embedding_size}, "
            f"got {item.embedding.shape[1]}"
        )


def _pack_flow_inputs(
    flow: Any,
    inputs: Sequence[FlowBatchInput],
) -> _PackedFlowBatch:
    """Pack prompt and target tokens contiguously for every request row."""

    if not inputs:
        raise ValueError("Flow batch must contain at least one input")
    for index, item in enumerate(inputs):
        _validate_flow_input(flow, item, index)

    device, dtype = _flow_device_and_dtype(flow)
    prompt_token_lengths = tuple(int(item.prompt_token.shape[1]) for item in inputs)
    target_token_lengths = tuple(int(item.token.shape[1]) for item in inputs)
    combined_token_lengths = tuple(
        prompt_tokens + target_tokens
        for prompt_tokens, target_tokens in zip(
            prompt_token_lengths, target_token_lengths, strict=True
        )
    )
    prompt_mel_lengths = tuple(int(item.prompt_feat.shape[1]) for item in inputs)
    total_mel_lengths = tuple(
        token_length * flow.token_mel_ratio for token_length in combined_token_lengths
    )
    combined_token_lengths_tensor = torch.tensor(
        combined_token_lengths,
        dtype=torch.int64,
        device=device,
    )
    total_mel_lengths_tensor = torch.tensor(
        total_mel_lengths,
        dtype=torch.int64,
        device=device,
    )

    max_tokens = max(combined_token_lengths)
    token = torch.zeros(len(inputs), max_tokens, dtype=torch.int32, device=device)
    for index, item in enumerate(inputs):
        prompt_tokens = prompt_token_lengths[index]
        target_tokens = target_token_lengths[index]
        token[index, :prompt_tokens] = item.prompt_token[0].to(
            device=device, dtype=torch.int32
        )
        token[index, prompt_tokens : prompt_tokens + target_tokens] = item.token[0].to(
            device=device, dtype=torch.int32
        )

    token_mask = (
        torch.arange(max_tokens, device=device).unsqueeze(0)
        < combined_token_lengths_tensor.unsqueeze(1)
    ).unsqueeze(-1)
    embedding = torch.cat(
        [item.embedding.to(device=device, dtype=dtype) for item in inputs], dim=0
    )
    prompt_feat = tuple(
        item.prompt_feat.to(device=device, dtype=dtype) for item in inputs
    )
    return _PackedFlowBatch(
        token=token,
        token_mask=token_mask,
        combined_token_lengths=combined_token_lengths,
        prompt_token_lengths=prompt_token_lengths,
        target_token_lengths=target_token_lengths,
        prompt_mel_lengths=prompt_mel_lengths,
        total_mel_lengths=total_mel_lengths,
        combined_token_lengths_tensor=combined_token_lengths_tensor,
        total_mel_lengths_tensor=total_mel_lengths_tensor,
        embedding=embedding,
        prompt_feat=prompt_feat,
    )


def _solve_euler_batch(
    decoder: Any,
    x: torch.Tensor,
    t_span: torch.Tensor,
    mu: torch.Tensor,
    mask: torch.Tensor,
    spks: torch.Tensor,
    cond: torch.Tensor,
) -> torch.Tensor:
    """Run upstream Euler/CFG semantics with conditional/unconditional ``2B``."""

    batch_size, channels, frames = x.shape
    speaker_dim = spks.shape[1]
    estimator_dtype = spks.dtype
    x_in = torch.zeros(
        2 * batch_size,
        channels,
        frames,
        device=x.device,
        dtype=estimator_dtype,
    )
    mask_in = torch.zeros(
        2 * batch_size, 1, frames, device=x.device, dtype=estimator_dtype
    )
    mu_in = torch.zeros_like(x_in)
    t_in = torch.zeros(2 * batch_size, device=x.device, dtype=estimator_dtype)
    spks_in = torch.zeros(
        2 * batch_size,
        speaker_dim,
        device=x.device,
        dtype=estimator_dtype,
    )
    cond_in = torch.zeros_like(x_in)

    t = t_span[0]
    dt = t_span[1] - t_span[0]
    for step in range(1, len(t_span)):
        x_in[:batch_size] = x
        x_in[batch_size:] = x
        mask_in[:batch_size] = mask
        mask_in[batch_size:] = mask
        mu_in[:batch_size] = mu
        t_in[:] = t
        spks_in[:batch_size] = spks
        cond_in[:batch_size] = cond

        dphi_dt = decoder.forward_estimator(
            x_in,
            mask_in,
            mu_in,
            t_in,
            spks_in,
            cond_in,
            streaming=False,
        )
        conditional = dphi_dt[:batch_size]
        unconditional = dphi_dt[batch_size:]
        guided = (
            1.0 + decoder.inference_cfg_rate
        ) * conditional - decoder.inference_cfg_rate * unconditional
        x = x + dt * guided
        t = t + dt
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t

    return x.float()


def _causal_cfm_forward_batch(
    decoder: Any,
    *,
    mu: torch.Tensor,
    mask: torch.Tensor,
    spks: torch.Tensor,
    cond: torch.Tensor,
    n_timesteps: int = 10,
) -> torch.Tensor:
    """Run the pinned causal CFM with deterministic per-request noise prefixes."""

    batch_size, _, max_mel = mu.shape
    available_frames = decoder.rand_noise.shape[2]
    if max_mel > available_frames:
        raise ValueError(
            f"decoder.rand_noise supports {available_frames} frames, "
            f"but batch requires {max_mel}"
        )
    base_noise = decoder.rand_noise[:, :, :max_mel]
    z = base_noise.to(device=mu.device, dtype=mu.dtype)
    z = z.expand(batch_size, -1, -1).clone()

    t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
    if decoder.t_scheduler == "cosine":
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    return _solve_euler_batch(decoder, z, t_span, mu, mask, spks, cond)


@torch.inference_mode()
def infer_flow_batch(
    flow: Any,
    inputs: Sequence[FlowBatchInput],
) -> list[torch.Tensor]:
    """Infer variable-length CosyVoice3 mel outputs in one true Flow batch."""

    packed = _pack_flow_inputs(flow, inputs)
    embedding = F.normalize(packed.embedding, dim=1)
    embedding = flow.spk_embed_affine_layer(embedding)

    token_embedding = flow.input_embedding(torch.clamp(packed.token, min=0))
    token_embedding = token_embedding * packed.token_mask.to(token_embedding.dtype)
    h = flow.pre_lookahead_layer(token_embedding)
    h = h.repeat_interleave(flow.token_mel_ratio, dim=1)
    mu = h.transpose(1, 2).contiguous()

    batch_size, channels, max_mel = mu.shape
    if channels != flow.output_size:
        raise ValueError(
            "Flow pre-lookahead output width does not match output_size "
            f"({channels} != {flow.output_size})"
        )
    mel_valid = torch.arange(max_mel, device=mu.device).unsqueeze(
        0
    ) < packed.total_mel_lengths_tensor.unsqueeze(1)
    mask = mel_valid.unsqueeze(1).to(mu.dtype)
    cond = torch.zeros(
        batch_size,
        flow.output_size,
        max_mel,
        device=mu.device,
        dtype=mu.dtype,
    )
    for index, prompt_feat in enumerate(packed.prompt_feat):
        prompt_frames = packed.prompt_mel_lengths[index]
        cond[index, :, :prompt_frames] = prompt_feat[0].transpose(0, 1)

    feat = _causal_cfm_forward_batch(
        flow.decoder,
        mu=mu,
        mask=mask,
        spks=embedding,
        cond=cond,
        n_timesteps=10,
    )
    outputs: list[torch.Tensor] = []
    for index in range(batch_size):
        start = packed.prompt_mel_lengths[index]
        end = packed.total_mel_lengths[index]
        mel = feat[index : index + 1, :, start:end]
        expected_frames = packed.target_token_lengths[index] * flow.token_mel_ratio
        expected_shape = (1, flow.output_size, expected_frames)
        if mel.shape != expected_shape:
            raise RuntimeError(
                f"Flow batch output {index} has shape {tuple(mel.shape)}, "
                f"expected {expected_shape}"
            )
        outputs.append(mel)
    return outputs
