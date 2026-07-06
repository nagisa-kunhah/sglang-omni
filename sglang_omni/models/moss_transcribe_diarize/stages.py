# SPDX-License-Identifier: Apache-2.0
"""Stage factory for SGLang-backed MOSS-Transcribe-Diarize inference."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import torch
from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from transformers import AutoConfig, AutoProcessor, GenerationConfig

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.moss_transcribe_diarize import (  # noqa: F401
    hf_config as _hf_config,
)
from sglang_omni.models.moss_transcribe_diarize.request_builders import (
    make_moss_transcribe_diarize_scheduler_adapters,
)
from sglang_omni.scheduling.bootstrap import (
    create_sglang_infrastructure_defer_cuda_graph,
)
from sglang_omni.scheduling.generation_batch_policy import (
    build_generation_batch_overrides,
    validate_generation_batch_policy,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend import (
    SGLangOutputProcessor,
    build_sglang_server_args,
)

logger = logging.getLogger(__name__)

# Note (yichi): Budget for long-form input and let the checkpoint window cap it.
_LONG_FORM_PROMPT_TOKENS = 72000


@contextmanager
def _missing_additional_chat_templates_compat() -> Iterator[None]:
    """Treat a missing optional chat-template directory as no extra templates."""
    import transformers.processing_utils as processing_utils
    import transformers.utils.hub as hub_utils
    from huggingface_hub.errors import RepositoryNotFoundError

    patched: list[tuple[Any, Any]] = []

    def patch_list_repo_templates(module: Any) -> None:
        original = getattr(module, "list_repo_templates", None)
        if original is None:
            return

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            try:
                return original(*args, **kwargs)
            except RepositoryNotFoundError as exc:
                if "additional_chat_templates" in str(exc):
                    return []
                raise

        setattr(module, "list_repo_templates", wrapped)
        patched.append((module, original))

    try:
        patch_list_repo_templates(processing_utils)
        patch_list_repo_templates(hub_utils)
        yield
    finally:
        for module, original in reversed(patched):
            setattr(module, "list_repo_templates", original)


def _default_context_length(model_path: str, max_new_tokens: int) -> int:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_config = getattr(config, "text_config", None)
    max_positions = int(getattr(text_config, "max_position_embeddings", 40960))
    return min(max_positions, _LONG_FORM_PROMPT_TOKENS + int(max_new_tokens))


def _default_max_new_tokens(model_path: str) -> int:
    try:
        generation_config = GenerationConfig.from_pretrained(model_path)
    except Exception:
        return 5120
    return int(getattr(generation_config, "max_new_tokens", None) or 5120)


def _resolve_encoder_torch_compile_mode(mode: str | None) -> str:
    return mode or os.environ.get(
        "SGLANG_TORCH_COMPILE_MODE",
        "max-autotune-no-cudagraphs",
    )


def _torch_dtype_from_name(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    normalized = str(dtype).replace("torch.", "").lower()
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "half", "fp16"}:
        return torch.float16
    if normalized in {"float32", "float", "fp32"}:
        return torch.float32
    return torch.bfloat16


def _warmup_moss_td_encoder_compile(
    model: Any,
    *,
    device: str,
    dtype: str | torch.dtype,
) -> None:
    compiled_encoder = getattr(model, "_compiled_whisper_encoder", None)
    if compiled_encoder is None:
        return

    try:
        first_param = next(model.whisper_encoder.parameters())
        warmup_device = first_param.device
        warmup_dtype = first_param.dtype
    except (AttributeError, StopIteration):
        warmup_device = torch.device(device)
        warmup_dtype = _torch_dtype_from_name(dtype)

    audio_config = model.config.audio_config
    num_mel_bins = int(getattr(audio_config, "num_mel_bins", 80))
    max_source_positions = int(getattr(audio_config, "max_source_positions", 1500))
    num_frames = max(2, max_source_positions * 2)
    input_features = torch.zeros(
        (1, num_mel_bins, num_frames),
        device=warmup_device,
        dtype=warmup_dtype,
    )
    encoder_len = (input_features.shape[-1] - 1) // 2 + 1
    encoder_position_ids = torch.arange(
        encoder_len,
        device=input_features.device,
        dtype=torch.long,
    )
    fake_forward_batch = SimpleNamespace()

    with torch.inference_mode():
        compiled_encoder(input_features, encoder_position_ids, fake_forward_batch)
    logger.info(
        "Warmed up MOSS-TD encoder compile with input_features shape=%s",
        tuple(input_features.shape),
    )


def create_sglang_moss_transcribe_diarize_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    max_running_requests: int = 16,
    max_new_tokens: int | None = None,
    context_length: int | None = None,
    mem_fraction_static: float | None = 0.80,
    mm_embedding_cache_size_bytes: int = 0,
    enable_torch_compile: bool = False,
    encoder_torch_compile: bool = False,
    encoder_torch_compile_mode: str | None = None,
    encoder_torch_compile_dynamic: bool = False,
    encoder_torch_compile_warmup: bool = True,
    encoder_torch_compile_adaptor: bool = False,
    request_build_max_workers: int = 2,
    request_build_max_pending: int | None = 16,
    server_args_overrides: dict[str, Any] | None = None,
):
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    with _missing_additional_chat_templates_compat():
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer

    resolved_max_new_tokens = (
        int(max_new_tokens)
        if max_new_tokens is not None
        else _default_max_new_tokens(model_path)
    )
    resolved_context_length = (
        int(context_length)
        if context_length is not None
        else _default_context_length(model_path, resolved_max_new_tokens)
    )

    overrides = build_generation_batch_overrides(
        max_running_requests=max_running_requests,
        server_args_overrides=server_args_overrides,
        disable_cuda_graph=False,
        disable_overlap_schedule=True,
        enable_torch_compile=enable_torch_compile,
        mem_fraction_static=mem_fraction_static,
        max_prefill_tokens=4096,
        chunked_prefill_size=4096,
        sampling_backend="pytorch",
        dtype=dtype,
    )

    server_args = build_sglang_server_args(
        model_path,
        context_length=resolved_context_length,
        **overrides,
    )
    validate_generation_batch_policy(
        model_name="MOSS-Transcribe-Diarize",
        server_args=server_args,
    )

    want_cuda_graph, (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure_defer_cuda_graph(
        server_args,
        gpu_id,
        model_arch_override="MossTranscribeDiarizeForConditionalGeneration",
    )

    if encoder_torch_compile:
        model = model_worker.model_runner.model
        compile_mode = _resolve_encoder_torch_compile_mode(encoder_torch_compile_mode)
        model.compile_audio_encoder(
            mode=compile_mode,
            dynamic=encoder_torch_compile_dynamic,
            compile_adaptor=encoder_torch_compile_adaptor,
        )
        if encoder_torch_compile_warmup:
            _warmup_moss_td_encoder_compile(model, device=device, dtype=dtype)

    if want_cuda_graph:
        model_worker.model_runner.init_device_graphs()

    init_mm_embedding_cache(mm_embedding_cache_size_bytes)

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model_worker.model_runner.model,
    )
    request_builder, result_adapter = make_moss_transcribe_diarize_scheduler_adapters(
        processor=processor,
        tokenizer=tokenizer,
        max_new_tokens=resolved_max_new_tokens,
    )

    return OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=ModelRunner(model_worker, output_proc),
        request_builder=request_builder,
        result_adapter=result_adapter,
        request_build_max_workers=request_build_max_workers,
        request_build_max_pending=request_build_max_pending,
    )


def create_moss_transcribe_diarize_executor(*args, **kwargs):
    return create_sglang_moss_transcribe_diarize_executor(*args, **kwargs)


__all__ = [
    "create_sglang_moss_transcribe_diarize_executor",
    "create_moss_transcribe_diarize_executor",
]
