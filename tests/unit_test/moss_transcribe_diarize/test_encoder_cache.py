# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.machinery
import inspect
import sys
import types
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn


def _install_sglang_unit_stubs() -> None:
    sentencepiece_stub = types.ModuleType("sentencepiece")
    sentencepiece_stub.__spec__ = importlib.machinery.ModuleSpec("sentencepiece", None)
    sys.modules.setdefault("sentencepiece", sentencepiece_stub)

    @dataclass
    class FakeSGLangARRequestData:
        input_ids: Any = None
        output_ids: list[int] | None = None
        finish_reason: str | None = None
        weight_version: str | None = None
        max_new_tokens: int | None = None
        temperature: float = 0.0
        req: Any = None
        stage_payload: Any = None

    class FakeModality(Enum):
        AUDIO = "audio"

    class FakeMultimodalDataItem:
        def __init__(
            self,
            *,
            modality,
            hash,
            feature=None,
            model_specific_data=None,
            **kwargs,
        ) -> None:
            self.modality = modality
            self.hash = hash
            self.feature = feature
            self.model_specific_data = model_specific_data or {}
            self.__dict__.update(self.model_specific_data)
            self.__dict__.update(kwargs)
            self.offsets = []
            self.pad_value = None

        def set_pad_value(self) -> None:
            self.pad_value = -1

    class FakeMultimodalInputs:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    class FakeReq:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    class FakeSamplingParams:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

        def normalize(self, tokenizer=None) -> None:
            del tokenizer

    class FakePattern:
        def pad_input_tokens(self, input_ids, mm_inputs):
            del mm_inputs
            return input_ids

    class FakeQwen3ForCausalLM(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

        def get_input_embeddings(self):
            return None

    class FakeWhisperEncoder(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            self.dummy = nn.Parameter(torch.empty(()))

    class FakeServerArgs:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    def default_weight_loader(*args, **kwargs) -> None:
        del args, kwargs

    def add_prefix(name: str, prefix: str = "") -> str:
        return f"{prefix}.{name}" if prefix else name

    modules = {
        "sglang": types.ModuleType("sglang"),
        "sglang.srt": types.ModuleType("sglang.srt"),
        "sglang.srt.layers": types.ModuleType("sglang.srt.layers"),
        "sglang.srt.layers.quantization": types.ModuleType(
            "sglang.srt.layers.quantization"
        ),
        "sglang.srt.layers.quantization.base_config": types.ModuleType(
            "sglang.srt.layers.quantization.base_config"
        ),
        "sglang.srt.managers": types.ModuleType("sglang.srt.managers"),
        "sglang.srt.managers.mm_utils": types.ModuleType(
            "sglang.srt.managers.mm_utils"
        ),
        "sglang.srt.managers.schedule_batch": types.ModuleType(
            "sglang.srt.managers.schedule_batch"
        ),
        "sglang.srt.model_executor": types.ModuleType("sglang.srt.model_executor"),
        "sglang.srt.model_executor.forward_batch_info": types.ModuleType(
            "sglang.srt.model_executor.forward_batch_info"
        ),
        "sglang.srt.model_loader": types.ModuleType("sglang.srt.model_loader"),
        "sglang.srt.model_loader.weight_utils": types.ModuleType(
            "sglang.srt.model_loader.weight_utils"
        ),
        "sglang.srt.models": types.ModuleType("sglang.srt.models"),
        "sglang.srt.models.qwen3": types.ModuleType("sglang.srt.models.qwen3"),
        "sglang.srt.models.whisper": types.ModuleType("sglang.srt.models.whisper"),
        "sglang.srt.sampling": types.ModuleType("sglang.srt.sampling"),
        "sglang.srt.sampling.sampling_params": types.ModuleType(
            "sglang.srt.sampling.sampling_params"
        ),
        "sglang.srt.server_args": types.ModuleType("sglang.srt.server_args"),
        "sglang.srt.utils": types.ModuleType("sglang.srt.utils"),
        "sglang_omni.scheduling.sglang_backend": types.ModuleType(
            "sglang_omni.scheduling.sglang_backend"
        ),
        "sglang_omni.scheduling.omni_scheduler": types.ModuleType(
            "sglang_omni.scheduling.omni_scheduler"
        ),
    }
    for package_name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.managers",
        "sglang.srt.model_executor",
        "sglang.srt.model_loader",
        "sglang.srt.models",
        "sglang.srt.sampling",
    ):
        modules[package_name].__path__ = []

    modules["sglang"].srt = modules["sglang.srt"]
    modules["sglang.srt"].layers = modules["sglang.srt.layers"]
    modules["sglang.srt"].managers = modules["sglang.srt.managers"]
    modules["sglang.srt"].model_executor = modules["sglang.srt.model_executor"]
    modules["sglang.srt"].model_loader = modules["sglang.srt.model_loader"]
    modules["sglang.srt"].models = modules["sglang.srt.models"]
    modules["sglang.srt"].sampling = modules["sglang.srt.sampling"]
    modules["sglang.srt"].utils = modules["sglang.srt.utils"]
    modules["sglang.srt.layers"].quantization = modules[
        "sglang.srt.layers.quantization"
    ]
    modules["sglang.srt.layers.quantization"].base_config = modules[
        "sglang.srt.layers.quantization.base_config"
    ]
    modules["sglang.srt.managers"].mm_utils = modules["sglang.srt.managers.mm_utils"]
    modules["sglang.srt.managers"].schedule_batch = modules[
        "sglang.srt.managers.schedule_batch"
    ]
    modules["sglang.srt.model_executor"].forward_batch_info = modules[
        "sglang.srt.model_executor.forward_batch_info"
    ]
    modules["sglang.srt.model_loader"].weight_utils = modules[
        "sglang.srt.model_loader.weight_utils"
    ]
    modules["sglang.srt.models"].qwen3 = modules["sglang.srt.models.qwen3"]
    modules["sglang.srt.models"].whisper = modules["sglang.srt.models.whisper"]
    modules["sglang.srt.sampling"].sampling_params = modules[
        "sglang.srt.sampling.sampling_params"
    ]

    modules["sglang.srt.layers.quantization.base_config"].QuantizationConfig = object
    modules[
        "sglang.srt.managers.mm_utils"
    ].MultiModalityDataPaddingPatternMultimodalTokens = FakePattern
    modules["sglang.srt.managers.mm_utils"].general_mm_embed_routine = (
        lambda **kwargs: None
    )
    modules["sglang.srt.managers.mm_utils"].init_mm_embedding_cache = lambda size: None
    modules["sglang.srt.managers.schedule_batch"].Modality = FakeModality
    modules["sglang.srt.managers.schedule_batch"].MultimodalDataItem = (
        FakeMultimodalDataItem
    )
    modules["sglang.srt.managers.schedule_batch"].MultimodalInputs = (
        FakeMultimodalInputs
    )
    modules["sglang.srt.managers.schedule_batch"].Req = FakeReq
    modules["sglang.srt.model_executor.forward_batch_info"].ForwardBatch = object
    modules["sglang.srt.model_loader.weight_utils"].default_weight_loader = (
        default_weight_loader
    )
    modules["sglang.srt.models.qwen3"].Qwen3ForCausalLM = FakeQwen3ForCausalLM
    modules["sglang.srt.models.whisper"].WhisperEncoder = FakeWhisperEncoder
    modules["sglang.srt.sampling.sampling_params"].SamplingParams = FakeSamplingParams
    modules["sglang.srt.server_args"].ServerArgs = FakeServerArgs
    modules["sglang.srt.utils"].add_prefix = add_prefix
    modules["sglang_omni.scheduling.sglang_backend"].SGLangARRequestData = (
        FakeSGLangARRequestData
    )
    modules["sglang_omni.scheduling.sglang_backend"].SGLangOutputProcessor = (
        lambda **kwargs: object()
    )
    modules["sglang_omni.scheduling.sglang_backend"].build_sglang_server_args = (
        lambda model_path, context_length, **kwargs: FakeServerArgs(
            model_path=model_path,
            context_length=context_length,
            **kwargs,
        )
    )
    modules["sglang_omni.scheduling.omni_scheduler"].OmniScheduler = (
        lambda **kwargs: SimpleNamespace(**kwargs)
    )

    for name, module in modules.items():
        sys.modules.setdefault(name, module)


_install_sglang_unit_stubs()

import sglang_omni.models.moss_transcribe_diarize.stages as moss_td_stages
from sglang_omni.models.moss_transcribe_diarize.sglang_model import (
    MossTranscribeDiarizeForConditionalGeneration,
)
from sglang_omni.scheduling.stage_cache import StageOutputCache


class CountingWhisperEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.empty((), dtype=torch.float32))
        self.calls = 0

    def forward(
        self,
        input_features: torch.Tensor,
        encoder_position_ids: torch.Tensor,
        forward_batch,
    ) -> torch.Tensor:
        del encoder_position_ids, forward_batch
        self.calls += 1
        chunks = input_features.shape[0]
        values = torch.arange(
            chunks * 4 * 2,
            device=input_features.device,
            dtype=input_features.dtype,
        )
        return values.reshape(chunks, 4, 2)


class IdentityAdaptor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.empty((), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _fake_model() -> MossTranscribeDiarizeForConditionalGeneration:
    model = object.__new__(MossTranscribeDiarizeForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        audio_merge_size=2,
        text_config=SimpleNamespace(hidden_size=4),
    )
    model.whisper_encoder = CountingWhisperEncoder()
    model.vq_adaptor = IdentityAdaptor()
    model.encoder_output_cache = None
    return model


def _fake_audio_item(
    *,
    audio_hash: int = 123,
    audio_feature_lengths: list[int] | None = None,
    audio_chunk_mapping: list[int] | None = None,
):
    item = SimpleNamespace()
    item.hash = audio_hash
    item.feature = torch.zeros((1, 80, 8), dtype=torch.float32)
    item.audio_feature_lengths = torch.tensor(
        audio_feature_lengths or [2],
        dtype=torch.long,
    )
    item.audio_chunk_mapping = torch.tensor(
        audio_chunk_mapping or [0],
        dtype=torch.long,
    )
    return item


def test_encoder_cache_hit_skips_whisper_encoder() -> None:
    model = _fake_model()
    model.set_encoder_output_cache(
        StageOutputCache(max_size=8, max_bytes=1024**2, cache_device="cpu")
    )
    item = _fake_audio_item()

    first = model._encode_one_audio_item(item, forward_batch=None)
    second = model._encode_one_audio_item(item, forward_batch=None)

    assert model.whisper_encoder.calls == 1
    assert len(first) == 1
    assert len(second) == 1
    assert torch.equal(first[0], second[0])
    adaptor_param = next(model.vq_adaptor.parameters())
    assert second[0].device == adaptor_param.device
    assert second[0].dtype == adaptor_param.dtype
    assert model.encoder_output_cache.current_bytes > 0


def test_encoder_cache_metadata_change_causes_miss() -> None:
    model = _fake_model()
    model.set_encoder_output_cache(
        StageOutputCache(max_size=8, max_bytes=1024**2, cache_device="cpu")
    )

    model._encode_one_audio_item(_fake_audio_item(), forward_batch=None)
    model._encode_one_audio_item(
        _fake_audio_item(audio_feature_lengths=[1]),
        forward_batch=None,
    )

    assert model.whisper_encoder.calls == 2


def test_disabled_encoder_cache_preserves_existing_behavior() -> None:
    model = _fake_model()
    model.set_encoder_output_cache(None)
    item = _fake_audio_item()

    model._encode_one_audio_item(item, forward_batch=None)
    model._encode_one_audio_item(item, forward_batch=None)

    assert model.whisper_encoder.calls == 2


def test_moss_td_stage_default_cache_signature() -> None:
    signature = inspect.signature(
        moss_td_stages.create_sglang_moss_transcribe_diarize_executor
    )

    assert signature.parameters["encoder_cache_max_entries"].default == 64
    assert signature.parameters["encoder_cache_max_bytes"].default == 4 * 1024**3
    assert signature.parameters["mm_embedding_cache_size_bytes"].default == 0


def test_moss_td_stage_wires_encoder_cache_by_default(monkeypatch) -> None:
    fake_model = _patch_stage_factory_dependencies(monkeypatch)

    moss_td_stages.create_sglang_moss_transcribe_diarize_executor("dummy")

    assert isinstance(fake_model.encoder_output_cache, StageOutputCache)
    assert fake_model.encoder_output_cache.max_size == 64
    assert fake_model.encoder_output_cache.max_bytes == 4 * 1024**3
    assert fake_model.encoder_output_cache.cache_device == torch.device("cpu")


def test_moss_td_stage_can_disable_encoder_cache(monkeypatch) -> None:
    fake_model = _patch_stage_factory_dependencies(monkeypatch)

    moss_td_stages.create_sglang_moss_transcribe_diarize_executor(
        "dummy",
        encoder_cache_max_entries=0,
    )

    assert fake_model.encoder_output_cache is None


def test_moss_td_stage_can_disable_encoder_cache_by_bytes(monkeypatch) -> None:
    fake_model = _patch_stage_factory_dependencies(monkeypatch)

    moss_td_stages.create_sglang_moss_transcribe_diarize_executor(
        "dummy",
        encoder_cache_max_bytes=0,
    )

    assert fake_model.encoder_output_cache is None


def _patch_stage_factory_dependencies(monkeypatch):
    class FakeModel:
        def __init__(self) -> None:
            self.encoder_output_cache = "unset"

        def set_encoder_output_cache(self, cache):
            self.encoder_output_cache = cache

    fake_model = FakeModel()

    monkeypatch.setattr(
        moss_td_stages.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(tokenizer=object()),
    )
    monkeypatch.setattr(
        moss_td_stages.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            text_config=SimpleNamespace(max_position_embeddings=40960)
        ),
    )
    monkeypatch.setattr(
        moss_td_stages.GenerationConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(max_new_tokens=128),
    )
    monkeypatch.setattr(
        moss_td_stages,
        "init_mm_embedding_cache",
        lambda size: None,
    )
    monkeypatch.setattr(
        moss_td_stages,
        "make_moss_transcribe_diarize_scheduler_adapters",
        lambda **kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        moss_td_stages,
        "ModelRunner",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        moss_td_stages,
        "SGLangOutputProcessor",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        moss_td_stages,
        "OmniScheduler",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        moss_td_stages,
        "build_sglang_server_args",
        lambda model_path, context_length, **overrides: SimpleNamespace(
            context_length=context_length,
            **overrides,
        ),
    )

    def _fake_create_infrastructure(server_args, gpu_id, **kwargs):
        del server_args, gpu_id, kwargs
        model_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                model=fake_model,
                init_device_graphs=lambda: None,
            )
        )
        return (
            model_worker,
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
        )

    monkeypatch.setattr(
        moss_td_stages,
        "create_sglang_infrastructure",
        _fake_create_infrastructure,
    )
    return fake_model
