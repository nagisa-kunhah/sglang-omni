# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import typer

from sglang_omni.cli.serve import apply_asr_encoder_torch_compile_cli_overrides
from sglang_omni.config import PipelineConfig, StageConfig, resolve_stage_factory_args
from sglang_omni.models.moss_transcribe_diarize.config import (
    MossTranscribeDiarizePipelineConfig,
)
from sglang_omni.models.qwen3_asr.config import Qwen3ASRPipelineConfig


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _install_fake_sglang_model_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    modules = {
        "sglang": _package("sglang"),
        "sglang.srt": _package("sglang.srt"),
        "sglang.srt.layers": _package("sglang.srt.layers"),
        "sglang.srt.layers.quantization": _package("sglang.srt.layers.quantization"),
        "sglang.srt.layers.quantization.base_config": ModuleType(
            "sglang.srt.layers.quantization.base_config"
        ),
        "sglang.srt.managers": _package("sglang.srt.managers"),
        "sglang.srt.managers.mm_utils": ModuleType("sglang.srt.managers.mm_utils"),
        "sglang.srt.managers.schedule_batch": ModuleType(
            "sglang.srt.managers.schedule_batch"
        ),
        "sglang.srt.model_executor": _package("sglang.srt.model_executor"),
        "sglang.srt.model_executor.forward_batch_info": ModuleType(
            "sglang.srt.model_executor.forward_batch_info"
        ),
        "sglang.srt.model_loader": _package("sglang.srt.model_loader"),
        "sglang.srt.model_loader.weight_utils": ModuleType(
            "sglang.srt.model_loader.weight_utils"
        ),
        "sglang.srt.models": _package("sglang.srt.models"),
        "sglang.srt.models.qwen3": ModuleType("sglang.srt.models.qwen3"),
        "sglang.srt.models.whisper": ModuleType("sglang.srt.models.whisper"),
        "sglang.srt.utils": ModuleType("sglang.srt.utils"),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    modules["sglang.srt.layers.quantization.base_config"].QuantizationConfig = object
    modules[
        "sglang.srt.managers.mm_utils"
    ].MultiModalityDataPaddingPatternMultimodalTokens = object
    modules["sglang.srt.managers.mm_utils"].general_mm_embed_routine = (
        lambda **kwargs: kwargs
    )
    modules["sglang.srt.managers.schedule_batch"].Modality = SimpleNamespace(
        AUDIO="audio"
    )
    modules["sglang.srt.managers.schedule_batch"].MultimodalDataItem = object
    modules["sglang.srt.managers.schedule_batch"].MultimodalInputs = object
    modules["sglang.srt.model_executor.forward_batch_info"].ForwardBatch = object
    modules["sglang.srt.model_loader.weight_utils"].default_weight_loader = (
        lambda param, loaded_weight: None
    )
    modules["sglang.srt.models.qwen3"].Qwen3ForCausalLM = object
    modules["sglang.srt.models.whisper"].WhisperEncoder = object
    modules["sglang.srt.utils"].add_prefix = lambda name, prefix: f"{prefix}.{name}"

    sys.modules.pop(
        "sglang_omni.models.moss_transcribe_diarize.sglang_model",
        None,
    )


def _install_fake_compile_config(monkeypatch: pytest.MonkeyPatch, calls: list[bool]):
    cuda_graph_runner = ModuleType("sglang.srt.model_executor.cuda_graph_runner")
    cuda_graph_runner.set_torch_compile_config = lambda: calls.append(True)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.cuda_graph_runner",
        cuda_graph_runner,
    )


def _import_sglang_model(monkeypatch: pytest.MonkeyPatch):
    _install_fake_sglang_model_deps(monkeypatch)
    return importlib.import_module(
        "sglang_omni.models.moss_transcribe_diarize.sglang_model"
    )


def test_compile_audio_encoder_uses_torch_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sglang_model = _import_sglang_model(monkeypatch)
    set_config_calls: list[bool] = []
    _install_fake_compile_config(monkeypatch, set_config_calls)
    compile_calls: list[tuple[object, str | None, bool | None]] = []

    def fake_compile(target, *, mode=None, dynamic=None):
        compile_calls.append((target, mode, dynamic))
        return f"compiled-{len(compile_calls)}"

    monkeypatch.setattr(torch, "compile", fake_compile)
    model = SimpleNamespace(whisper_encoder=object(), vq_adaptor=object())

    sglang_model.MossTranscribeDiarizeForConditionalGeneration.compile_audio_encoder(
        model,
        mode="reduce-overhead",
        dynamic=True,
        compile_adaptor=False,
    )

    assert set_config_calls == [True]
    assert compile_calls == [(model.whisper_encoder, "reduce-overhead", True)]
    assert model._compiled_whisper_encoder == "compiled-1"
    assert getattr(model, "_compiled_vq_adaptor", None) is None


def test_compile_audio_encoder_optionally_compiles_adaptor_and_encode_uses_compiled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sglang_model = _import_sglang_model(monkeypatch)
    set_config_calls: list[bool] = []
    _install_fake_compile_config(monkeypatch, set_config_calls)
    compile_calls: list[tuple[object, str | None, bool | None]] = []

    def fake_compile(target, *, mode=None, dynamic=None):
        compile_calls.append((target, mode, dynamic))
        return f"compiled-{len(compile_calls)}"

    monkeypatch.setattr(torch, "compile", fake_compile)

    class OriginalEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

        def forward(self, *args, **kwargs):
            raise AssertionError("eager encoder should not be called")

    class OriginalAdaptor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

        def forward(self, *args, **kwargs):
            raise AssertionError("eager adaptor should not be called")

    model = SimpleNamespace(
        whisper_encoder=OriginalEncoder(),
        vq_adaptor=OriginalAdaptor(),
    )
    sglang_model.MossTranscribeDiarizeForConditionalGeneration.compile_audio_encoder(
        model,
        mode="max-autotune-no-cudagraphs",
        dynamic=False,
        compile_adaptor=True,
    )
    assert compile_calls == [
        (model.whisper_encoder, "max-autotune-no-cudagraphs", False),
        (model.vq_adaptor, "max-autotune-no-cudagraphs", False),
    ]

    encoder_calls: list[torch.Tensor] = []
    adaptor_calls: list[torch.Tensor] = []

    def compiled_encoder(input_features, encoder_position_ids, forward_batch):
        del encoder_position_ids, forward_batch
        encoder_calls.append(input_features)
        return torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)

    def compiled_adaptor(features):
        adaptor_calls.append(features)
        return torch.ones((features.shape[0], 5), dtype=torch.float32)

    model._compiled_whisper_encoder = compiled_encoder
    model._compiled_vq_adaptor = compiled_adaptor
    model.config = SimpleNamespace(audio_merge_size=2)
    model_class = sglang_model.MossTranscribeDiarizeForConditionalGeneration
    model.time_merge = lambda features: model_class.time_merge(
        model,
        features,
    )
    model._audio_encoder_callable = lambda: model_class._audio_encoder_callable(model)
    model._vq_adaptor_callable = lambda: model_class._vq_adaptor_callable(model)
    item = SimpleNamespace(
        feature=torch.zeros((1, 80, 8), dtype=torch.float32),
        audio_feature_lengths=torch.tensor([2]),
        audio_chunk_mapping=torch.tensor([0]),
    )

    adapted = model_class.get_audio_feature(
        model,
        [item],
        forward_batch=object(),
    )

    assert len(encoder_calls) == 1
    assert len(adaptor_calls) == 1
    assert adaptor_calls[0].shape == (2, 6)
    assert adapted.shape == (2, 5)


def _install_fake_stage_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    modules = {
        "sglang": _package("sglang"),
        "sglang.srt": _package("sglang.srt"),
        "sglang.srt.managers": _package("sglang.srt.managers"),
        "sglang.srt.managers.mm_utils": ModuleType("sglang.srt.managers.mm_utils"),
        "sglang_omni.model_runner.base": ModuleType("sglang_omni.model_runner.base"),
        "sglang_omni.models.moss_transcribe_diarize.request_builders": ModuleType(
            "sglang_omni.models.moss_transcribe_diarize.request_builders"
        ),
        "sglang_omni.scheduling.bootstrap": ModuleType(
            "sglang_omni.scheduling.bootstrap"
        ),
        "sglang_omni.scheduling.omni_scheduler": ModuleType(
            "sglang_omni.scheduling.omni_scheduler"
        ),
        "sglang_omni.scheduling.sglang_backend": ModuleType(
            "sglang_omni.scheduling.sglang_backend"
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    modules["sglang.srt.managers.mm_utils"].init_mm_embedding_cache = lambda size: None
    modules["sglang_omni.model_runner.base"].ModelRunner = (
        lambda model_worker, output_proc: SimpleNamespace(
            model_worker=model_worker,
            output_proc=output_proc,
        )
    )
    modules[
        "sglang_omni.models.moss_transcribe_diarize.request_builders"
    ].make_moss_transcribe_diarize_scheduler_adapters = lambda **kwargs: (
        object(),
        object(),
    )
    modules[
        "sglang_omni.scheduling.bootstrap"
    ].create_sglang_infrastructure_defer_cuda_graph = lambda *args, **kwargs: None

    class FakeScheduler:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    modules["sglang_omni.scheduling.omni_scheduler"].OmniScheduler = FakeScheduler

    backend = modules["sglang_omni.scheduling.sglang_backend"]
    backend.SGLangOutputProcessor = lambda **kwargs: SimpleNamespace(**kwargs)

    def fake_build_sglang_server_args(model_path, context_length, **kwargs):
        return SimpleNamespace(
            model_path=model_path,
            context_length=context_length,
            **kwargs,
        )

    backend.build_sglang_server_args = fake_build_sglang_server_args

    sys.modules.pop("sglang_omni.models.moss_transcribe_diarize.stages", None)


def _import_stages(monkeypatch: pytest.MonkeyPatch):
    _install_fake_stage_deps(monkeypatch)
    return importlib.import_module("sglang_omni.models.moss_transcribe_diarize.stages")


def test_moss_td_stage_applies_encoder_compile_without_enabling_sglang_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages = _import_stages(monkeypatch)
    monkeypatch.setattr(
        stages.AutoProcessor,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: SimpleNamespace(tokenizer=object())),
    )
    monkeypatch.setattr(stages, "_default_max_new_tokens", lambda model_path: 8)
    monkeypatch.setattr(
        stages,
        "_default_context_length",
        lambda model_path, max_new_tokens: 128,
    )
    monkeypatch.setattr(
        stages,
        "_warmup_moss_td_encoder_compile",
        lambda *a, **k: None,
    )
    compile_calls: list[dict[str, object]] = []
    init_graph_calls: list[bool] = []
    infrastructure_enable_torch_compile: list[bool] = []

    class FakeModel:
        def compile_audio_encoder(self, **kwargs):
            compile_calls.append(kwargs)

    class FakeRunner:
        def __init__(self, server_args):
            self.server_args = server_args
            self.model = FakeModel()

        def init_device_graphs(self):
            init_graph_calls.append(True)

    class FakeWorker:
        def __init__(self, server_args):
            self.model_runner = FakeRunner(server_args)

    def fake_create_sglang_infrastructure_defer_cuda_graph(
        server_args,
        gpu_id,
        *,
        model_arch_override,
    ):
        del gpu_id, model_arch_override
        infrastructure_enable_torch_compile.append(server_args.enable_torch_compile)
        return True, (
            FakeWorker(server_args),
            object(),
            object(),
            object(),
            object(),
            object(),
            SimpleNamespace(),
        )

    monkeypatch.setattr(
        stages,
        "create_sglang_infrastructure_defer_cuda_graph",
        fake_create_sglang_infrastructure_defer_cuda_graph,
    )

    scheduler = stages.create_sglang_moss_transcribe_diarize_executor(
        "model",
        device="cuda:0",
        encoder_torch_compile=True,
        encoder_torch_compile_mode="reduce-overhead",
        encoder_torch_compile_dynamic=False,
        encoder_torch_compile_warmup=False,
        encoder_torch_compile_adaptor=True,
    )

    assert infrastructure_enable_torch_compile == [False]
    assert compile_calls == [
        {
            "mode": "reduce-overhead",
            "dynamic": False,
            "compile_adaptor": True,
        }
    ]
    assert init_graph_calls == [True]
    assert scheduler.server_args.enable_torch_compile is False


def test_moss_td_stage_skips_encoder_compile_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages = _import_stages(monkeypatch)
    monkeypatch.setattr(
        stages.AutoProcessor,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: SimpleNamespace(tokenizer=object())),
    )
    monkeypatch.setattr(stages, "_default_max_new_tokens", lambda model_path: 8)
    monkeypatch.setattr(
        stages,
        "_default_context_length",
        lambda model_path, max_new_tokens: 128,
    )
    compile_calls: list[dict[str, object]] = []

    class FakeModel:
        def compile_audio_encoder(self, **kwargs):
            compile_calls.append(kwargs)

    class FakeRunner:
        def __init__(self, server_args):
            self.model = FakeModel()

        def init_device_graphs(self):
            raise AssertionError("CUDA graphs should be disabled in this fake")

    def fake_create_sglang_infrastructure_defer_cuda_graph(
        server_args,
        gpu_id,
        *,
        model_arch_override,
    ):
        del gpu_id, model_arch_override
        return False, (
            SimpleNamespace(model_runner=FakeRunner(server_args)),
            object(),
            object(),
            object(),
            object(),
            object(),
            SimpleNamespace(),
        )

    monkeypatch.setattr(
        stages,
        "create_sglang_infrastructure_defer_cuda_graph",
        fake_create_sglang_infrastructure_defer_cuda_graph,
    )

    stages.create_sglang_moss_transcribe_diarize_executor("model")

    assert compile_calls == []


def test_asr_encoder_torch_compile_cli_targets_moss_td_asr_stage() -> None:
    config = MossTranscribeDiarizePipelineConfig(model_path="model")

    apply_asr_encoder_torch_compile_cli_overrides(
        config,
        asr_encoder_torch_compile="on",
        asr_encoder_torch_compile_mode="reduce-overhead",
    )

    asr_stage = next(stage for stage in config.stages if stage.name == "asr")
    args = resolve_stage_factory_args(asr_stage, config)
    assert args["encoder_torch_compile"] is True
    assert args["encoder_torch_compile_mode"] == "reduce-overhead"


def test_asr_encoder_torch_compile_rejects_unsupported_factory() -> None:
    config = Qwen3ASRPipelineConfig(model_path="model")

    with pytest.raises(
        typer.BadParameter,
        match="MOSS-Transcribe-Diarize ASR",
    ):
        apply_asr_encoder_torch_compile_cli_overrides(
            config,
            asr_encoder_torch_compile="on",
            asr_encoder_torch_compile_mode=None,
        )


def test_asr_encoder_torch_compile_default_is_noop() -> None:
    config = PipelineConfig(
        model_path="model",
        stages=[
            StageConfig(
                name="stage",
                process="stage",
                factory="tests.unit_test.fixtures.pipeline_fakes.dummy_factory",
                terminal=True,
            )
        ],
    )
    before = config.model_dump()

    apply_asr_encoder_torch_compile_cli_overrides(
        config,
        asr_encoder_torch_compile="default",
        asr_encoder_torch_compile_mode=None,
    )

    assert config.model_dump() == before
