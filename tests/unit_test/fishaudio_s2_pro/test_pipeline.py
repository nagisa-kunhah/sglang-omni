# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.fishaudio_s2_pro.config import S2ProPipelineConfig
from sglang_omni.models.fishaudio_s2_pro.payload_types import S2ProState
from sglang_omni.models.fishaudio_s2_pro.request_builders import (
    S2ProSGLangRequestData,
    apply_tts_result,
    build_sglang_tts_request,
    make_tts_scheduler_adapters,
)
from sglang_omni.models.fishaudio_s2_pro.tokenizer import (
    Reference,
    S2ProTokenizerAdapter,
)
from tests.unit_test.fixtures.fish_fakes import (
    FakeFishTokenizer,
    make_s2pro_payload,
    make_s2pro_state,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def fast_sampling_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.normalize",
        lambda self, tokenizer: None,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        lambda self, vocab_size: None,
    )


def _install_s2pro_tts_factory_fakes(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    from sglang_omni.engines.omni import compile as compile_mod
    from sglang_omni.models.fishaudio_s2_pro import bootstrap, model_runner, stages
    from sglang_omni.scheduling import bootstrap as scheduling_bootstrap
    from sglang_omni.scheduling import sglang_backend

    calls = SimpleNamespace(
        build_kwargs={},
        infrastructure_disable_cuda_graph=None,
        init_device_graphs=[],
        compile_count=0,
        events=[],
    )

    class FakeSGLangRunner:
        def __init__(self, server_args) -> None:
            self.server_args = server_args
            self.model = object()

        def init_device_graphs(self) -> None:
            calls.events.append("init_device_graphs")
            calls.init_device_graphs.append(
                {
                    "disable_cuda_graph": self.server_args.disable_cuda_graph,
                    "enable_torch_compile": self.server_args.enable_torch_compile,
                    "torch_compile_max_bs": self.server_args.torch_compile_max_bs,
                }
            )

    monkeypatch.setattr(stages, "_resolve_checkpoint", lambda model_path: model_path)
    monkeypatch.setattr(bootstrap, "patch_fish_config_for_sglang", lambda: None)
    monkeypatch.setattr(bootstrap, "truncate_rope_to_bf16", lambda model: None)
    monkeypatch.setattr(
        bootstrap, "bootstrap_text_model_for_decode", lambda **kwargs: None
    )
    monkeypatch.setattr(
        bootstrap,
        "load_audio_decoder",
        lambda checkpoint_dir, device: (
            SimpleNamespace(),
            2,
            4096,
            FakeFishTokenizer(),
        ),
    )
    monkeypatch.setattr(
        stages,
        "make_tts_scheduler_adapters",
        lambda **kwargs: (lambda payload: payload, lambda data: data),
    )
    monkeypatch.setattr(
        sglang_backend,
        "SGLangOutputProcessor",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        model_runner,
        "FishS2ProModelRunner",
        lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs),
    )

    def fake_build_sglang_server_args(model_path, context_length, **kwargs):
        del model_path, context_length
        calls.build_kwargs = kwargs
        return SimpleNamespace(
            disable_cuda_graph=kwargs["disable_cuda_graph"],
            enable_torch_compile=kwargs.get("enable_torch_compile", False),
            torch_compile_max_bs=kwargs.get("torch_compile_max_bs", 16),
            max_running_requests=kwargs["max_running_requests"],
            chunked_prefill_size=kwargs["chunked_prefill_size"],
            max_prefill_tokens=kwargs.get("max_prefill_tokens", 8192),
            page_size=1,
            attention_backend=kwargs.get("attention_backend"),
        )

    def fake_create_sglang_infrastructure(server_args, gpu_id):
        del gpu_id
        calls.infrastructure_disable_cuda_graph = server_args.disable_cuda_graph
        return (
            SimpleNamespace(model_runner=FakeSGLangRunner(server_args)),
            object(),
            object(),
            object(),
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
        )

    def fake_apply_compile_targets(model):
        del model
        calls.events.append("apply_compile_targets")
        calls.compile_count += 1
        return ["fake-target"]

    monkeypatch.setattr(
        sglang_backend, "build_sglang_server_args", fake_build_sglang_server_args
    )
    monkeypatch.setattr(
        scheduling_bootstrap,
        "create_sglang_infrastructure",
        fake_create_sglang_infrastructure,
    )
    monkeypatch.setattr(compile_mod, "apply_compile_targets", fake_apply_compile_targets)

    return calls


def test_fish_config_state_and_tokenizer_prompt_contracts() -> None:
    """Preserves S2-Pro topology, state tensor round-trip, and prompt VQ layout."""
    config = S2ProPipelineConfig(model_path="model")
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "tts_engine",
        "vocoder",
    ]
    assert config.terminal_stages == ["vocoder"]
    assert config.gpu_placement == {"tts_engine": 0, "vocoder": 0}

    state = S2ProState(
        input_ids=torch.tensor([1, 2, 3]),
        vq_mask_tokens=torch.tensor([False, True, False]),
        vq_parts=[torch.tensor([[10, 11], [20, 21]])],
        output_codes=torch.tensor([[100, 101], [1, 2], [3, 4]]),
    )
    restored = S2ProState.from_dict(state.to_dict())
    assert restored.input_ids == [1, 2, 3]
    assert torch.equal(restored.vq_parts[0], torch.tensor([[10, 11], [20, 21]]))
    assert torch.equal(
        restored.output_codes, torch.tensor([[100, 101], [1, 2], [3, 4]])
    )

    tokenizer = FakeFishTokenizer()
    adapter = S2ProTokenizerAdapter(tokenizer)
    prompt = adapter.build_prompt(
        "target",
        references=[
            Reference(
                audio_bytes=b"",
                text="ref",
                vq_codes=torch.tensor([[0, 1], [10, 11]], dtype=torch.long),
            )
        ],
        num_codebooks=2,
        speaker="alice",
    )
    assert adapter.eos_token_ids == [99]
    assert prompt["vq_mask_tokens"].dtype == torch.bool
    assert prompt["vq_mask_tokens"].sum().item() == 2
    assert torch.equal(prompt["vq_parts"][0], torch.tensor([[0, 1], [10, 11]]))
    assert any("<|speaker:alice|>target" in text for text in tokenizer.encoded_texts)


def test_fish_compile_example_config_targets_current_stage_factory() -> None:
    manager = ConfigManager.from_file(
        str(_REPO_ROOT / "examples" / "configs" / "s2pro_tts_compile.yaml")
    )
    config = manager.config
    tts_stage = next(stage for stage in config.stages if stage.name == "tts_engine")

    assert config.runtime_overrides["tts_engine"]["compile_level"] == "partial"
    assert tts_stage.factory == (
        "sglang_omni.models.fishaudio_s2_pro.stages."
        "create_sglang_tts_engine_executor"
    )


def test_fish_full_compile_example_config_uses_server_args_overrides() -> None:
    manager = ConfigManager.from_file(
        str(_REPO_ROOT / "examples" / "configs" / "s2pro_tts_full_compile.yaml")
    )
    config = manager.config

    overrides = config.runtime_overrides["tts_engine"]["server_args_overrides"]
    assert overrides == {
        "enable_torch_compile": True,
        "torch_compile_max_bs": 32,
    }


def test_fish_tts_full_compile_survives_deferred_cuda_graph_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_s2pro_tts_factory_fakes(monkeypatch)
    from sglang_omni.models.fishaudio_s2_pro import stages

    scheduler = stages.create_sglang_tts_engine_executor(
        "model",
        device="cuda:0",
        server_args_overrides={
            "enable_torch_compile": True,
            "torch_compile_max_bs": 32,
        },
    )

    assert calls.build_kwargs["enable_torch_compile"] is True
    assert calls.build_kwargs["torch_compile_max_bs"] == 32
    assert calls.infrastructure_disable_cuda_graph is True
    assert calls.init_device_graphs == [
        {
            "disable_cuda_graph": False,
            "enable_torch_compile": True,
            "torch_compile_max_bs": 32,
        }
    ]
    assert calls.compile_count == 0
    assert scheduler.batch_planner.server_args.enable_torch_compile is True


def test_fish_tts_partial_compile_runs_before_cuda_graph_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_s2pro_tts_factory_fakes(monkeypatch)
    from sglang_omni.models.fishaudio_s2_pro import stages

    stages.create_sglang_tts_engine_executor(
        "model",
        device="cuda:0",
        compile_level="partial",
    )

    assert calls.compile_count == 1
    assert calls.events == ["apply_compile_targets", "init_device_graphs"]
    assert calls.init_device_graphs == [
        {
            "disable_cuda_graph": False,
            "enable_torch_compile": False,
            "torch_compile_max_bs": 16,
        }
    ]


def test_fish_tts_rejects_partial_and_sglang_full_compile_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_s2pro_tts_factory_fakes(monkeypatch)
    from sglang_omni.models.fishaudio_s2_pro import stages

    with pytest.raises(
        ValueError,
        match=(
            "compile_level='partial' cannot be combined with "
            "server_args_overrides.enable_torch_compile=True"
        ),
    ):
        stages.create_sglang_tts_engine_executor(
            "model",
            compile_level="partial",
            server_args_overrides={"enable_torch_compile": True},
        )

    assert calls.build_kwargs == {}
    assert calls.infrastructure_disable_cuda_graph is None
    assert calls.compile_count == 0
    assert calls.init_device_graphs == []


def test_fish_tts_request_and_result_adapters_preserve_tensor_contracts() -> None:
    """Preserves TTS request tensor fields and result adapter output-code shape."""
    tokenizer = FakeFishTokenizer()
    state = make_s2pro_state(
        input_ids=[10, 11, 12],
        vq_mask_tokens=[False, True, True],
        vq_parts=[[[1, 2], [3, 4]]],
        max_new_tokens=6,
        temperature=0.6,
    )

    req_data = build_sglang_tts_request(state, tokenizer, request_id="req-1")
    assert torch.equal(req_data.input_ids, torch.tensor([10, 11, 12]))
    assert req_data.vq_mask_tokens.dtype == torch.bool
    assert torch.equal(req_data.vq_parts[0], torch.tensor([[1, 2], [3, 4]]))
    assert req_data.req.eos_token_ids == {99}

    req_data.output_codes = [
        torch.tensor([[100], [1], [2]], dtype=torch.long),
        torch.tensor([[101], [3], [4]], dtype=torch.long),
    ]
    apply_tts_result(state, req_data)
    assert torch.equal(
        state.output_codes,
        torch.tensor([[100, 101], [1, 3], [2, 4]], dtype=torch.long),
    )
    assert state.prompt_tokens == 3
    assert state.completion_tokens == 2

    payload = make_s2pro_payload(request_id="req-2")
    request_builder, result_adapter = make_tts_scheduler_adapters(tokenizer=tokenizer)
    adapted = request_builder(payload)
    adapted.output_codes = [torch.tensor([[100], [1], [2]], dtype=torch.long)]
    result_payload = result_adapter(adapted)
    assert adapted.stage_payload is payload
    assert result_payload.request is payload.request
    assert result_payload.data["output_codes"] == [[100], [1], [2]]


@pytest.mark.parametrize("top_k", [0, 31])
def test_fish_tts_rejects_top_k_outside_graph_width(top_k: int) -> None:
    tokenizer = FakeFishTokenizer()
    state = make_s2pro_state(top_k=top_k)

    with pytest.raises(ValueError, match="S2-Pro top_k must be -1 or between 1 and 30"):
        build_sglang_tts_request(state, tokenizer, request_id="bad-top-k")

    with pytest.raises(ValueError, match="S2-Pro top_k must be -1 or between 1 and 30"):
        S2ProSGLangRequestData(
            input_ids=torch.tensor([], dtype=torch.long),
            req=object(),
            top_k=top_k,
        )


def test_fish_tts_accepts_graph_top_k_width() -> None:
    tokenizer = FakeFishTokenizer()
    state = make_s2pro_state(top_k=30)

    req_data = build_sglang_tts_request(state, tokenizer, request_id="top-k-30")

    assert req_data.top_k == 30


def test_fish_tts_accepts_default_top_k_sentinel() -> None:
    tokenizer = FakeFishTokenizer()
    state = make_s2pro_state(top_k=-1)

    req_data = build_sglang_tts_request(state, tokenizer, request_id="top-k-default")

    assert req_data.top_k == -1
