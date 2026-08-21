# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint invariance checks for Fun-CosyVoice3 Flow batching."""

from __future__ import annotations

import importlib
import os
import subprocess
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download

from sglang_omni.models.fun_cosyvoice3.flow_batch import (
    FlowBatchInput,
    infer_flow_batch,
)

MODEL_PATH = os.environ.get(
    "FUN_COSYVOICE3_TEST_MODEL",
    "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
)
COSYVOICE_COMMIT = "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"


def _assert_pinned_cosyvoice_checkout() -> None:
    flow_module = importlib.import_module("cosyvoice.flow.flow")
    module_path = Path(flow_module.__file__).resolve()
    repository = next(
        (parent for parent in module_path.parents if (parent / ".git").exists()),
        None,
    )
    if repository is None:
        raise AssertionError(
            "CosyVoice must be imported from the documented git checkout so its "
            "pinned commit can be verified"
        )
    actual_commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert actual_commit == COSYVOICE_COMMIT


def _input(
    *,
    target_tokens: int,
    prompt_tokens: int,
    seed: int,
) -> FlowBatchInput:
    generator = torch.Generator().manual_seed(seed)
    return FlowBatchInput(
        token=torch.randint(
            0,
            6561,
            (1, target_tokens),
            dtype=torch.int32,
            generator=generator,
        ),
        prompt_token=torch.randint(
            0,
            6561,
            (1, prompt_tokens),
            dtype=torch.int32,
            generator=generator,
        ),
        prompt_feat=torch.randn(1, prompt_tokens * 2, 80, generator=generator),
        embedding=torch.randn(1, 192, generator=generator),
    )


def _serial_flow(flow, item: FlowBatchInput, *, fp16: bool) -> torch.Tensor:
    device = next(flow.parameters()).device
    with torch.autocast(device_type="cuda", enabled=fp16):
        mel, _ = flow.inference(
            token=item.token.to(device),
            token_len=torch.tensor(
                [item.token.shape[1]], dtype=torch.int32, device=device
            ),
            prompt_token=item.prompt_token.to(device),
            prompt_token_len=torch.tensor(
                [item.prompt_token.shape[1]], dtype=torch.int32, device=device
            ),
            prompt_feat=item.prompt_feat.to(device),
            prompt_feat_len=torch.tensor(
                [item.prompt_feat.shape[1]], dtype=torch.int32, device=device
            ),
            embedding=item.embedding.to(device),
            streaming=False,
            finalize=True,
        )
    return mel


@pytest.mark.benchmark
@pytest.mark.gpu
@pytest.mark.parametrize("fp16", [False, True], ids=["float32", "float16"])
def test_fun_cosyvoice3_flow_batch_matches_upstream_and_batch_neighbors(
    fp16: bool,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Fun-CosyVoice3 Flow batch invariance requires CUDA")

    _assert_pinned_cosyvoice_checkout()
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice3
    except ImportError as exc:
        pytest.fail(f"CosyVoice is not importable: {exc}")

    checkpoint_dir = (
        MODEL_PATH if Path(MODEL_PATH).is_dir() else snapshot_download(MODEL_PATH)
    )
    cosyvoice = CosyVoice3(checkpoint_dir, fp16=fp16)
    flow = cosyvoice.model.flow.to("cuda").eval()
    del cosyvoice.model.llm
    del cosyvoice.model.hift

    a = _input(target_tokens=10, prompt_tokens=4, seed=11)
    same_total_length = _input(target_tokens=12, prompt_tokens=2, seed=22)
    longer_neighbor = _input(target_tokens=15, prompt_tokens=6, seed=33)
    serial = {
        id(item): _serial_flow(flow, item, fp16=fp16)
        for item in (a, same_total_length, longer_neighbor)
    }

    with torch.autocast(device_type="cuda", enabled=fp16):
        singleton_a = infer_flow_batch(flow, [a])[0]
        same_length_batch = infer_flow_batch(flow, [a, same_total_length])
        padded_batch = infer_flow_batch(flow, [a, longer_neighbor])

    rtol, atol = (2e-2, 2e-2) if fp16 else (1e-4, 1e-4)
    torch.testing.assert_close(singleton_a, serial[id(a)], rtol=rtol, atol=atol)
    torch.testing.assert_close(
        same_length_batch[0], serial[id(a)], rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        same_length_batch[1],
        serial[id(same_total_length)],
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(padded_batch[0], serial[id(a)], rtol=rtol, atol=atol)
    torch.testing.assert_close(
        padded_batch[1], serial[id(longer_neighbor)], rtol=rtol, atol=atol
    )
