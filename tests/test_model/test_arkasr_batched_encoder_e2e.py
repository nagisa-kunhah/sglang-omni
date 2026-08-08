# SPDX-License-Identifier: Apache-2.0
"""Opt-in E2E coverage for ARK-ASR variable-length encoder batching.

Run with a local ARK-ASR checkpoint and CUDA GPU::

    RUN_ARKASR_BATCHED_ENCODER_E2E=1 \
    ARKASR_MODEL_PATH=/path/to/ARK-ASR-3B \
    pytest -q tests/test_model/test_arkasr_batched_encoder_e2e.py -s

The default test suite skips this file: it starts a real server and requires
the multi-gigabyte checkpoint.  The request payloads are distinct valid WAV
files with varying trailing silence, preventing audio-embedding cache hits and
forcing the real variable-length encoder path.
"""

from __future__ import annotations

import concurrent.futures
import io
import os
import shlex
import sys
import time
import wave
from pathlib import Path

import pytest
import requests

from sglang_omni.utils import find_available_port
from tests.utils import disable_proxy, start_server_from_cmd, stop_server


RUN_ENV = "RUN_ARKASR_BATCHED_ENCODER_E2E"
MODEL_PATH_ENV = "ARKASR_MODEL_PATH"
MODEL_PATH = os.environ.get(MODEL_PATH_ENV, "")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIO_FIXTURES = (
    (
        PROJECT_ROOT / "tests/data/query_to_cars.wav",
        "how many cars are there in the picture.",
    ),
    (
        PROJECT_ROOT / "tests/data/query_to_draw.wav",
        "what is happening in this video answer in ten words or fewer.",
    ),
)
STARTUP_TIMEOUT_S = 600
REQUEST_TIMEOUT_S = 180
BURST_SIZE = 8

pytestmark = pytest.mark.skipif(
    os.environ.get(RUN_ENV) != "1",
    reason=(
        f"set {RUN_ENV}=1 and {MODEL_PATH_ENV}=/path/to/ARK-ASR-3B "
        "to run this real-model GPU E2E test"
    ),
)


def _wav_with_trailing_silence(path: Path, silence_index: int) -> bytes:
    """Return a valid WAV whose waveform and duration differ per request."""
    with wave.open(str(path), "rb") as source:
        params = source.getparams()
        frames = source.readframes(source.getnframes())
    silence_frames = params.framerate * (silence_index + 1) // 20
    silence = b"\0" * silence_frames * params.nchannels * params.sampwidth
    output = io.BytesIO()
    with wave.open(output, "wb") as destination:
        destination.setparams(params)
        destination.writeframes(frames + silence)
    return output.getvalue()


@pytest.fixture(scope="module")
def arkasr_server(tmp_path_factory: pytest.TempPathFactory):
    model_dir = Path(MODEL_PATH)
    if not model_dir.is_dir():
        pytest.skip(f"{MODEL_PATH_ENV} must name a local checkpoint directory")

    port = find_available_port()
    log_file = tmp_path_factory.mktemp("arkasr_batched_encoder") / "server.log"
    serve_cmd = [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-path",
        str(model_dir),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "info",
    ]
    # ARK-ASR's model repository has custom-code metadata.  Piping one "yes"
    # makes this test non-interactive without changing the serving API.
    cmd = ["bash", "-lc", f"yes y | {shlex.join(serve_cmd)}"]
    process = start_server_from_cmd(cmd, log_file, port, timeout=STARTUP_TIMEOUT_S)
    try:
        yield f"http://127.0.0.1:{port}", log_file
    finally:
        stop_server(process)


def _transcribe(base_url: str, payload: bytes, filename: str) -> str:
    with disable_proxy():
        response = requests.post(
            f"{base_url}/v1/audio/transcriptions",
            data={"model": MODEL_PATH},
            files={"file": (filename, payload, "audio/wav")},
            timeout=REQUEST_TIMEOUT_S,
        )
    response.raise_for_status()
    text = response.json().get("text")
    assert isinstance(text, str) and text
    return text


@pytest.mark.gpu
def test_arkasr_variable_length_audio_burst_batches_and_transcribes(arkasr_server):
    """A burst must batch real variable-length audio and preserve transcripts."""
    base_url, log_file = arkasr_server
    requests_to_send = []
    for index in range(BURST_SIZE):
        audio_path, expected_text = AUDIO_FIXTURES[index % len(AUDIO_FIXTURES)]
        assert audio_path.is_file(), f"audio fixture not found: {audio_path}"
        requests_to_send.append(
            (
                _wav_with_trailing_silence(audio_path, index),
                f"batch-{index}-{audio_path.name}",
                expected_text,
            )
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=BURST_SIZE) as pool:
        futures = [
            pool.submit(_transcribe, base_url, payload, filename)
            for payload, filename, _ in requests_to_send
        ]
        texts = [future.result() for future in futures]

    assert texts == [expected_text for _, _, expected_text in requests_to_send]

    # The server can partition one client burst into multiple scheduler
    # micro-batches, so do not require all eight requests in a single batch.
    # At least one multi-sequence prefill proves this reached the real serving
    # batching path rather than a sequence of independent HTTP requests.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        log_text = log_file.read_text(errors="replace")
        if "Prefill batch, #new-seq: 2" in log_text or any(
            f"Prefill batch, #new-seq: {count}" in log_text
            for count in range(3, BURST_SIZE + 1)
        ):
            break
        time.sleep(0.05)
    else:
        raise AssertionError(
            "expected at least one multi-sequence prefill batch; server log:\n"
            f"{log_file.read_text(errors='replace')}"
        )
