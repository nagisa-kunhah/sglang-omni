# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

from benchmarks.eval import benchmark_higgs_audio_cache


def test_higgs_audio_cache_benchmark_reports_expected_counts() -> None:
    args = argparse.Namespace(
        iterations=5,
        warmup=1,
        waveform_samples=32,
        frames=4,
        num_codebooks=2,
        codec_delay_ms=0.0,
        cache_size=128,
        cache_max_bytes=256 * 1024 * 1024,
        json_output=None,
    )

    rows = benchmark_higgs_audio_cache.run_benchmark(args)
    by_name = {row["scenario"]: row for row in rows}

    assert by_name["disabled_cache_same_waveform"]["codec_calls"] == 5
    assert by_name["disabled_cache_same_waveform"]["hash_calls"] == 0

    assert by_name["cache_enabled_same_waveform_hit"]["codec_calls"] == 0
    assert by_name["cache_enabled_same_waveform_hit"]["hash_calls"] == 5

    assert by_name["cache_enabled_unique_waveform_miss"]["codec_calls"] == 5
    assert by_name["cache_enabled_unique_waveform_miss"]["hash_calls"] == 5

    for row in rows:
        assert row["mean_payload_build_ms"] >= 0
        assert row["mean_compute_ms"] >= 0
        assert "mean_hash_ms" in row


def test_higgs_audio_cache_benchmark_formats_markdown_table() -> None:
    row = {
        "scenario": "cache_enabled_same_waveform_hit",
        "iterations": 2,
        "codec_calls": 0,
        "hash_calls": 2,
        "prompt_build_calls": 2,
        "total_compute_ms": 1.0,
        "mean_compute_ms": 0.5,
        "total_payload_build_ms": 0.2,
        "mean_payload_build_ms": 0.1,
        "total_hash_ms": 0.02,
        "mean_hash_ms": 0.01,
        "total_fake_codec_ms": 0.0,
        "mean_fake_codec_ms": 0.0,
        "total_prompt_build_ms": 0.04,
        "mean_prompt_build_ms": 0.02,
    }

    table = benchmark_higgs_audio_cache._format_markdown_table([row])

    assert "| scenario |" in table
    assert "codec_calls" in table
    assert "hash_calls" in table
    assert "mean_payload_build_ms" in table
