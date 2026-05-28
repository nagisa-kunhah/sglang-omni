# SPDX-License-Identifier: Apache-2.0

import pytest

from sglang_omni.utils.lru_cache import (
    LruCache,
    normalize_lru_limit,
)


def test_normalize_lru_limit() -> None:
    assert normalize_lru_limit(2) == 2
    assert normalize_lru_limit(0) == 0
    assert normalize_lru_limit(-1) == 0
    assert normalize_lru_limit(None) == 0


def test_lru_cache_eviction_uses_lru_order() -> None:
    cache: LruCache[str, int] = LruCache(max_size=2)

    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1

    cache.put("c", 3)

    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3


def test_lru_cache_copy_on_get_protects_cached_value() -> None:
    cache: LruCache[str, list[list[int]]] = LruCache(
        max_size=1,
        copy_on_get=lambda rows: [row.copy() for row in rows],
    )
    cache.put("ref", [[1, 2], [3, 4]])

    cached = cache.get("ref")
    assert cached == [[1, 2], [3, 4]]
    cached[0][0] = 99

    assert cache.get("ref") == [[1, 2], [3, 4]]


def test_lru_cache_max_size_zero_disables_cache() -> None:
    cache: LruCache[str, int] = LruCache(max_size=0)

    cache.put("a", 1)

    assert cache.get("a") is None


def test_lru_cache_max_size_none_disables_cache() -> None:
    cache: LruCache[str, int] = LruCache(max_size=None)

    cache.put("a", 1)

    assert cache.get("a") is None


def test_lru_cache_byte_budget_evicts_lru_entries() -> None:
    cache: LruCache[str, str] = LruCache(
        max_size=10,
        max_bytes=5,
        size_of=len,
    )

    cache.put("a", "aa")
    cache.put("b", "bb")
    assert cache.get("a") == "aa"

    cache.put("c", "cc")

    assert cache.total_bytes == 4
    assert cache.get("b") is None
    assert cache.get("a") == "aa"
    assert cache.get("c") == "cc"


def test_lru_cache_large_value_can_evict_multiple_entries() -> None:
    cache: LruCache[str, str] = LruCache(
        max_size=10,
        max_bytes=6,
        size_of=len,
    )

    cache.put("a", "aa")
    cache.put("b", "bb")
    cache.put("c", "cc")
    cache.put("d", "dddddd")

    assert cache.total_bytes == 6
    assert cache.get("a") is None
    assert cache.get("b") is None
    assert cache.get("c") is None
    assert cache.get("d") == "dddddd"


def test_lru_cache_skips_oversized_value_without_changing_existing_entries() -> None:
    cache: LruCache[str, str] = LruCache(
        max_size=10,
        max_bytes=3,
        size_of=len,
    )

    cache.put("a", "aa")
    cache.put("big", "xxxx")

    assert cache.total_bytes == 2
    assert cache.get("a") == "aa"
    assert cache.get("big") is None


def test_lru_cache_oversized_update_keeps_existing_entry() -> None:
    cache: LruCache[str, str] = LruCache(
        max_size=10,
        max_bytes=3,
        size_of=len,
    )

    cache.put("a", "aa")
    cache.put("a", "xxxx")

    assert cache.total_bytes == 2
    assert cache.get("a") == "aa"


def test_lru_cache_update_existing_key_adjusts_total_bytes() -> None:
    cache: LruCache[str, str] = LruCache(
        max_size=10,
        max_bytes=10,
        size_of=len,
    )

    cache.put("a", "aa")
    cache.put("a", "aaaa")

    assert cache.total_bytes == 4
    assert cache.get("a") == "aaaa"


def test_lru_cache_clear_resets_byte_accounting() -> None:
    cache: LruCache[str, str] = LruCache(
        max_size=10,
        max_bytes=10,
        size_of=len,
    )
    cache.put("a", "aa")

    cache.clear()

    assert cache.total_bytes == 0
    assert cache.get("a") is None


def test_lru_cache_requires_size_of_when_byte_budget_enabled() -> None:
    with pytest.raises(ValueError, match="size_of"):
        LruCache[str, str](max_size=10, max_bytes=10)


def test_lru_cache_rejects_negative_value_size() -> None:
    cache: LruCache[str, str] = LruCache(
        max_size=10,
        max_bytes=10,
        size_of=lambda _value: -1,
    )

    with pytest.raises(ValueError, match="non-negative"):
        cache.put("a", "aa")
