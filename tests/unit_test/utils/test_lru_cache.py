# SPDX-License-Identifier: Apache-2.0

from sglang_omni.utils.lru_cache import LruCache


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
