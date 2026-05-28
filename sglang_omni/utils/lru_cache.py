# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Generic, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


def normalize_lru_max_size(max_size: int | None) -> int:
    return max(int(max_size), 0) if max_size is not None else 0


class LruCache(Generic[K, V]):
    """Small in-memory LRU cache for process-local reusable values."""

    def __init__(
        self,
        max_size: int | None,
        *,
        copy_on_get: Callable[[V], V] | None = None,
    ) -> None:
        self.max_size = normalize_lru_max_size(max_size)
        self.copy_on_get = copy_on_get
        self._entries: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K | None) -> V | None:
        if key is None:
            return None
        try:
            value = self._entries[key]
        except KeyError:
            return None
        self._entries.move_to_end(key)
        if self.copy_on_get is not None:
            return self.copy_on_get(value)
        return value

    def put(self, key: K | None, value: V) -> None:
        if key is None or self.max_size == 0:
            return
        self._entries[key] = value
        self._entries.move_to_end(key)
        self._evict_over_budget()

    def clear(self) -> None:
        self._entries.clear()

    def _evict_over_budget(self) -> None:
        while len(self._entries) > self.max_size:
            self._entries.popitem(last=False)


__all__ = ["LruCache", "normalize_lru_max_size"]
