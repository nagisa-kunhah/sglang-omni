# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Generic, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


def normalize_lru_limit(limit: int | None) -> int:
    return max(int(limit), 0) if limit is not None else 0


@dataclass(frozen=True)
class _LruEntry(Generic[V]):
    value: V
    bytes: int


class LruCache(Generic[K, V]):
    """Small in-memory LRU cache for process-local reusable values."""

    def __init__(
        self,
        max_size: int | None,
        *,
        copy_on_get: Callable[[V], V] | None = None,
        max_bytes: int | None = None,
        size_of: Callable[[V], int] | None = None,
    ) -> None:
        self.max_size = normalize_lru_limit(max_size)
        self.max_bytes = normalize_lru_limit(max_bytes)
        if self.max_bytes > 0 and size_of is None:
            raise ValueError("size_of must be provided when max_bytes is positive")
        self.copy_on_get = copy_on_get
        self.size_of = size_of
        self.total_bytes = 0
        self._entries: OrderedDict[K, _LruEntry[V]] = OrderedDict()

    def get(self, key: K | None) -> V | None:
        if key is None:
            return None
        try:
            entry = self._entries[key]
        except KeyError:
            return None
        self._entries.move_to_end(key)
        if self.copy_on_get is not None:
            return self.copy_on_get(entry.value)
        return entry.value

    def put(self, key: K | None, value: V) -> None:
        if key is None or self.max_size == 0:
            return
        value_bytes = self._value_bytes(value) if self.max_bytes > 0 else 0
        if self.max_bytes > 0 and value_bytes > self.max_bytes:
            return

        if key in self._entries:
            old_entry = self._entries.pop(key)
            self.total_bytes -= old_entry.bytes

        self._evict_for_new_entry(value_bytes)

        self._entries[key] = _LruEntry(value=value, bytes=value_bytes)
        self.total_bytes += value_bytes
        self._entries.move_to_end(key)

    def clear(self) -> None:
        self._entries.clear()
        self.total_bytes = 0

    def _value_bytes(self, value: V) -> int:
        assert self.size_of is not None
        value_bytes = int(self.size_of(value))
        if value_bytes < 0:
            raise ValueError("size_of must return a non-negative integer")
        return value_bytes

    def _would_exceed_budget(self, value_bytes: int) -> bool:
        return len(self._entries) + 1 > self.max_size or (
            self.max_bytes > 0 and self.total_bytes + value_bytes > self.max_bytes
        )

    def _evict_for_new_entry(self, value_bytes: int) -> None:
        while self._would_exceed_budget(value_bytes):
            _, entry = self._entries.popitem(last=False)
            self.total_bytes -= entry.bytes


__all__ = [
    "LruCache",
    "normalize_lru_limit",
]
