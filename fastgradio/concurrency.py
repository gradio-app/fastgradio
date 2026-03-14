from __future__ import annotations

import asyncio


class ConcurrencyLimiter:
    def __init__(self):
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    def get_semaphore(self, key: str, limit: int) -> asyncio.Semaphore:
        if key not in self._semaphores:
            self._semaphores[key] = asyncio.Semaphore(limit)
        return self._semaphores[key]
