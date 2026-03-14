from __future__ import annotations

import asyncio
from typing import Any, Callable

from starlette.concurrency import run_in_threadpool


class BatchProcessor:
    def __init__(
        self,
        func: Callable,
        batch_size: int,
        timeout: float,
        run_with_context: Callable | None = None,
    ):
        self._func = func
        self._batch_size = batch_size
        self._timeout = timeout
        self._run_with_context = run_with_context
        self._queue: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None

    async def start(self):
        self._queue = asyncio.Queue()
        self._task = asyncio.create_task(self._batch_loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def submit(self, *args, **kwargs) -> Any:
        future = asyncio.get_event_loop().create_future()
        await self._queue.put((args, kwargs, future))
        return await future

    async def _batch_loop(self):
        while True:
            batch: list[tuple] = []
            try:
                item = await self._queue.get()
                batch.append(item)

                deadline = asyncio.get_event_loop().time() + self._timeout
                while len(batch) < self._batch_size:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(
                            self._queue.get(), timeout=remaining
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
            except asyncio.CancelledError:
                for _, _, future in batch:
                    if not future.done():
                        future.cancel()
                raise

            batched_args = [item[0] for item in batch]
            batched_kwargs = [item[1] for item in batch]

            try:
                if self._run_with_context:
                    results = await run_in_threadpool(
                        self._run_with_context,
                        self._func,
                        batched_args,
                        batched_kwargs,
                    )
                else:
                    results = await run_in_threadpool(
                        self._func, batched_args, batched_kwargs
                    )

                if not isinstance(results, (list, tuple)):
                    results = [results] * len(batch)

                for (_, _, future), result in zip(batch, results):
                    if not future.done():
                        future.set_result(result)
            except Exception as exc:
                for _, _, future in batch:
                    if not future.done():
                        future.set_exception(exc)
