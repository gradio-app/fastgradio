from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable


@dataclass
class QueueEvent:
    event_id: str
    endpoint_name: str
    data: dict
    created_at: float
    status: str = "queued"  # queued → processing → complete | error


class ETAEstimator:
    def __init__(self, window: int = 20):
        self._times: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))

    def record(self, endpoint: str, duration: float):
        self._times[endpoint].append(duration)

    def avg_time(self, endpoint: str) -> float | None:
        times = self._times.get(endpoint)
        if not times:
            return None
        return sum(times) / len(times)

    def estimate(self, endpoint: str, rank: int, concurrency: int) -> float | None:
        avg = self.avg_time(endpoint)
        if avg is None:
            return None
        return avg * math.ceil((rank + 1) / max(concurrency, 1))


class QueueProcessor:
    def __init__(self):
        self._queues: dict[str, deque[QueueEvent]] = defaultdict(deque)
        self._subscribers: dict[str, asyncio.Queue[dict]] = {}
        self._concurrency_limits: dict[str, int] = {}
        self._active_count: dict[str, int] = defaultdict(int)
        self._handlers: dict[str, Callable] = {}
        self._eta = ETAEstimator()
        self._task: asyncio.Task | None = None

    def register(self, endpoint: str, handler: Callable, concurrency_limit: int):
        self._handlers[endpoint] = handler
        self._concurrency_limits[endpoint] = concurrency_limit

    async def start(self):
        self._task = asyncio.create_task(self._dispatch_loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def join(self, endpoint: str, data: dict) -> dict:
        if endpoint not in self._handlers:
            return {"event_id": None, "msg": "error", "message": f"Unknown endpoint: {endpoint}"}

        event_id = uuid.uuid4().hex
        event = QueueEvent(
            event_id=event_id,
            endpoint_name=endpoint,
            data=data,
            created_at=time.monotonic(),
        )
        self._subscribers[event_id] = asyncio.Queue()
        self._queues[endpoint].append(event)

        rank = len(self._queues[endpoint]) - 1
        queue_size = sum(len(q) for q in self._queues.values())
        concurrency = self._concurrency_limits.get(endpoint, 1)

        await self._send(event_id, {
            "msg": "estimation",
            "rank": rank,
            "queue_size": queue_size,
            "rank_eta": self._eta.estimate(endpoint, rank, concurrency),
        })

        return {"event_id": event_id}

    async def listen(self, event_id: str) -> AsyncGenerator[str, None]:
        sub = self._subscribers.get(event_id)
        if sub is None:
            yield _sse({"msg": "unexpected_error", "message": "Unknown event_id"})
            return

        try:
            while True:
                try:
                    msg = await asyncio.wait_for(sub.get(), timeout=15.0)
                    yield _sse(msg)
                    if msg.get("msg") in ("process_completed", "unexpected_error"):
                        break
                except asyncio.TimeoutError:
                    yield _sse({"msg": "heartbeat"})
        finally:
            self._subscribers.pop(event_id, None)

    async def _dispatch_loop(self):
        endpoints = list(self._handlers.keys())
        while True:
            dispatched = False
            for ep in endpoints:
                limit = self._concurrency_limits.get(ep, 1)
                if self._active_count[ep] >= limit:
                    continue
                queue = self._queues.get(ep)
                if not queue:
                    continue

                event = queue.popleft()
                event.status = "processing"
                self._active_count[ep] += 1
                dispatched = True

                # Broadcast updated estimations to remaining queued events
                await self._broadcast_estimations(ep)

                asyncio.create_task(self._process_event(event))

            # Re-scan handler list in case new endpoints were registered
            endpoints = list(self._handlers.keys())

            if not dispatched:
                await asyncio.sleep(0.05)

    async def _process_event(self, event: QueueEvent):
        handler = self._handlers[event.endpoint_name]
        t0 = time.monotonic()
        eta = self._eta.avg_time(event.endpoint_name)

        await self._send(event.event_id, {
            "msg": "process_starts",
            "eta": eta,
        })

        try:
            result = await handler(event.data)

            # Check if result is an async generator (streaming)
            if hasattr(result, "__aiter__"):
                async for chunk in result:
                    await self._send(event.event_id, {
                        "msg": "process_generating",
                        "output": {"data": chunk},
                        "success": True,
                    })
                await self._send(event.event_id, {
                    "msg": "process_completed",
                    "output": {"data": None},
                    "success": True,
                })
            else:
                event.status = "complete"
                await self._send(event.event_id, {
                    "msg": "process_completed",
                    "output": {"data": result},
                    "success": True,
                })
        except Exception as exc:
            event.status = "error"
            await self._send(event.event_id, {
                "msg": "process_completed",
                "output": {"error": str(exc)},
                "success": False,
            })
        finally:
            duration = time.monotonic() - t0
            self._eta.record(event.endpoint_name, duration)
            self._active_count[event.endpoint_name] -= 1

    async def _broadcast_estimations(self, endpoint: str):
        concurrency = self._concurrency_limits.get(endpoint, 1)
        queue_size = sum(len(q) for q in self._queues.values())
        for rank, event in enumerate(self._queues.get(endpoint, [])):
            if event.event_id in self._subscribers:
                await self._send(event.event_id, {
                    "msg": "estimation",
                    "rank": rank,
                    "queue_size": queue_size,
                    "rank_eta": self._eta.estimate(endpoint, rank, concurrency),
                })

    async def _send(self, event_id: str, msg: dict):
        sub = self._subscribers.get(event_id)
        if sub is not None:
            await sub.put(msg)


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"
