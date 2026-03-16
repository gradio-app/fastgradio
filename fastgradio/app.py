from __future__ import annotations

import asyncio
import functools
import inspect
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from ._utils import parse_params_from_body
from .batching import BatchProcessor
from .concurrency import ConcurrencyLimiter
from .decorators import FunctionMeta, _detect_generator, _get_or_create_meta
from .gpu import GPUManager
from .health import build_health_endpoint
from .queue import QueueProcessor
from .streaming import make_streaming_response


class App(FastAPI):
    def __init__(self, **kwargs):
        self.gpu_manager = GPUManager()
        self._registered_functions: dict[str, FunctionMeta] = {}
        self._batch_processors: dict[str, BatchProcessor] = {}
        self._concurrency_limiter = ConcurrencyLimiter()
        self._queue_processor: QueueProcessor | None = None
        self._queue_routes_registered: bool = False

        if "lifespan" not in kwargs:
            kwargs["lifespan"] = self._default_lifespan

        super().__init__(**kwargs)

        # Register health endpoint eagerly
        health_endpoint = build_health_endpoint(self.gpu_manager, self._registered_functions)
        self.add_route("/health/gpu", health_endpoint, methods=["GET"])

    @asynccontextmanager
    async def _default_lifespan(self, app):
        self.gpu_manager.initialize()

        for processor in self._batch_processors.values():
            await processor.start()

        if self._queue_processor:
            await self._queue_processor.start()

        yield

        if self._queue_processor:
            await self._queue_processor.stop()

        for processor in self._batch_processors.values():
            await processor.stop()

    # ── ML Decorators ──────────────────────────────────────────

    def gpu(
        self,
        *,
        device: int | None = None,
        batch_size: int | None = None,
        batch_timeout: float = 0.01,
    ):
        def decorator(func):
            meta = _get_or_create_meta(func)
            meta.compute_type = "gpu"
            meta.device = device
            meta.is_generator = _detect_generator(func)

            if batch_size:
                meta.batch_size = batch_size
                meta.batch_timeout = batch_timeout

            assigned_device = device

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                dev = assigned_device
                if dev is None:
                    dev = self.gpu_manager.allocate_device()
                else:
                    dev = self.gpu_manager.allocate_device(dev)

                if dev is not None:
                    with self.gpu_manager.device_context(dev):
                        return func(*args, **kwargs)
                return func(*args, **kwargs)

            wrapper._fastgradio_meta = meta
            self._registered_functions[func.__name__] = meta

            if batch_size:
                self._batch_processors[func.__name__] = BatchProcessor(
                    func=func,
                    batch_size=batch_size,
                    timeout=batch_timeout,
                    run_with_context=self._make_gpu_runner(assigned_device),
                )

            return wrapper

        return decorator

    def cpu(self, *, concurrency_limit: int | None = None):
        def decorator(func):
            meta = _get_or_create_meta(func)
            meta.compute_type = "cpu"
            meta.concurrency_limit = concurrency_limit
            meta.is_generator = _detect_generator(func)
            func._fastgradio_meta = meta
            self._registered_functions[func.__name__] = meta
            return func

        return decorator

    def api(
        self,
        *,
        name: str | None = None,
        method: str = "POST",
        queue: bool = False,
        concurrency_limit: int | None = None,
    ):
        def decorator(func):
            meta = _get_or_create_meta(func)
            api_name = name or func.__name__
            meta.api_name = api_name
            meta.api_method = method

            use_queue = queue or concurrency_limit is not None
            if use_queue:
                meta.queue_enabled = True
                meta.queue_concurrency = concurrency_limit or 1
                self._ensure_queue_routes()
                handler = self._build_queue_handler(func, meta)
                self._queue_processor.register(
                    api_name, handler, meta.queue_concurrency
                )

            endpoint = self._build_endpoint(func, meta)
            path = f"/api/{api_name}"
            self.add_route(path, endpoint, methods=[method])
            return func

        return decorator

    def launch(
        self,
        host: str = "0.0.0.0",
        port: int = 7860,
        workers: int = 1,
        log_level: str = "info",
        **kwargs,
    ):
        import uvicorn

        uvicorn.run(self, host=host, port=port, workers=workers, log_level=log_level, **kwargs)

    # ── Queue endpoints ────────────────────────────────────────

    def _ensure_queue_routes(self):
        if self._queue_routes_registered:
            return
        self._queue_processor = QueueProcessor()
        self.add_route("/queue/join", self._queue_join_endpoint, methods=["POST"])
        self.add_route("/queue/data", self._queue_data_endpoint, methods=["GET"])
        self._queue_routes_registered = True

    async def _queue_join_endpoint(self, request: Request):
        body = await request.json()
        endpoint = body.get("endpoint")
        data = body.get("data", {})
        if not endpoint:
            return JSONResponse(
                {"msg": "error", "message": "Missing 'endpoint' field"},
                status_code=400,
            )
        result = await self._queue_processor.join(endpoint, data)
        if result.get("msg") == "error":
            return JSONResponse(result, status_code=400)
        return JSONResponse(result)

    async def _queue_data_endpoint(self, request: Request):
        event_id = request.query_params.get("event_id")
        if not event_id:
            return JSONResponse(
                {"msg": "error", "message": "Missing 'event_id' query param"},
                status_code=400,
            )

        async def sse_stream():
            async for msg in self._queue_processor.listen(event_id):
                yield msg

        return StreamingResponse(sse_stream(), media_type="text/event-stream")

    # ── Internal ────────────────────────────────────────────────

    def _make_gpu_runner(self, preferred_device: int | None):
        def run_with_context(func, *args, **kwargs):
            dev = self.gpu_manager.allocate_device(preferred_device)
            if dev is not None:
                with self.gpu_manager.device_context(dev):
                    return func(*args, **kwargs)
            return func(*args, **kwargs)

        return run_with_context

    def _build_queue_handler(self, func: Callable, meta: FunctionMeta):
        """Build an async callable for the queue processor to invoke."""
        sig = inspect.signature(func)

        async def handler(data: dict):
            kwargs = parse_params_from_body(sig, data)

            if meta.batch_size and func.__name__ in self._batch_processors:
                return await self._batch_processors[func.__name__].submit(**kwargs)

            if meta.is_generator:
                async def stream():
                    if inspect.isgeneratorfunction(func):
                        gen = func(**kwargs)
                        async for item in iterate_in_threadpool(gen):
                            yield item
                    else:
                        async for item in func(**kwargs):
                            yield item
                return stream()

            if inspect.iscoroutinefunction(func):
                return await func(**kwargs)
            return await run_in_threadpool(func, **kwargs)

        return handler

    def _build_endpoint(self, func: Callable, meta: FunctionMeta):
        sig = inspect.signature(func)

        async def endpoint(request: Request):
            if request.method == "POST":
                body = await request.json()
            else:
                body = dict(request.query_params)

            kwargs = parse_params_from_body(sig, body)

            semaphore = None
            if meta.concurrency_limit and not meta.queue_enabled:
                semaphore = self._concurrency_limiter.get_semaphore(
                    func.__name__, meta.concurrency_limit
                )

            async def _call():
                if meta.batch_size and func.__name__ in self._batch_processors:
                    return await self._batch_processors[func.__name__].submit(**kwargs)

                if meta.is_generator:
                    return make_streaming_response(func, kwargs)

                if inspect.iscoroutinefunction(func):
                    return await func(**kwargs)
                return await run_in_threadpool(func, **kwargs)

            if semaphore:
                async with semaphore:
                    result = await _call()
            else:
                result = await _call()

            if isinstance(result, (JSONResponse, StreamingResponse)):
                return result
            return JSONResponse({"data": result})

        return endpoint
