from __future__ import annotations

import asyncio
import functools
import inspect
from contextlib import asynccontextmanager
from typing import Any, Callable

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ._utils import auto_response, parse_params_from_body
from .batching import BatchProcessor
from .concurrency import ConcurrencyLimiter
from .decorators import FunctionMeta, _detect_generator, _get_or_create_meta
from .gpu import GPUManager
from .health import build_health_endpoint
from .streaming import make_streaming_response


class App(Starlette):
    def __init__(self, *, debug: bool = False, lifespan=None, **kwargs):
        self.gpu_manager = GPUManager()
        self._registered_functions: dict[str, FunctionMeta] = {}
        self._batch_processors: dict[str, BatchProcessor] = {}
        self._concurrency_limiter = ConcurrencyLimiter()
        self._pending_routes: list[Route] = []

        if lifespan is None:
            lifespan = self._default_lifespan

        super().__init__(debug=debug, lifespan=lifespan, **kwargs)

        # Register health endpoint eagerly so Starlette includes it in routing
        health_endpoint = build_health_endpoint(self.gpu_manager, self._registered_functions)
        self.add_route("/health/gpu", health_endpoint, methods=["GET"])

    @asynccontextmanager
    async def _default_lifespan(self, app):
        self.gpu_manager.initialize()

        for processor in self._batch_processors.values():
            await processor.start()

        yield

        for processor in self._batch_processors.values():
            await processor.stop()

    # ── Decorators ──────────────────────────────────────────────

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

            assigned_device = device  # may be None for auto-assign

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

    def api(self, *, name: str | None = None, method: str = "POST"):
        def decorator(func):
            meta = _get_or_create_meta(func)
            api_name = name or func.__name__
            meta.api_name = api_name
            meta.api_method = method

            endpoint = self._build_endpoint(func, meta)
            path = f"/api/{api_name}"
            self.add_route(path, endpoint, methods=[method])
            return func

        return decorator

    def get(self, path: str, **kwargs):
        def decorator(func):
            endpoint = self._build_route_endpoint(func)
            self.add_route(path, endpoint, methods=["GET"], **kwargs)
            return func

        return decorator

    def post(self, path: str, **kwargs):
        def decorator(func):
            endpoint = self._build_route_endpoint(func)
            self.add_route(path, endpoint, methods=["POST"], **kwargs)
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

    # ── Internal ────────────────────────────────────────────────

    def _make_gpu_runner(self, preferred_device: int | None):
        def run_with_context(func, *args, **kwargs):
            dev = self.gpu_manager.allocate_device(preferred_device)
            if dev is not None:
                with self.gpu_manager.device_context(dev):
                    return func(*args, **kwargs)
            return func(*args, **kwargs)

        return run_with_context

    def _build_endpoint(self, func: Callable, meta: FunctionMeta):
        sig = inspect.signature(func)

        async def endpoint(request: Request):
            if request.method == "POST":
                body = await request.json()
            else:
                body = dict(request.query_params)

            kwargs = parse_params_from_body(sig, body)

            # Concurrency limiting
            semaphore = None
            if meta.concurrency_limit:
                semaphore = self._concurrency_limiter.get_semaphore(
                    func.__name__, meta.concurrency_limit
                )

            async def _call():
                # Batching
                if meta.batch_size and func.__name__ in self._batch_processors:
                    return await self._batch_processors[func.__name__].submit(**kwargs)

                # Streaming
                if meta.is_generator:
                    return make_streaming_response(func, kwargs)

                # Sync or async
                if inspect.iscoroutinefunction(func):
                    return await func(**kwargs)
                return await run_in_threadpool(func, **kwargs)

            if semaphore:
                async with semaphore:
                    result = await _call()
            else:
                result = await _call()

            if isinstance(result, (JSONResponse,)):
                return result
            from starlette.responses import StreamingResponse

            if isinstance(result, StreamingResponse):
                return result
            return JSONResponse({"data": result})

        return endpoint

    def _build_route_endpoint(self, func: Callable):
        async def endpoint(request: Request):
            if inspect.iscoroutinefunction(func):
                result = await func()
            else:
                result = await run_in_threadpool(func)
            return auto_response(result)

        return endpoint
