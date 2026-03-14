from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse

from .gpu import GPUManager


def build_health_endpoint(gpu_manager: GPUManager, registered_functions: dict):
    async def gpu_health(request: Request):
        gpu_info = gpu_manager.get_device_info()

        functions = {}
        for name, meta in registered_functions.items():
            if meta.compute_type == "gpu":
                functions[name] = {
                    "device": meta.device,
                    "batch_size": meta.batch_size,
                }

        # Try pynvml for richer info
        enriched = _enrich_with_pynvml(gpu_info)

        return JSONResponse({
            "gpus": enriched if enriched else gpu_info,
            "functions": functions,
        })

    return gpu_health


def _enrich_with_pynvml(gpu_info: list[dict]) -> list[dict] | None:
    try:
        import pynvml
        pynvml.nvmlInit()
        result = []
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            result.append({
                "device": i,
                "name": pynvml.nvmlDeviceGetName(handle),
                "memory_total_mb": mem.total // (1024 * 1024),
                "memory_used_mb": mem.used // (1024 * 1024),
                "memory_free_mb": mem.free // (1024 * 1024),
                "utilization_percent": util.gpu,
                "temperature_c": temp,
            })
        pynvml.nvmlShutdown()
        return result
    except (ImportError, Exception):
        return None
