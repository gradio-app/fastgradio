from __future__ import annotations

from contextlib import contextmanager


class GPUManager:
    def __init__(self):
        self._available_devices: list[int] = []
        self._next_device: int = 0
        self._torch = None

    def initialize(self):
        try:
            import torch
            self._torch = torch
            if torch.cuda.is_available():
                self._available_devices = list(range(torch.cuda.device_count()))
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return len(self._available_devices) > 0

    def allocate_device(self, preferred: int | None = None) -> int | None:
        if not self._available_devices:
            return None
        if preferred is not None:
            if preferred in self._available_devices:
                return preferred
            raise ValueError(
                f"GPU device {preferred} not available. "
                f"Available: {self._available_devices}"
            )
        device = self._available_devices[self._next_device % len(self._available_devices)]
        self._next_device += 1
        return device

    @contextmanager
    def device_context(self, device: int):
        if self._torch is not None and self._torch.cuda.is_available():
            with self._torch.cuda.device(device):
                yield
        else:
            yield

    def get_device_info(self) -> list[dict]:
        if not self._available_devices:
            return []
        info = []
        for dev in self._available_devices:
            entry = {"device": dev}
            if self._torch is not None:
                entry["name"] = self._torch.cuda.get_device_name(dev)
                mem = self._torch.cuda.mem_get_info(dev)
                entry["memory_free_mb"] = mem[0] // (1024 * 1024)
                entry["memory_total_mb"] = mem[1] // (1024 * 1024)
                entry["memory_used_mb"] = entry["memory_total_mb"] - entry["memory_free_mb"]
            info.append(entry)
        return info
