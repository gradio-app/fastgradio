# FastGradio

FastGradio is a Python web framework for ML inference, built on Starlette. It makes it trivial to build production ML APIs with explicit GPU/CPU resource management, automatic request batching, streaming, and health monitoring — all with a decorator-based API that feels like FastAPI.

## Quickstart

```bash
pip install fastgradio
```

```python
from fastgradio import App

app = App()

@app.gpu()
@app.api(name="generate")
def generate(prompt: str):
    for token in model.generate(prompt):
        yield token  # streams via SSE

@app.cpu(concurrency_limit=50)
@app.api(name="health_check")
def health_check():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "Hello from FastGradio"}

app.launch()
```

## Features

- **`@app.gpu(device=N)`** — Runs your function inside a `torch.cuda.device` context. Auto-assigns GPUs round-robin, or pin to a specific device.
- **`@app.cpu(concurrency_limit=N)`** — Marks CPU-bound functions with HTTP-layer concurrency limiting.
- **`@app.api(name="...")`** — Auto-generates a POST endpoint at `/api/{name}` from the function signature.
- **`@app.get()` / `@app.post()`** — Standard route decorators with automatic JSON response wrapping.
- **Streaming** — Functions that `yield` automatically stream responses via SSE.
- **Batching** — `@app.gpu(batch_size=8, batch_timeout=0.05)` collects concurrent requests and dispatches them as a single batch.
- **GPU Health** — Built-in `/health/gpu` endpoint with memory, utilization, and temperature stats.

## GPU Batching

```python
@app.gpu(batch_size=8, batch_timeout=0.05)
@app.api(name="predict")
def predict(images: list[bytes]) -> list[str]:
    return model(images)  # called once with up to 8 inputs
```

## Multi-GPU

```python
@app.gpu(device=0)
@app.api(name="model_a")
def model_a(text: str):
    ...

@app.gpu(device=1)
@app.api(name="model_b")
def model_b(text: str):
    ...
```

## Requirements

- Python 3.10+
- `starlette`, `uvicorn` (installed automatically)
- `torch` (optional, for GPU features)
- `nvidia-ml-py` (optional, for detailed GPU health stats)
