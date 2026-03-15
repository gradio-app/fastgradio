# FastGradio

FastGradio is a Python web framework for ML inference, built on Starlette. It makes it trivial to build production ML APIs with explicit GPU/CPU resource management, automatic request batching, streaming, and health monitoring with a decorator-based API that feels like FastAPI.

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
- **Queue** — Built-in request queue with position tracking, ETA estimation, and SSE status updates.
- **GPU Health** — Built-in `/health/gpu` endpoint with memory, utilization, and temperature stats.
- **Gradio compatible** — Mount Gradio apps on FastGradio, or mount FastGradio on FastAPI.

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

## Request Queue

Add `concurrency_limit` to `@app.api()` to enable a Gradio-style request queue with position tracking and ETA:

```python
@app.gpu()
@app.api(name="predict", concurrency_limit=2)
def predict(text: str):
    return model(text)
```

Clients interact with the queue via two endpoints:

```python
# 1. Join the queue
POST /queue/join  {"endpoint": "predict", "data": {"text": "hello"}}
# Returns: {"event_id": "abc123"}

# 2. Listen for updates via SSE
GET /queue/data?event_id=abc123
# Stream of events:
#   {"msg": "estimation", "rank": 2, "queue_size": 5, "rank_eta": 3.4}
#   {"msg": "process_starts", "eta": 1.2}
#   {"msg": "process_completed", "output": {"data": "result"}, "success": true}
```

The direct endpoint (`POST /api/predict`) still works for non-queued access.

## Mounting Gradio

Mount a Gradio app on FastGradio — works exactly like mounting on FastAPI:

```python
import gradio as gr
from fastgradio import App

app = App()

@app.gpu()
@app.api(name="predict", concurrency_limit=2)
def predict(text: str):
    return model(text)

demo = gr.Interface(predict, "text", "text")
gr.mount_gradio_app(app, demo, path="/demo")

app.launch()
```

## Mounting on FastAPI

FastGradio apps are ASGI applications, so you can mount them directly on a FastAPI app — just like Gradio:

```python
from fastapi import FastAPI
from fastgradio import App

api = FastAPI()

ml = App()

@ml.gpu()
@ml.api(name="predict")
def predict(text: str):
    return model(text)

# Mount the FastGradio app under /ml
api.mount("/ml", ml)
```

All FastGradio routes are now available under the mount path (`/ml/api/predict`, `/ml/health/gpu`, etc.). The parent FastAPI app keeps its own routes, middleware, and docs.

## Requirements

- Python 3.10+
- `starlette`, `uvicorn` (installed automatically)
- `torch` (optional, for GPU features)
- `nvidia-ml-py` (optional, for detailed GPU health stats)
