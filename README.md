# `gradio.App` — Low-Level FastAPI Entrypoint

`gradio.App` is a lower-level entrypoint for Gradio that gives you direct access to the underlying FastAPI application. Use it when you need full control: add custom routes that return pages entirely in HTML/JS/CSS, define REST endpoints with Pydantic validation and dependency injection, or mix standard web routes with Gradio's backend features like GPU management, request queuing, batching, and streaming.

## Quickstart

```python
from gradio import App

app = App()

@app.gpu()
@app.api(name="generate", concurrency_limit=2)
def generate(prompt: str):
    for token in model.generate(prompt):
        yield token  # streams via SSE

@app.get("/")
async def root():
    return {"message": "Hello World"}

app.launch()
```

Since `App` extends FastAPI, everything you know works: `@app.get()`, `@app.post()`, path params, query params, `Depends()`, `APIRouter`, Pydantic models, `/docs`, `/openapi.json`.

## Features

- **Full FastAPI access** — `App` subclasses FastAPI. All FastAPI features work: Pydantic validation, dependency injection, OpenAPI docs, middleware, routers.
- **Custom HTML/JS/CSS routes** — Serve pages built entirely with raw HTML, JavaScript, and CSS alongside Gradio components.
- **`@app.gpu(device=N)`** — Runs your function inside a `torch.cuda.device` context. Auto-assigns GPUs round-robin, or pin to a specific device.
- **`@app.cpu(concurrency_limit=N)`** — Marks CPU-bound functions with HTTP-layer concurrency limiting.
- **`@app.api(name="...")`** — Auto-generates a POST endpoint at `/api/{name}` from the function signature.
- **Streaming** — Functions that `yield` automatically stream responses via SSE.
- **Batching** — `@app.gpu(batch_size=8, batch_timeout=0.05)` collects concurrent requests and dispatches them as a single batch.
- **Queue** — Built-in request queue with position tracking, ETA estimation, and SSE status updates.
- **GPU Health** — Built-in `/health/gpu` endpoint with memory, utilization, and temperature stats.

## Custom HTML Routes

Serve full HTML pages alongside your Gradio backend:

```python
from gradio import App
from fastapi.responses import HTMLResponse

app = App()

@app.get("/", response_class=HTMLResponse)
async def homepage():
    return """
    <html>
      <head><script src="/static/app.js"></script></head>
      <body>
        <h1>My App</h1>
        <div id="root"></div>
      </body>
    </html>
    """

@app.gpu()
@app.api(name="predict", concurrency_limit=2)
def predict(text: str):
    return model(text)

app.launch()
```

Your HTML/JS frontend can call the generated `/api/predict` endpoint directly, giving you full control over the UI while leveraging Gradio's backend for GPU management and queuing.

## Using with FastAPI

Anywhere you use `FastAPI()`, you can use `App()` instead:

```python
# Before
from fastapi import FastAPI, Depends
app = FastAPI()

# After
from gradio import App
from fastapi import Depends
app = App()
```

Everything works: path parameters, query parameters, Pydantic request/response models, dependency injection, middleware, `APIRouter`, OpenAPI docs at `/docs`, and more. `App` just adds ML-specific features on top.

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

Add `concurrency_limit` to `@app.api()` to enable a request queue with position tracking and ETA:

```python
@app.gpu()
@app.api(name="predict", concurrency_limit=2)
def predict(text: str):
    return model(text)
```

Clients interact with the queue via two endpoints:

```
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

## Mounting Gradio UIs

Mount a Gradio app alongside your custom routes:

```python
import gradio as gr
from gradio import App

app = App()

@app.gpu()
@app.api(name="predict", concurrency_limit=2)
def predict(text: str):
    return model(text)

demo = gr.Interface(predict, "text", "text")
gr.mount_gradio_app(app, demo, path="/demo")

app.launch()
```

## Requirements

- Python 3.10+
- `fastapi`, `uvicorn` (installed automatically)
- `torch` (optional, for GPU features)
- `nvidia-ml-py` (optional, for detailed GPU health stats)
