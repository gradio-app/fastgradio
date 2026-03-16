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

