from __future__ import annotations

import inspect
import json

from starlette.concurrency import iterate_in_threadpool
from starlette.responses import StreamingResponse


def make_streaming_response(func, kwargs: dict, run_with_context=None):
    async def event_stream():
        if inspect.isgeneratorfunction(func):
            if run_with_context:
                gen = run_with_context(func, **kwargs)
            else:
                gen = func(**kwargs)
            async for item in iterate_in_threadpool(gen):
                yield f"data: {json.dumps(item)}\n\n"
        else:
            if run_with_context:
                agen = run_with_context(func, **kwargs)
            else:
                agen = func(**kwargs)
            async for item in agen:
                yield f"data: {json.dumps(item)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
