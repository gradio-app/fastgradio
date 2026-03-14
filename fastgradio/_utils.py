from __future__ import annotations

import inspect
from typing import Any

from starlette.responses import JSONResponse, Response


def auto_response(result: Any) -> Response:
    if isinstance(result, Response):
        return result
    return JSONResponse(result)


def parse_params_from_body(sig: inspect.Signature, body: dict) -> dict:
    kwargs = {}
    for param_name, param in sig.parameters.items():
        if param_name in body:
            value = body[param_name]
            annotation = param.annotation
            if annotation != inspect.Parameter.empty and not isinstance(value, annotation):
                try:
                    value = annotation(value)
                except (TypeError, ValueError):
                    pass
            kwargs[param_name] = value
        elif param.default != inspect.Parameter.empty:
            kwargs[param_name] = param.default
    return kwargs
