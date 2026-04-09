# Design: Single-Model Serving Mode (`--model` flag)

**Date:** 2026-04-09
**Status:** Approved
**File:** `mlx_audio/server.py`

## Context

The HANK LLM Arena manages model processes as one-model-per-port. vLLM-MLX follows this pattern: `vllm-mlx serve <model> --port <port>`. The mlx-audio server currently has no equivalent — it accepts any model per-request and loads on demand.

This design adds a `--model` flag that mirrors vLLM-MLX's single-model serving pattern, positioning mlx-audio for eventual integration into vLLM.

## Architecture: Matching vLLM-MLX

vLLM-MLX's pattern (from `vllm_mlx/cli.py` and `vllm_mlx/server.py`):

1. CLI parses model as a positional arg
2. Sets module-level globals (`_engine`, `_model_name`) on the server module
3. Calls `load_model()` before uvicorn starts — model is ready before first request
4. Passes `app` object directly to `uvicorn.run()` (single process, no fork)
5. `_validate_model_name()` rejects requests for other models

We replicate this pattern exactly, adapted for mlx-audio's existing structure.

## Module-Level Globals

Add to `server.py` at module level, near the existing `model_provider` singleton:

```python
# Single-model mode state (set by main() before uvicorn starts)
_served_model: str | None = None
```

When `_served_model` is `None`, the server operates in multi-model mode (today's behavior). When set, single-model enforcement is active.

No separate `_model_instance` global is needed — the model lives in `model_provider.models[_served_model]` as it does today. The global only stores the name for validation.

## CLI Changes

Add `--model` to the argparse parser in `main()`:

```
--model MODEL    Serve only this model. Pre-loads at startup, rejects
                 requests for other models. Without this flag, behavior
                 is unchanged (load any model on demand).
```

## Startup Flow

### With `--model`

```
main()
  ├─ parse --model
  ├─ set global: _served_model = args.model
  ├─ pre-load: model_provider.load_model(args.model)
  │   ├─ success → log model name + load time
  │   └─ failure → print error, sys.exit(1)
  └─ uvicorn.run(app, host=..., port=...)
       ← app object directly (not string)
       ← no workers param (single process)
```

### Without `--model`

```
main()
  └─ uvicorn.run("mlx_audio.server:app", ..., workers=N)
       ← string import (current behavior, unchanged)
```

The branch point is a single `if args.model:` in `main()`. The `--model` path passes `app` directly (vLLM-MLX pattern); the other path is today's code untouched.

## Model Validation

A helper function matching vLLM-MLX's `_validate_model_name()`:

```python
def _validate_model_name(request_model: str) -> None:
    if _served_model is not None and request_model != _served_model:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (
                        f"This server is configured to serve only '{_served_model}'. "
                        f"Requested model '{request_model}' is not available on this instance."
                    ),
                    "type": "invalid_request_error",
                    "served_model": _served_model,
                }
            },
        )
```

No model name normalization — exact string match, same as vLLM-MLX.

## Endpoint Changes

### Inference endpoints — add `_validate_model_name()` call

- **`POST /v1/audio/speech`**: Call `_validate_model_name(payload.model)` before `model_provider.load_model()`.
- **`POST /v1/audio/transcriptions`**: Call `_validate_model_name(model)` before `model_provider.load_model()`.
- **`WebSocket /v1/audio/transcriptions/realtime`**: After receiving the config message, validate `config["model"]`. On mismatch, send error JSON and close the socket (no HTTPException — already upgraded to WS).

### Management endpoints — disable in single-model mode

- **`POST /v1/models`** (add model): Return HTTP 403 with message explaining single-model mode is active.
- **`DELETE /v1/models`** (remove model): Return HTTP 403 with same message.

### Listing endpoint — filter to served model

- **`GET /v1/models`**: When `_served_model` is set, return only that model unconditionally (source of truth is the global, not `model_provider`). When not set, current behavior (list loaded models).

Response in single-model mode:

```json
{
  "object": "list",
  "data": [
    {
      "id": "mlx-community/whisper-large-v3-turbo",
      "object": "model",
      "created": 1234567890,
      "owned_by": "system"
    }
  ]
}
```

## Error Handling

**Startup failure:** If `model_provider.load_model()` raises during pre-load, catch the exception, print a clear error message to stderr, and call `sys.exit(1)`. The server must never start in a broken state.

**WebSocket validation:** Since the connection is already upgraded, model mismatch sends a JSON error message (`{"error": "...", "status": "error"}`) and closes the socket gracefully.

**No-op when `--model` absent:** Every guard checks `if _served_model is not None`. When `None`, all code paths are identical to today.

## Usage

```bash
# Single-model mode (arena integration)
python -m mlx_audio.server --host 0.0.0.0 --port 8020 --model mlx-community/whisper-large-v3-turbo

# Multi-model mode (existing behavior, unchanged)
python -m mlx_audio.server --host 0.0.0.0 --port 8020
```

## Verification

1. `python -m mlx_audio.server --model mlx-community/whisper-large-v3-turbo --port 8020` — starts and pre-loads
2. `curl localhost:8020/v1/models` — returns only the served model
3. Transcription with matching model — succeeds
4. Transcription with different model — returns 400
5. `POST /v1/models` — returns 403
6. `DELETE /v1/models` — returns 403
7. Server without `--model` — works exactly as before
