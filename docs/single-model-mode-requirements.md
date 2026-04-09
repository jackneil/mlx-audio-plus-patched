# Requirements: Single-Model Serving Mode (`--model` flag)

## Problem

The HANK LLM Arena manages model processes uniformly: one model per process per port. The admin UI starts/stops individual models, and the port allocation system assigns each model its own port.

Currently `mlx_audio.server` has no `--model` flag. It accepts any model name per-request and loads it on demand via `ModelProvider.load_model()`. This means:
- Starting the server makes ALL downloaded audio models available
- There's no way to control which model is loaded
- No pre-loading at startup means the first request has high latency (model load time)
- It doesn't fit the arena's one-model-per-port management pattern

## Requirements

### 1. New `--model` CLI flag

Add `--model` argument to the argparse parser in `main()`:

```
--model MODEL_NAME    Serve only this model. Pre-loads at startup,
                      rejects requests for other models. Without this
                      flag, behavior is unchanged (load any model on demand).
```

When `--model` is provided:
- Pre-load the model during server startup (before accepting requests)
- Log the model name and load time at startup
- The `/v1/models` endpoint should only list this one model

### 2. Reject requests for other models

When `--model` is set and a request comes in for a different model:
- Return HTTP 400 with a clear error:
  ```json
  {
    "error": {
      "message": "This server is configured to serve only 'mlx-community/whisper-large-v3-turbo'. Requested model 'mlx-community/parakeet-tdt-0.6b-v3' is not available on this instance.",
      "type": "invalid_request_error",
      "served_model": "mlx-community/whisper-large-v3-turbo"
    }
  }
  ```
- This applies to all endpoints that accept a `model` parameter:
  - `POST /v1/audio/transcriptions` (form field `model`)
  - `POST /v1/audio/speech` (JSON body `model`)
  - `POST /v1/models` (add model endpoint — should be disabled in single-model mode)
  - `DELETE /v1/models` (remove model — should be disabled in single-model mode)
  - `WebSocket /v1/audio/transcriptions/realtime` (config message `model`)

### 3. Pre-load at startup

When `--model` is set, call `model_provider.load_model(model_name)` during startup, before uvicorn starts accepting connections. This eliminates cold-start latency on the first request.

If the model fails to load (not downloaded, incompatible, etc.), the server should exit with a clear error message rather than starting in a broken state.

### 4. Health check compatibility

The arena checks model health by hitting `GET /v1/models` and looking for a model in the response. In single-model mode, this endpoint should return the served model:

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

### 5. No behavior change without `--model`

When `--model` is NOT provided, the server works exactly as it does today: any model can be requested, models are loaded on demand, `/v1/models` lists all loaded models, etc.

## Implementation Notes

The changes are isolated to `mlx_audio/server.py`:

1. **`main()`**: Add `--model` arg to argparse, pass to `MLXAudioStudioServer`
2. **`MLXAudioStudioServer.start_server()`**: If model specified, pre-load before starting uvicorn
3. **`ModelProvider`**: Add an `allowed_model` attribute. When set, `load_model()` raises if a different model is requested. A module-level variable or app state can communicate the `--model` value from `main()` to the `ModelProvider` instance.
4. **`/v1/models`**: Filter to only the allowed model when in single-model mode
5. **`POST /v1/models`** and **`DELETE /v1/models`**: Return 403 when in single-model mode

## Example Usage

```bash
# Single-model mode (for arena integration)
python -m mlx_audio.server --host 0.0.0.0 --port 8020 --model mlx-community/whisper-large-v3-turbo

# Multi-model mode (existing behavior, unchanged)
python -m mlx_audio.server --host 0.0.0.0 --port 8020
```

## Verification

1. `python -m mlx_audio.server --model mlx-community/whisper-large-v3-turbo --port 8020` starts and pre-loads the model
2. `curl localhost:8020/v1/models` returns only Whisper
3. Transcription request with `model=mlx-community/whisper-large-v3-turbo` succeeds
4. Transcription request with `model=mlx-community/parakeet-tdt-0.6b-v3` returns 400
5. Server without `--model` still works as before
