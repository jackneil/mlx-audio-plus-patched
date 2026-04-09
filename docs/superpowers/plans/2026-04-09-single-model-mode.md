# Single-Model Serving Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--model` flag to the MLX Audio server that pre-loads a single model at startup and rejects requests for other models, matching vLLM-MLX's single-model serving pattern.

**Architecture:** Module-level `_served_model` global set by `main()` before uvicorn starts. `_validate_model_name()` enforces the constraint at each endpoint. When `--model` is set, `uvicorn.run(app, ...)` receives the app object directly (single process, no fork) — matching vLLM-MLX. Without `--model`, all behavior is unchanged.

**Tech Stack:** FastAPI, uvicorn, pytest, FastAPI TestClient

**Spec:** `docs/superpowers/specs/2026-04-09-single-model-mode-design.md`

---

## File Map

- **Modify:** `mlx_audio/server.py` — all changes live here (global, validation, endpoints, main)
- **Modify:** `mlx_audio/tests/test_server.py` — all new tests

---

### Task 1: Core Infrastructure — Global and Validation Function

**Files:**
- Modify: `mlx_audio/tests/test_server.py`
- Modify: `mlx_audio/server.py:191`

- [ ] **Step 1: Write tests for `_validate_model_name` and the `_served_model` global**

Add to `mlx_audio/tests/test_server.py`:

```python
import mlx_audio.server as server_module


@pytest.fixture
def single_model_mode():
    """Activate single-model mode for the duration of a test."""
    original = server_module._served_model
    server_module._served_model = "mlx-community/test-served-model"
    yield "mlx-community/test-served-model"
    server_module._served_model = original


def test_served_model_default_is_none():
    assert server_module._served_model is None


def test_validate_model_name_passes_when_no_served_model():
    """No-op when _served_model is None (multi-model mode)."""
    server_module._validate_model_name("any-model")  # should not raise


def test_validate_model_name_passes_when_matching(single_model_mode):
    server_module._validate_model_name(single_model_mode)  # should not raise


def test_validate_model_name_rejects_mismatch(single_model_mode):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        server_module._validate_model_name("wrong-model")
    assert exc_info.value.status_code == 400
    detail = exc_info.value.detail
    assert "error" in detail
    assert single_model_mode in detail["error"]["message"]
    assert "wrong-model" in detail["error"]["message"]
    assert detail["error"]["served_model"] == single_model_mode
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest mlx_audio/tests/test_server.py::test_served_model_default_is_none mlx_audio/tests/test_server.py::test_validate_model_name_passes_when_no_served_model mlx_audio/tests/test_server.py::test_validate_model_name_passes_when_matching mlx_audio/tests/test_server.py::test_validate_model_name_rejects_mismatch -v`

Expected: FAIL — `_served_model` and `_validate_model_name` don't exist yet.

- [ ] **Step 3: Implement the global and validation function**

In `mlx_audio/server.py`, add after line 191 (`model_provider = ModelProvider()`):

```python
# Single-model mode state (set by main() before uvicorn starts)
_served_model: str | None = None


def _validate_model_name(request_model: str) -> None:
    """Reject requests for models other than the served model.

    No-op when _served_model is None (multi-model mode).
    """
    if _served_model is not None and request_model != _served_model:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (
                        f"This server is configured to serve only '{_served_model}'. "
                        f"Requested model '{request_model}' is not available "
                        "on this instance."
                    ),
                    "type": "invalid_request_error",
                    "served_model": _served_model,
                }
            },
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest mlx_audio/tests/test_server.py::test_served_model_default_is_none mlx_audio/tests/test_server.py::test_validate_model_name_passes_when_no_served_model mlx_audio/tests/test_server.py::test_validate_model_name_passes_when_matching mlx_audio/tests/test_server.py::test_validate_model_name_rejects_mismatch -v`

Expected: All 4 PASS.

- [ ] **Step 5: Run existing tests to confirm no regressions**

Run: `python -m pytest mlx_audio/tests/test_server.py -v`

Expected: All existing tests PASS (global defaults to `None`, so no behavior change).

- [ ] **Step 6: Commit**

```bash
git add mlx_audio/server.py mlx_audio/tests/test_server.py
git commit -m "feat(server): add _served_model global and _validate_model_name()"
```

---

### Task 2: GET /v1/models — Single-Model Filtering

**Files:**
- Modify: `mlx_audio/tests/test_server.py`
- Modify: `mlx_audio/server.py:200-216`

- [ ] **Step 1: Write tests for filtered model listing**

Add to `mlx_audio/tests/test_server.py`:

```python
def test_list_models_single_model_mode(client, single_model_mode):
    """In single-model mode, /v1/models returns only the served model."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == single_model_mode
    assert data["data"][0]["object"] == "model"
    assert data["data"][0]["owned_by"] == "system"
    assert isinstance(data["data"][0]["created"], int)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest mlx_audio/tests/test_server.py::test_list_models_single_model_mode -v`

Expected: FAIL — endpoint still queries `model_provider` and returns empty list.

- [ ] **Step 3: Implement filtered listing**

In `mlx_audio/server.py`, replace the `list_models()` function:

```python
@app.get("/v1/models")
async def list_models():
    """
    Get list of models - provided in OpenAI API compliant format.
    """
    if _served_model is not None:
        return {
            "object": "list",
            "data": [
                {
                    "id": _served_model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "system",
                }
            ],
        }

    models = await model_provider.get_available_models()
    models_data = []
    for model in models:
        models_data.append(
            {
                "id": model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "system",
            }
        )
    return {"object": "list", "data": models_data}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest mlx_audio/tests/test_server.py::test_list_models_single_model_mode mlx_audio/tests/test_server.py::test_list_models_empty mlx_audio/tests/test_server.py::test_list_models_with_data -v`

Expected: All 3 PASS (new test passes, existing tests still pass because `_served_model` is `None`).

- [ ] **Step 5: Commit**

```bash
git add mlx_audio/server.py mlx_audio/tests/test_server.py
git commit -m "feat(server): filter /v1/models to served model in single-model mode"
```

---

### Task 3: Management Endpoints — Disable in Single-Model Mode

**Files:**
- Modify: `mlx_audio/tests/test_server.py`
- Modify: `mlx_audio/server.py:219-253`

- [ ] **Step 1: Write tests for 403 responses**

Add to `mlx_audio/tests/test_server.py`:

```python
def test_add_model_blocked_in_single_model_mode(client, single_model_mode):
    response = client.post("/v1/models?model_name=some-other-model")
    assert response.status_code == 403
    assert "single-model mode" in response.json()["detail"].lower()


def test_remove_model_blocked_in_single_model_mode(client, single_model_mode):
    response = client.delete("/v1/models?model_name=some-other-model")
    assert response.status_code == 403
    assert "single-model mode" in response.json()["detail"].lower()


def test_add_model_works_without_single_model_mode(client, mock_model_provider):
    """Existing behavior unchanged when _served_model is None."""
    response = client.post("/v1/models?model_name=test_model")
    assert response.status_code == 200


def test_remove_model_works_without_single_model_mode(client, mock_model_provider):
    """Existing behavior unchanged when _served_model is None."""
    mock_model_provider.remove_model = AsyncMock(return_value=True)
    response = client.delete("/v1/models?model_name=test_model")
    assert response.status_code == 204
```

- [ ] **Step 2: Run tests to verify the 403 tests fail**

Run: `python -m pytest mlx_audio/tests/test_server.py::test_add_model_blocked_in_single_model_mode mlx_audio/tests/test_server.py::test_remove_model_blocked_in_single_model_mode -v`

Expected: FAIL — endpoints don't check `_served_model` yet.

- [ ] **Step 3: Add guards to management endpoints**

In `mlx_audio/server.py`, add a guard at the top of `add_model()`:

```python
@app.post("/v1/models")
async def add_model(model_name: str):
    """
    Add a new model to the API.
    """
    if _served_model is not None:
        raise HTTPException(
            status_code=403,
            detail="Cannot add models in single-model mode. "
            f"This server is configured to serve only '{_served_model}'.",
        )
    model_provider.load_model(model_name)
    return {"status": "success", "message": f"Model {model_name} added successfully"}
```

Add a guard at the top of `remove_model()`:

```python
@app.delete("/v1/models")
async def remove_model(model_name: str):
    """
    Remove a model from the API.
    """
    if _served_model is not None:
        raise HTTPException(
            status_code=403,
            detail="Cannot remove models in single-model mode. "
            f"This server is configured to serve only '{_served_model}'.",
        )
    model_name = unquote(model_name).strip('"')
    removed = await model_provider.remove_model(model_name)
    if removed:
        return Response(status_code=204)
    else:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
```

- [ ] **Step 4: Run all management endpoint tests**

Run: `python -m pytest mlx_audio/tests/test_server.py -k "model" -v`

Expected: All model-related tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mlx_audio/server.py mlx_audio/tests/test_server.py
git commit -m "feat(server): return 403 for add/remove model in single-model mode"
```

---

### Task 4: POST /v1/audio/speech — Model Validation

**Files:**
- Modify: `mlx_audio/tests/test_server.py`
- Modify: `mlx_audio/server.py:319-329`

- [ ] **Step 1: Write tests for TTS model validation**

Add to `mlx_audio/tests/test_server.py`:

```python
def test_tts_speech_rejects_wrong_model(client, single_model_mode, mock_model_provider):
    payload = {"model": "wrong-model", "input": "Hello world"}
    response = client.post("/v1/audio/speech", json=payload)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "error" in detail
    assert "wrong-model" in detail["error"]["message"]
    mock_model_provider.load_model.assert_not_called()


def test_tts_speech_accepts_correct_model(client, single_model_mode, mock_model_provider):
    mock_tts_model = MagicMock()
    mock_tts_model.generate = MagicMock(wraps=sync_mock_audio_stream_generator)
    mock_model_provider.load_model = MagicMock(return_value=mock_tts_model)

    payload = {"model": single_model_mode, "input": "Hello world", "voice": "alloy"}
    response = client.post("/v1/audio/speech", json=payload)
    assert response.status_code == 200
    mock_model_provider.load_model.assert_called_once_with(single_model_mode)
```

- [ ] **Step 2: Run tests to verify the reject test fails**

Run: `python -m pytest mlx_audio/tests/test_server.py::test_tts_speech_rejects_wrong_model -v`

Expected: FAIL — endpoint doesn't validate model yet (returns 200 or mock error, not 400).

- [ ] **Step 3: Add validation to the speech endpoint**

In `mlx_audio/server.py`, add `_validate_model_name()` call at the top of `tts_speech()`:

```python
@app.post("/v1/audio/speech")
async def tts_speech(payload: SpeechRequest):
    """Generate speech audio following the OpenAI text-to-speech API."""
    _validate_model_name(payload.model)
    model = model_provider.load_model(payload.model)
    return StreamingResponse(
        generate_audio(model, payload),
        media_type=f"audio/{payload.response_format}",
        headers={
            "Content-Disposition": f"attachment; filename=speech.{payload.response_format}"
        },
    )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest mlx_audio/tests/test_server.py::test_tts_speech_rejects_wrong_model mlx_audio/tests/test_server.py::test_tts_speech_accepts_correct_model mlx_audio/tests/test_server.py::test_tts_speech -v`

Expected: All 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add mlx_audio/server.py mlx_audio/tests/test_server.py
git commit -m "feat(server): validate model in POST /v1/audio/speech"
```

---

### Task 5: POST /v1/audio/transcriptions — Model Validation

**Files:**
- Modify: `mlx_audio/tests/test_server.py`
- Modify: `mlx_audio/server.py:364-426`

- [ ] **Step 1: Write tests for STT model validation**

Add to `mlx_audio/tests/test_server.py`. We need a helper to create a test audio file:

```python
def _make_test_audio():
    """Create a minimal MP3 audio buffer for testing."""
    sample_rate = 16000
    t = np.linspace(0, 1, sample_rate, False)
    audio_data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buffer = io.BytesIO()
    audio_write(buffer, audio_data, sample_rate, format="mp3")
    buffer.seek(0)
    return buffer


def test_stt_transcriptions_rejects_wrong_model(
    client, single_model_mode, mock_model_provider
):
    buffer = _make_test_audio()
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.mp3", buffer, "audio/mp3")},
        data={"model": "wrong-model"},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "error" in detail
    assert "wrong-model" in detail["error"]["message"]
    mock_model_provider.load_model.assert_not_called()


def test_stt_transcriptions_accepts_correct_model(
    client, single_model_mode, mock_model_provider
):
    mock_stt_model = MagicMock()
    mock_stt_model.generate = MagicMock(
        return_value={"text": "Test transcription."}
    )
    mock_model_provider.load_model = MagicMock(return_value=mock_stt_model)

    buffer = _make_test_audio()
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.mp3", buffer, "audio/mp3")},
        data={"model": single_model_mode},
    )
    assert response.status_code == 200
    mock_model_provider.load_model.assert_called_once_with(single_model_mode)
```

- [ ] **Step 2: Run tests to verify the reject test fails**

Run: `python -m pytest mlx_audio/tests/test_server.py::test_stt_transcriptions_rejects_wrong_model -v`

Expected: FAIL — endpoint doesn't validate model yet.

- [ ] **Step 3: Add validation to the transcriptions endpoint**

In `mlx_audio/server.py`, add `_validate_model_name(model)` early in `stt_transcriptions()`, before reading the file or loading the model. Add it right after the function signature and docstring, before creating the `TranscriptionRequest`:

Add `_validate_model_name(model)` as the first line inside the function body, immediately after the docstring and before the `TranscriptionRequest` construction:

```python
    """Transcribe audio using an STT model in OpenAI format."""
    _validate_model_name(model)

    # Create TranscriptionRequest from form fields
    payload = TranscriptionRequest(
```

Only this one line is added. Everything else in the function stays exactly as-is.

- [ ] **Step 4: Run tests**

Run: `python -m pytest mlx_audio/tests/test_server.py::test_stt_transcriptions_rejects_wrong_model mlx_audio/tests/test_server.py::test_stt_transcriptions_accepts_correct_model mlx_audio/tests/test_server.py::test_stt_transcriptions -v`

Expected: All 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add mlx_audio/server.py mlx_audio/tests/test_server.py
git commit -m "feat(server): validate model in POST /v1/audio/transcriptions"
```

---

### Task 6: WebSocket — Model Validation

**Files:**
- Modify: `mlx_audio/tests/test_server.py`
- Modify: `mlx_audio/server.py:429-450`

- [ ] **Step 1: Write tests for WebSocket model validation**

Add to `mlx_audio/tests/test_server.py`:

```python
def test_websocket_rejects_wrong_model(client, single_model_mode):
    with client.websocket_connect("/v1/audio/transcriptions/realtime") as ws:
        ws.send_json({"model": "wrong-model", "sample_rate": 16000})
        response = ws.receive_json()
        assert response["status"] == "error"
        assert "wrong-model" in response["error"]


def test_websocket_defaults_to_served_model(
    client, single_model_mode, mock_model_provider
):
    """When model is omitted from config, use _served_model in single-model mode."""
    mock_stt_model = MagicMock()
    mock_model_provider.load_model = MagicMock(return_value=mock_stt_model)

    with client.websocket_connect("/v1/audio/transcriptions/realtime") as ws:
        ws.send_json({"sample_rate": 16000})  # no "model" field
        response = ws.receive_json()
        assert response["status"] == "ready"
        mock_model_provider.load_model.assert_called_once_with(single_model_mode)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest mlx_audio/tests/test_server.py::test_websocket_rejects_wrong_model mlx_audio/tests/test_server.py::test_websocket_defaults_to_served_model -v`

Expected: FAIL — WebSocket doesn't validate or default to `_served_model`.

- [ ] **Step 3: Add validation to the WebSocket endpoint**

In `mlx_audio/server.py`, modify the config handling section of `stt_realtime_transcriptions()` (around lines 436-449):

```python
    try:
        # Receive initial configuration
        config = await websocket.receive_json()

        # In single-model mode, default to served model; otherwise keep
        # the existing hardcoded default for backwards compatibility.
        default_model = (
            _served_model
            if _served_model is not None
            else "mlx-community/whisper-large-v3-turbo-asr-fp16"
        )
        model_name = config.get("model", default_model)
        language = config.get("language", None)
        sample_rate = config.get("sample_rate", 16000)

        # Validate model in single-model mode
        if _served_model is not None and model_name != _served_model:
            await websocket.send_json(
                {
                    "error": (
                        f"This server is configured to serve only '{_served_model}'. "
                        f"Requested model '{model_name}' is not available "
                        "on this instance."
                    ),
                    "status": "error",
                }
            )
            await websocket.close()
            return
```

Everything after this block (the `print`, model loading, VAD init, etc.) stays exactly as-is.

- [ ] **Step 4: Run tests**

Run: `python -m pytest mlx_audio/tests/test_server.py::test_websocket_rejects_wrong_model mlx_audio/tests/test_server.py::test_websocket_defaults_to_served_model -v`

Expected: Both PASS.

- [ ] **Step 5: Run all tests to confirm no regressions**

Run: `python -m pytest mlx_audio/tests/test_server.py -v`

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add mlx_audio/server.py mlx_audio/tests/test_server.py
git commit -m "feat(server): validate model in WebSocket realtime transcription"
```

---

### Task 7: main() — `--model` Flag, Pre-load, and Startup Branching

**Files:**
- Modify: `mlx_audio/tests/test_server.py`
- Modify: `mlx_audio/server.py:1` (add `sys` import), `mlx_audio/server.py:796-862` (main function)

- [ ] **Step 1: Write tests for the startup flow**

Add to `mlx_audio/tests/test_server.py`:

```python
from unittest.mock import call


def test_main_with_model_flag(mock_model_provider):
    """--model flag sets global, pre-loads, and calls uvicorn.run(app) directly."""
    mock_model_provider.load_model = MagicMock(return_value="fake-model-obj")

    with (
        patch("sys.argv", ["server", "--model", "mlx-community/test-model", "--port", "9999"]),
        patch("mlx_audio.server.uvicorn") as mock_uvicorn,
    ):
        server_module._served_model = None  # reset
        from mlx_audio.server import main
        main()

        # Global was set
        assert server_module._served_model == "mlx-community/test-model"

        # Model was pre-loaded
        mock_model_provider.load_model.assert_called_once_with(
            "mlx-community/test-model"
        )

        # uvicorn.run called with app object directly (not string)
        mock_uvicorn.run.assert_called_once()
        args, kwargs = mock_uvicorn.run.call_args
        from mlx_audio.server import app as server_app
        assert args[0] is server_app
        assert kwargs["port"] == 9999

    # Cleanup
    server_module._served_model = None


def test_main_without_model_flag(mock_model_provider):
    """Without --model, existing behavior: string import, workers, no pre-load."""
    with (
        patch("sys.argv", ["server", "--port", "9999"]),
        patch("mlx_audio.server.uvicorn") as mock_uvicorn,
    ):
        server_module._served_model = None
        from mlx_audio.server import main
        main()

        # Global stays None
        assert server_module._served_model is None

        # No pre-load
        mock_model_provider.load_model.assert_not_called()

        # uvicorn.run called with string import
        mock_uvicorn.run.assert_called_once()
        args, kwargs = mock_uvicorn.run.call_args
        assert args[0] == "mlx_audio.server:app"

    server_module._served_model = None


def test_main_with_model_flag_load_failure(mock_model_provider):
    """If model fails to load, exit with error."""
    mock_model_provider.load_model = MagicMock(
        side_effect=ValueError("Model not found")
    )

    with (
        patch("sys.argv", ["server", "--model", "bad-model"]),
        patch("mlx_audio.server.uvicorn") as mock_uvicorn,
        pytest.raises(SystemExit) as exc_info,
    ):
        server_module._served_model = None
        from mlx_audio.server import main
        main()

    assert exc_info.value.code == 1
    mock_uvicorn.run.assert_not_called()

    server_module._served_model = None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest mlx_audio/tests/test_server.py::test_main_with_model_flag mlx_audio/tests/test_server.py::test_main_without_model_flag mlx_audio/tests/test_server.py::test_main_with_model_flag_load_failure -v`

Expected: FAIL — `--model` flag doesn't exist in argparse yet.

- [ ] **Step 3: Add `sys` import**

In `mlx_audio/server.py`, add `import sys` to the existing imports (around line 8, after `import json`):

```python
import sys
```

- [ ] **Step 4: Implement the `--model` flag and startup branching in `main()`**

Replace the `main()` function in `mlx_audio/server.py`:

```python
def main():
    parser = argparse.ArgumentParser(description="MLX Audio API server")
    parser.add_argument(
        "--allowed-origins",
        nargs="+",
        default=["*"],
        help="List of allowed origins for CORS",
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="Host to run the server on"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Serve only this model. Pre-loads at startup, rejects requests for "
        "other models. Without this flag, behavior is unchanged (load any model "
        "on demand).",
    )
    parser.add_argument(
        "--reload",
        type=bool,
        default=False,
        help="Enable auto-reload of the server. Only works when 'workers' is set to None.",
    )

    parser.add_argument(
        "--workers",
        type=int_or_float,
        default=calculate_default_workers(),
        help="""Number of workers. Overrides the `MLX_AUDIO_NUM_WORKERS` env variable.
        Can be either an int or a float.
        If an int, it will be the number of workers to use.
        If a float, number of workers will be this fraction of the  number of CPU cores available, with a minimum of 1.
        Defaults to the `MLX_AUDIO_NUM_WORKERS` env variable if set and to 2 if not.
        To use all available CPU cores, set it to 1.0.

        Examples:
        --workers 1 (will use 1 worker)
        --workers 1.0 (will use all available CPU cores)
        --workers 0.5 (will use half the number of CPU cores available)
        --workers 0.0 (will use 1 worker)""",
    )
    parser.add_argument(
        "--start-ui",
        action="store_true",
        help="Start the Studio UI alongside the API server",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to save server logs",
    )

    args = parser.parse_args()

    setup_cors(app, args.allowed_origins)

    if args.model:
        # Single-model mode — matches vLLM-MLX pattern:
        # set module global, pre-load, pass app directly to uvicorn
        global _served_model
        _served_model = args.model

        print(f"Loading model '{args.model}'...")
        load_start = time.time()
        try:
            model_provider.load_model(args.model)
        except Exception as e:
            print(f"Error: Failed to load model '{args.model}': {e}", file=sys.stderr)
            sys.exit(1)
        load_time = time.time() - load_start
        print(f"Model '{args.model}' loaded in {load_time:.1f}s")

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    else:
        # Multi-model mode — existing behavior unchanged
        if isinstance(args.workers, float):
            args.workers = max(1, int(os.cpu_count() * args.workers))

        client = MLXAudioStudioServer(start_ui=args.start_ui, log_dir=args.log_dir)
        client.start_server(
            host=args.host,
            port=args.port,
            reload=args.reload if args.workers is None else False,
            workers=args.workers,
        )
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest mlx_audio/tests/test_server.py::test_main_with_model_flag mlx_audio/tests/test_server.py::test_main_without_model_flag mlx_audio/tests/test_server.py::test_main_with_model_flag_load_failure -v`

Expected: All 3 PASS.

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest mlx_audio/tests/test_server.py -v`

Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add mlx_audio/server.py mlx_audio/tests/test_server.py
git commit -m "feat(server): add --model flag with pre-load and single-process startup"
```

---

### Task 8: Final Verification

- [ ] **Step 1: Run the complete server test suite one final time**

Run: `python -m pytest mlx_audio/tests/test_server.py -v`

Expected: All tests PASS — no regressions, all new tests green.

- [ ] **Step 2: Verify the import works cleanly**

Run: `python -c "from mlx_audio.server import app, _served_model, _validate_model_name; print('OK')"` 

Expected: `OK`

- [ ] **Step 3: Verify --model flag appears in help**

Run: `python -m mlx_audio.server --help`

Expected: `--model MODEL` appears in the help output with the description.

- [ ] **Step 4: Commit any final fixes if needed, then tag completion**

```bash
git log --oneline -10
```

Verify the commit history shows the clean progression of tasks.
