import io
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from mlx_audio.audio_io import read as audio_read
from mlx_audio.audio_io import write as audio_write

# python-multipart is required for FastAPI file uploads
pytest.importorskip("multipart", reason="python-multipart is required for server tests")

from fastapi.testclient import TestClient

import mlx_audio.server as server_module
from mlx_audio.server import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_model_provider():
    # mock the model_provider.load_model method
    with patch(
        "mlx_audio.server.model_provider", new_callable=AsyncMock
    ) as mock_provider:
        mock_provider.load_model = MagicMock()
        yield mock_provider


def test_list_models_empty(client, mock_model_provider):
    # mock the model_provider.get_available_models method
    mock_model_provider.get_available_models = AsyncMock(return_value=[])
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json() == {"object": "list", "data": []}


def test_list_models_with_data(client, mock_model_provider):
    # Test that the list_models endpoint
    mock_model_provider.get_available_models = AsyncMock(
        return_value=["model1", "model2"]
    )
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 2
    assert data["data"][0]["id"] == "model1"
    assert data["data"][1]["id"] == "model2"


def test_add_model(client, mock_model_provider):
    # Test that the add_model endpoint
    response = client.post("/v1/models?model_name=test_model")
    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": "Model test_model added successfully",
    }
    mock_model_provider.load_model.assert_called_once_with("test_model")


def test_remove_model_success(client, mock_model_provider):
    # Test that the remove_model endpoint returns a 204 status code
    mock_model_provider.remove_model = AsyncMock(return_value=True)
    response = client.delete("/v1/models?model_name=test_model")
    assert response.status_code == 204
    mock_model_provider.remove_model.assert_called_once_with("test_model")


def test_remove_model_not_found(client, mock_model_provider):
    # Test that the remove_model endpoint returns a 404 status code
    mock_model_provider.remove_model = AsyncMock(return_value=False)
    response = client.delete("/v1/models?model_name=non_existent_model")
    assert response.status_code == 404
    assert response.json() == {"detail": "Model 'non_existent_model' not found"}
    mock_model_provider.remove_model.assert_called_once_with("non_existent_model")


def test_remove_model_with_quotes_in_name(client, mock_model_provider):
    # Test that the remove_model endpoint returns a 204 status code
    mock_model_provider.remove_model = AsyncMock(return_value=True)
    response = client.delete('/v1/models?model_name="test_model_quotes"')
    assert response.status_code == 204
    mock_model_provider.remove_model.assert_called_once_with("test_model_quotes")


class MockAudioResult:
    def __init__(self, audio_data, sample_rate):
        self.audio = audio_data
        self.sample_rate = sample_rate


def sync_mock_audio_stream_generator(input_text: str, **kwargs):
    sample_rate = 16000
    duration = 1
    frequency = 440
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
    yield MockAudioResult(audio_data.astype(np.float32), sample_rate)


def test_tts_speech(client, mock_model_provider):
    # Test that the tts_speech endpoint returns a 200 status code
    mock_tts_model = MagicMock()
    mock_tts_model.generate = MagicMock(wraps=sync_mock_audio_stream_generator)

    mock_model_provider.load_model = MagicMock(return_value=mock_tts_model)

    payload = {"model": "test_tts_model", "input": "Hello world", "voice": "alloy"}
    response = client.post("/v1/audio/speech", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"].lower() == "audio/mp3"
    assert (
        response.headers["content-disposition"].lower()
        == "attachment; filename=speech.mp3"
    )

    mock_model_provider.load_model.assert_called_once_with("test_tts_model")
    mock_tts_model.generate.assert_called_once()

    args, kwargs = mock_tts_model.generate.call_args
    assert args[0] == payload["input"]
    assert kwargs.get("voice") == payload["voice"]

    try:
        audio_data, sample_rate = audio_read(io.BytesIO(response.content))
        assert sample_rate > 0
        assert len(audio_data) > 0
    except Exception as e:
        pytest.fail(f"Failed to read or validate MP3 content: {e}")


def test_stt_transcriptions(client, mock_model_provider):
    # Test that the stt_transcriptions endpoint returns a 200 status code
    mock_stt_model = MagicMock()
    mock_stt_model.generate = MagicMock(
        return_value={"text": "This is a test transcription."}
    )

    mock_model_provider.load_model = MagicMock(return_value=mock_stt_model)

    sample_rate = 16000
    duration = 1
    frequency = 440
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

    buffer = io.BytesIO()
    audio_write(buffer, audio_data, sample_rate, format="mp3")
    buffer.seek(0)

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.mp3", buffer, "audio/mp3")},
        data={"model": "test_stt_model"},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "This is a test transcription."}

    mock_model_provider.load_model.assert_called_once_with("test_stt_model")
    mock_stt_model.generate.assert_called_once()

    assert mock_stt_model.generate.call_args[0][0].startswith("/tmp/")


@pytest.fixture
def single_model_mode():
    """Activate single-model mode for the duration of a test."""
    original = server_module._served_model
    server_module._served_model = "mlx-community/test-served-model"
    yield "mlx-community/test-served-model"
    server_module._served_model = original


def test_served_model_default_is_none():
    assert server_module._served_model is None


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


def test_validate_model_name_passes_when_no_served_model():
    """No-op when _served_model is None (multi-model mode)."""
    server_module._validate_model_name("any-model")  # should not raise


def test_validate_model_name_passes_when_matching(single_model_mode):
    server_module._validate_model_name(single_model_mode)  # should not raise


def test_add_model_blocked_in_single_model_mode(client, single_model_mode):
    response = client.post("/v1/models?model_name=some-other-model")
    assert response.status_code == 403
    assert "single-model mode" in response.json()["detail"].lower()


def test_remove_model_blocked_in_single_model_mode(client, single_model_mode):
    response = client.delete("/v1/models?model_name=some-other-model")
    assert response.status_code == 403
    assert "single-model mode" in response.json()["detail"].lower()


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


def test_main_with_model_flag(mock_model_provider):
    """--model flag sets global, pre-loads, and calls uvicorn.run(app) directly."""
    mock_model_provider.load_model = MagicMock(return_value="fake-model-obj")

    with (
        patch("sys.argv", ["server", "--model", "mlx-community/test-model", "--port", "9999"]),
        patch("mlx_audio.server.uvicorn") as mock_uvicorn,
        patch("mlx_audio.server.setup_cors"),
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
        patch("mlx_audio.server.setup_cors"),
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
        patch("mlx_audio.server.setup_cors"),
        pytest.raises(SystemExit) as exc_info,
    ):
        server_module._served_model = None
        from mlx_audio.server import main
        main()

    assert exc_info.value.code == 1
    mock_uvicorn.run.assert_not_called()

    server_module._served_model = None
