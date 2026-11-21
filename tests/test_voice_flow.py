"""Voice input returns text without calling speech synthesis."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from services.router import route_voice_request
from services.openai_client import openai_client
from utils.helpers import cleanup_file


@pytest.mark.asyncio
async def test_voice_routes_transcription_without_tts():
    with patch("services.router.transcribe_voice_message", new=AsyncMock(return_value="Статус проекта?")), patch("services.router.route_text_request", new=AsyncMock(return_value={"text": "Ответ", "mode": "rag"})) as route, patch.object(openai_client, "generate_speech", new=AsyncMock()) as speech:
        result = await route_voice_request(42, Path("voice.ogg"))
        route.assert_awaited_once_with(42, "Статус проекта?")
        assert result == {"text": "Ответ", "mode": "rag", "transcription": "Статус проекта?"}
        speech.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_at", ["stt", "llm"])
async def test_voice_errors_preserve_available_transcription(fail_at):
    with patch("services.router.transcribe_voice_message", new=AsyncMock(return_value="Вопрос")) as stt, patch("services.router.route_text_request", new=AsyncMock()) as route, patch.object(openai_client, "generate_speech", new=AsyncMock()) as speech:
        (stt if fail_at == "stt" else route).side_effect = RuntimeError("offline")
        result = await route_voice_request(42, Path("voice.ogg"))
        assert result["error"] == "offline"
        assert result["transcription"] == (None if fail_at == "stt" else "Вопрос")
        speech.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [
    {"text": "Ответ", "transcription": "Вопрос"},
    {"text": "Ошибка", "error": "offline"},
])
async def test_voice_handler_sends_text_and_cleans_only_real_paths(response, tmp_path):
    from handlers import voice
    path = tmp_path / "voice.ogg"
    message = SimpleNamespace(from_user=SimpleNamespace(id=42), chat=SimpleNamespace(id=42), voice=SimpleNamespace(file_id="voice"))
    with patch.object(voice.bot, "send_chat_action", new=AsyncMock()), patch.object(voice.bot, "get_file", new=AsyncMock(return_value=SimpleNamespace(file_path="voice.ogg"))), patch.object(voice.bot, "download_file", new=AsyncMock(return_value=b"audio")), patch.object(voice, "save_file_async", new=AsyncMock(return_value=path)), patch.object(voice, "route_voice_request", new=AsyncMock(return_value=response)), patch.object(voice.bot, "send_message", new=AsyncMock()) as send, patch.object(voice.bot, "send_voice", new=AsyncMock()) as send_voice, patch.object(voice, "cleanup_files") as cleanup:
        await voice.handle_voice_message(message)
        assert send.call_args_list[-1].args == (42, response["text"])
        assert send.await_count == (2 if response.get("transcription") else 1)
        cleanup.assert_called_once_with(path)
        send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_text_response_does_not_call_tts():
    from handlers import text
    from utils.helpers import user_sessions
    message = SimpleNamespace(from_user=SimpleNamespace(id=42), chat=SimpleNamespace(id=42), text="Вопрос")
    with patch.object(text, "route_text_request", new=AsyncMock(return_value={"text": "Ответ"})), patch.object(text.bot, "send_chat_action", new=AsyncMock()), patch.object(text.bot, "send_message", new=AsyncMock()) as send, patch.object(openai_client, "generate_speech", new=AsyncMock()) as speech:
        await text.handle_text_message(message)
        send.assert_awaited_once_with(42, "Ответ")
        speech.assert_not_awaited()


def test_cleanup_none_is_silent():
    with patch("utils.helpers.logger.warning") as warning:
        cleanup_file(None)
        warning.assert_not_called()
