"""Input types share a single RAG on/off processing choice."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from services import router
from utils.helpers import UserSession
from handlers import start


def message(text=""):
    return SimpleNamespace(from_user=SimpleNamespace(id=77), chat=SimpleNamespace(id=77), text=text)


@pytest.mark.asyncio
@pytest.mark.parametrize("rag_enabled, voice", [(False, False), (True, False), (True, True)])
async def test_input_routing(rag_enabled, voice):
    session = UserSession()
    session.set_rag_enabled(77, rag_enabled)
    with patch.object(router, "user_sessions", session), patch.object(router.openai_client, "generate_text_response", new=AsyncMock(return_value="LLM")) as llm, patch("rag.query.query_knowledge_base", new=AsyncMock(return_value="RAG")) as rag, patch.object(router, "transcribe_voice_message", new=AsyncMock(return_value="Вопрос")) as stt:
        result = (await router.route_voice_request(77, Path("voice.ogg")) if voice
                  else await router.route_text_request(77, "Вопрос"))
        assert result["text"] == ("RAG" if rag_enabled else "LLM")
        assert rag.await_count == int(rag_enabled)
        assert llm.await_count == int(not rag_enabled)
        assert stt.await_count == int(voice)
        assert session.get_history(77)[0] == {"role": "user", "content": "Вопрос"}


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled, caption", [(False, "Какие риски видны на графике?"), (True, "Что должен сделать руководитель проекта?"), (False, None)])
async def test_image_vision_then_shared_pipeline(enabled, caption):
    session = UserSession()
    session.set_rag_enabled(77, enabled)
    facts = "График: план 80%, факт 60%. Подписи осей читаются."
    with patch.object(router, "user_sessions", session), patch.object(router, "analyze_image", new=AsyncMock(return_value=facts)) as vision, patch.object(router.openai_client, "generate_text_response", new=AsyncMock(return_value="Итог LLM")) as llm, patch("rag.query.query_knowledge_base", new=AsyncMock(return_value="Итог RAG")) as rag:
        result = await router.route_image_request(77, image_url="https://example.test/image", caption=caption)
        assert result["text"] == ("Итог RAG" if enabled else "Итог LLM")
        vision.assert_awaited_once()
        query = rag.call_args.args[0] if enabled else llm.call_args.args[0][-1]["content"]
        assert facts in query
        assert "ФАКТЫ ИЗ ПРИЛОЖЕННОГО ИЗОБРАЖЕНИЯ" in query
        assert (caption or "Проанализируй приложенное изображение") in query
        assert rag.await_count == int(enabled)
        assert llm.await_count == int(not enabled)
        history = session.get_history(77)
        assert len(history) == 2
        assert history[0]["content"].startswith("[Пользователь отправил изображение]")
        assert facts not in history[0]["content"]
        assert history[1]["content"] == result["text"]


@pytest.mark.asyncio
async def test_toggle_uses_current_state_and_updates_keyboard():
    session = UserSession()
    session.set_rag_enabled(77, False)
    with patch.object(start, "user_sessions", session), patch.object(start.bot, "send_message", new=AsyncMock()) as send:
        for expected in [True, False]:
            await start.toggle_rag(message("📚 RAG: ВЫКЛ"))
            assert session.is_rag_enabled(77) is expected
            label = "📚 RAG: " + ("ВКЛ" if expected else "ВЫКЛ")
            assert send.call_args.args[1] == label
            assert send.call_args.kwargs["reply_markup"].keyboard[0][0]["text"] == label


@pytest.mark.asyncio
@pytest.mark.parametrize("button, command", [("button_stats", "cmd_stats"), ("button_index", "cmd_index"), ("button_reset", "cmd_reset")])
async def test_buttons_reuse_commands(button, command):
    msg = message()
    with patch.object(start, command, new=AsyncMock()) as handler:
        await getattr(start, button)(msg)
        handler.assert_awaited_once_with(msg)


@pytest.mark.asyncio
async def test_reset_preserves_rag():
    session = UserSession()
    session.set_rag_enabled(77, True)
    session.add_message(77, "user", "Вопрос")
    with patch.object(start, "user_sessions", session), patch.object(start.bot, "send_message", new=AsyncMock()):
        await start.cmd_reset(message())
    assert session.get_history(77) == []
    assert session.is_rag_enabled(77)


@pytest.mark.asyncio
async def test_stats_and_index_existing_implementations():
    with patch.object(start.bot, "send_message", new=AsyncMock()) as send, patch("rag.query.get_knowledge_base_stats", return_value={"total_documents": 12}):
        await start.cmd_stats(message())
        assert "12" in send.call_args.args[1]
    with patch.object(start.bot, "send_message", new=AsyncMock()), patch("rag.index.vector_index") as index:
        index.index_documents_directory.return_value = 12
        await start.cmd_index(message())
        index.index_documents_directory.assert_called_once_with(force_reindex=True)


@pytest.mark.asyncio
async def test_image_rag_retrieves_and_passes_sources_with_visual_facts():
    from langchain_core.documents import Document
    session = UserSession()
    session.set_rag_enabled(77, True)
    source = Document(page_content="РП контролирует риски проекта.", metadata={"source": "Кодекс.docx"})
    with patch.object(router, "user_sessions", session), patch.object(router, "analyze_image", new=AsyncMock(return_value="График: план 80%, факт 60%.")), patch("rag.query.vector_index") as index, patch.object(router.openai_client, "generate_text_response", new=AsyncMock(return_value="Итог с источником Кодекс.docx")) as llm:
        index.similarity_search_with_score.return_value = [(source, 0.1)]
        result = await router.route_image_request(77, image_url="https://example.test/chart", caption="Оцени риски проекта")
        retrieval_query = index.similarity_search_with_score.call_args.args[0]
        assert "Оцени риски проекта" in retrieval_query
        assert "план 80%, факт 60%" in retrieval_query
        messages = llm.call_args.args[0]
        assert "Кодекс.docx" in messages[0]["content"]
        assert "РП контролирует риски проекта." in messages[0]["content"]
        assert "план 80%, факт 60%" in messages[-1]["content"]
        assert result["text"] == "Итог с источником Кодекс.docx"


@pytest.mark.asyncio
async def test_photo_handler_sends_only_final_answer():
    from handlers import image
    msg = SimpleNamespace(from_user=SimpleNamespace(id=77), chat=SimpleNamespace(id=77),
                          photo=[SimpleNamespace(file_id="photo")], caption="Оцени график")
    with patch.object(image.bot, "send_chat_action", new=AsyncMock()), patch.object(image.bot, "get_file", new=AsyncMock(return_value=SimpleNamespace(file_path="photo.jpg"))), patch.object(image, "route_image_request", new=AsyncMock(return_value={"text": "Итоговый ответ"})) as route, patch.object(image.bot, "send_message", new=AsyncMock()) as send:
        await image.handle_photo_message(msg)
        assert route.call_args.kwargs["caption"] == "Оцени график"
        send.assert_awaited_once_with(77, "Итоговый ответ", parse_mode="")
