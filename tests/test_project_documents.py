"""Document loading and RAG safety checks without external API calls."""
from unittest.mock import AsyncMock, Mock, patch

import pytest
from docx import Document
from config import DOCUMENTS_DIR
from rag.loader import document_loader


def test_docx_paragraph_order_and_source(tmp_path):
    path = tmp_path / "Регламент.docx"
    doc = Document()
    doc.add_paragraph("Первый абзац")
    doc.add_paragraph("")
    doc.add_paragraph("Второй абзац")
    doc.save(path)
    chunks = document_loader.load_document(path)
    assert len(chunks) == 1
    assert chunks[0].page_content == "Первый абзац\n\nВторой абзац"
    assert chunks[0].metadata["source"] == path.name


def test_three_real_documents_are_loaded():
    expected = {p.name for p in DOCUMENTS_DIR.glob("*.docx")}
    assert len(expected) == 3
    chunks = document_loader.load_directory()
    assert {c.metadata["source"] for c in chunks} == expected
    assert all(c.page_content.strip() for c in chunks)


@pytest.mark.asyncio
async def test_empty_or_failed_retrieval_does_not_invent_answer():
    from rag.query import query_knowledge_base
    with patch("rag.query.vector_index") as index, patch("rag.query.openai_client") as client:
        client.generate_text_response = AsyncMock()
        index.similarity_search_with_score.return_value = []
        assert "Недостаточно данных" in await query_knowledge_base("Обязанности РП?")
        index.similarity_search_with_score.side_effect = RuntimeError("offline")
        assert "Недостаточно данных" in await query_knowledge_base("Обязанности РП?")
        client.generate_text_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_rag_prompt_contains_document_name():
    from rag.query import query_knowledge_base
    doc = Mock(page_content="РП ведёт отчётность.", metadata={"source": "Инструкция РП.docx"})
    with patch("rag.query.vector_index") as index, patch("rag.query.openai_client") as client:
        index.similarity_search_with_score.return_value = [(doc, 0.2)]
        client.generate_text_response = AsyncMock(return_value="Ответ")
        await query_knowledge_base("Что делает РП?")
        prompt = client.generate_text_response.call_args.args[0][0]["content"]
        assert "AI Project Manager Assistant" in prompt
        assert "Инструкция РП.docx" in prompt
        assert "РП ведёт отчётность." in prompt


@pytest.mark.asyncio
async def test_startup_registers_handlers_without_indexing():
    import main
    from types import SimpleNamespace
    with patch.object(main.bot, "get_me", new=AsyncMock(return_value=SimpleNamespace(username="offline"))), patch("rag.index.vector_index") as index:
        await main.setup_bot()
        index.index_documents_directory.assert_not_called()
    commands = {command for handler in main.bot.message_handlers
                for command in handler["filters"].get("commands", [])}
    assert {"start", "index", "stats", "mode"} <= commands
