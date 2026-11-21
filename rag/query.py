"""
RAG Query Handler.
Handles queries against the knowledge base with context-aware responses.
"""

from typing import List, Dict, Optional

from rag.index import vector_index
from services.openai_client import openai_client
from utils.logging import logger
from config import RAG_TOP_K, SYSTEM_PROMPT


async def query_knowledge_base(
    query: str,
    conversation_history: Optional[List[Dict]] = None
) -> str:
    """
    Query the knowledge base and generate response.
    
    Args:
        query: User's query
        conversation_history: Previous conversation messages
    
    Returns:
        Generated response based on retrieved context
    """
    try:
        # Search for relevant documents
        logger.debug(f"Searching knowledge base for: {query}")
        results = vector_index.similarity_search_with_score(query, k=RAG_TOP_K)
        
        if not results:
            logger.warning("No relevant documents found, using fallback")
            return await _fallback_response(query, conversation_history)
        
        # Prepare context from retrieved documents
        context = _prepare_context(results)
        
        # Generate response with context
        response = await _generate_rag_response(
            query=query,
            context=context,
            conversation_history=conversation_history
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error querying knowledge base: {e}")
        # Report unavailable context without inventing an answer
        return await _fallback_response(query, conversation_history)


def _prepare_context(results: List[tuple]) -> str:
    """
    Prepare context from search results.
    
    Args:
        results: List of (document, score) tuples
    
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get('source', 'Unknown')
        content = doc.page_content.strip()
        
        context_parts.append(
            f"[Источник {i}: {source}]\n{content}\n"
        )
    
    return "\n".join(context_parts)


async def _generate_rag_response(
    query: str,
    context: str,
    conversation_history: Optional[List[Dict]] = None
) -> str:
    """
    Generate response using RAG context.
    
    Args:
        query: User's query
        context: Retrieved context from knowledge base
        conversation_history: Previous conversation
    
    Returns:
        Generated response
    """
    # Build prompt with context
    system_prompt = SYSTEM_PROMPT + """
В режиме RAG отвечай прежде всего по найденным нормативным документам.
Не дополняй отсутствующие положения регламентов общими знаниями или историей диалога.
Если фрагментов недостаточно, явно сообщи об этом. Указывай названия документов
для подтверждённых утверждений; не выдумывай номера пунктов или источники.

ВАЖНЫЕ ПРАВИЛА:
1. Отвечай на основе предоставленного контекста
2. Если контекст содержит ответ - используй его
3. Если контекст не содержит ответа - честно скажи об этом
4. Всегда указывай источники информации
5. Отвечай на русском языке четко и структурированно

КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
{context}

Используй этот контекст для ответа на вопрос пользователя."""
    
    # Prepare messages
    messages = [
        {
            "role": "system",
            "content": system_prompt.replace("{context}", context)
        }
    ]
    
    # Add conversation history if available
    if conversation_history:
        # Limit history to avoid token limits
        recent_history = conversation_history[-6:]  # Last 3 exchanges
        messages.extend(recent_history)
    
    # Add current query
    messages.append({
        "role": "user",
        "content": query
    })
    
    # Generate response
    response = await openai_client.generate_text_response(messages)
    
    return response


async def _fallback_response(
    query: str,
    conversation_history: Optional[List[Dict]] = None
) -> str:
    """
    Report insufficient context when RAG fails.
    
    Args:
        query: User's query
        conversation_history: Previous conversation
    
    Returns:
        Generated response
    """
    logger.info("RAG context unavailable")
    return ("⚠️ Недостаточно данных из нормативных документов для ответа. "
            "База знаний пуста, поиск недоступен или подходящие фрагменты не найдены. "
            "Проверьте /stats и выполните /index, если документы ещё не индексировались.")



async def add_document_to_knowledge_base(file_path: str) -> dict:
    """
    Add a document to the knowledge base.
    
    Args:
        file_path: Path to document file
    
    Returns:
        Dictionary with status and details
    """
    try:
        from pathlib import Path
        from rag.loader import document_loader
        
        # Load document
        file_path = Path(file_path)
        documents = document_loader.load_document(file_path)
        
        # Add to index
        vector_index.add_documents(documents)
        
        logger.info(f"Added {file_path.name} to knowledge base")
        
        return {
            "success": True,
            "file": file_path.name,
            "chunks": len(documents),
            "message": f"Документ {file_path.name} успешно добавлен ({len(documents)} фрагментов)"
        }
        
    except Exception as e:
        logger.error(f"Error adding document to knowledge base: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Ошибка при добавлении документа: {e}"
        }


def get_knowledge_base_stats() -> dict:
    """
    Get statistics about the knowledge base.
    
    Returns:
        Dictionary with statistics
    """
    return vector_index.get_stats()

