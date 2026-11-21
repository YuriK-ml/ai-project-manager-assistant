"""
Start and Help Command Handlers.
Handles /start and /help commands using pyTelegramBotAPI.
"""

from telebot import types
from bot import bot
from utils.logging import logger
from utils.helpers import user_sessions


def main_keyboard(user_id: int):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True, is_persistent=True)
    state = "ВКЛ" if user_sessions.is_rag_enabled(user_id) else "ВЫКЛ"
    keyboard.row(f"📚 RAG: {state}", "📊 Статистика")
    keyboard.row("🔄 Обновить индекс", "🧹 Новый диалог")
    return keyboard


@bot.message_handler(commands=['start'])
async def cmd_start(message: types.Message):
    user_id = message.from_user.id
    await bot.send_message(
        message.chat.id,
        "AI Project Manager Assistant — помощник руководителя корпоративного ИТ-проекта.\n\n"
        "Принимаю текст, голосовые сообщения и изображения. Голос автоматически "
        "распознаётся и обрабатывается как текст. Ответ всегда текстовый.\n\n"
        "Кнопка RAG включает и выключает работу с загруженной базой документов. "
        "При RAG ВКЛ текстовые и голосовые вопросы используют эту базу. "
        "Изображения анализируются отдельно через Vision.\n\n/help — справка.",
        reply_markup=main_keyboard(user_id)
    )


@bot.message_handler(commands=['help'])
async def cmd_help(message: types.Message):
    await bot.send_message(
        message.chat.id,
        "Отправьте текст, голосовое сообщение или фото с вопросом. "
        "Переключать тип ввода не требуется. Ответ возвращается текстом.\n\n"
        "📚 RAG — включить/выключить базу нормативных документов для текста и голоса.\n"
        "📊 Статистика — число фрагментов в индексе.\n"
        "🔄 Обновить индекс — перестроить индекс текущих документов (использует API).\n"
        "🧹 Новый диалог — очистить вашу историю, сохранив состояние RAG.\n\n"
        "Документы размещаются в data/documents/. После изменения файлов обновите индекс.\n"
        "Команды: /stats, /index, /reset; /mode rag — RAG ВКЛ, /mode off — RAG ВЫКЛ.",
        reply_markup=main_keyboard(message.from_user.id)
    )


@bot.message_handler(func=lambda message: message.text in {"📚 RAG: ВКЛ", "📚 RAG: ВЫКЛ"})
async def toggle_rag(message: types.Message):
    user_id = message.from_user.id
    user_sessions.set_rag_enabled(user_id, not user_sessions.is_rag_enabled(user_id))
    await send_rag_status(message)


async def send_rag_status(message: types.Message):
    state = "ВКЛ" if user_sessions.is_rag_enabled(message.from_user.id) else "ВЫКЛ"
    await bot.send_message(message.chat.id, f"📚 RAG: {state}",
                           reply_markup=main_keyboard(message.from_user.id))


@bot.message_handler(func=lambda message: message.text == "📊 Статистика")
async def button_stats(message: types.Message):
    await cmd_stats(message)


@bot.message_handler(func=lambda message: message.text == "🔄 Обновить индекс")
async def button_index(message: types.Message):
    await cmd_index(message)


@bot.message_handler(func=lambda message: message.text == "🧹 Новый диалог")
async def button_reset(message: types.Message):
    await cmd_reset(message)


@bot.message_handler(commands=['reset'])
async def cmd_reset(message: types.Message):
    """Handle /reset command - clear conversation history."""
    user_id = message.from_user.id
    
    user_sessions.clear_history(user_id)
    logger.info(f"User {user_id} cleared conversation history")
    
    await bot.send_message(
        message.chat.id,
        "✅ История диалога очищена!\n\n"
        "Начнем с чистого листа. Чем могу помочь?",
        reply_markup=main_keyboard(user_id)
    )


@bot.message_handler(commands=['stats'])
async def cmd_stats(message: types.Message):
    """Handle /stats command - show knowledge base statistics."""
    user_id = message.from_user.id
    logger.info(f"User {user_id} requested stats")
    
    try:
        from rag.query import get_knowledge_base_stats
        
        stats = get_knowledge_base_stats()
        
        if "error" in stats:
            await bot.send_message(
                message.chat.id,
                f"⚠️ Ошибка получения статистики:\n{stats['error']}",
                parse_mode=""
            )
            return
        
        total_docs = stats.get("total_documents", 0)
        persist_dir = stats.get("persist_directory", "N/A")
        
        stats_text = f"""📊 **Статистика базы знаний**

📄 Фрагментов в индексе: {total_docs}
💾 Директория: {persist_dir}

{"✅ База знаний готова к использованию!" if total_docs > 0 else "⚠️ База знаний пуста. Добавьте документы в data/documents/"}

Включите RAG кнопкой для работы с базой знаний."""
        
        await bot.send_message(message.chat.id, stats_text, parse_mode="")
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        await bot.send_message(
            message.chat.id,
            "⚠️ Ошибка получения статистики базы знаний.",
            parse_mode=""
        )


@bot.message_handler(commands=['index'])
async def cmd_index(message: types.Message):
    """Rebuild the existing collection from current documents."""
    import asyncio
    from rag.index import vector_index

    await bot.send_message(message.chat.id, "Индексирую документы. Используется API эмбеддингов.")
    try:
        count = await asyncio.to_thread(
            vector_index.index_documents_directory, force_reindex=True
        )
        await bot.send_message(message.chat.id, f"Индекс обновлён. Фрагментов: {count}. Включите RAG кнопкой.")
    except Exception:
        logger.exception("Document indexing failed")
        await bot.send_message(message.chat.id, "Ошибка индексации. Проверьте настройки API и bot.log, затем повторите /index.")
