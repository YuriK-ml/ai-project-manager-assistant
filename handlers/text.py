"""
Text Message Handler.
Handles regular text messages from users using pyTelegramBotAPI.
"""

from telebot import types
from bot import bot
from services.router import route_text_request
from utils.logging import logger
from utils.helpers import user_sessions


@bot.message_handler(commands=['mode'])
async def cmd_mode(message: types.Message):
    """Technical compatibility command; the primary UI is the RAG button."""
    from handlers.start import send_rag_status
    args = message.text.split(maxsplit=1)
    if len(args) > 1:
        value = args[1].lower()
        if value not in {"rag", "on", "off", "text", "voice", "vision"}:
            await bot.send_message(message.chat.id, "Используйте /mode rag или /mode off.")
            return
        user_sessions.set_rag_enabled(message.from_user.id, value in {"rag", "on"})
    await send_rag_status(message)


@bot.message_handler(func=lambda message: message.content_type == 'text' and not message.text.startswith('/'))
async def handle_text_message(message: types.Message):
    """Handle regular text messages."""
    user_id = message.from_user.id
    text = message.text
    
    logger.info(f"Text message from user {user_id}: {text[:50]}...")
    
    # Show typing indicator
    await bot.send_chat_action(message.chat.id, 'typing')
    
    try:
        # Route request
        response = await route_text_request(user_id, text)
        
        # Both RAG states return text; voice is an input type.
        await bot.send_message(message.chat.id, response["text"])

    except Exception as e:
        logger.error(f"Error handling text message: {e}", exc_info=True)
        await bot.send_message(
            message.chat.id,
            "❌ Произошла ошибка при обработке сообщения.\n"
            "Попробуйте еще раз или используйте /reset для сброса."
        )
