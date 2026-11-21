"""
Image Handler.
Handles image analysis with GPT-4 Vision using pyTelegramBotAPI.
"""

from pathlib import Path
from telebot import types
from bot import bot
from services.router import route_image_request
from utils.logging import logger
from utils.helpers import cleanup_file


@bot.message_handler(content_types=['photo'])
async def handle_photo_message(message: types.Message):
    """Handle photo messages."""
    user_id = message.from_user.id
    
    logger.info(f"Photo message from user {user_id}")
    
    # Show typing indicator
    await bot.send_chat_action(message.chat.id, 'typing')
    
    try:
        # Get the largest photo
        photo = message.photo[-1]
        
        # Get caption if provided
        caption = message.caption
        
        # Get file URL (for Vision API)
        file_info = await bot.get_file(photo.file_id)
        file_url = f"https://api.telegram.org/file/bot{bot.token}/{file_info.file_path}"
        
        logger.debug(f"Image URL: {file_url}")
        
        # Process image request
        response = await route_image_request(
            user_id=user_id,
            image_url=file_url,
            caption=caption
        )
        
        # Send analysis result
        await bot.send_message(
            message.chat.id,
            response["text"],
            parse_mode=""
        )
    
    except Exception as e:
        logger.error(f"Error handling photo message: {e}")
        await bot.send_message(
            message.chat.id,
            "❌ Произошла ошибка при анализе изображения.\n"
            "Попробуйте отправить другое изображение."
        )


@bot.message_handler(content_types=['document'])
async def handle_document_message(message: types.Message):
    """Handle document messages (could be PDFs for RAG)."""
    user_id = message.from_user.id
    document = message.document
    
    # Check if it's a supported document type
    if Path(document.file_name or "").suffix.lower() in {".pdf", ".txt", ".md", ".docx"}:
        await bot.send_message(
            message.chat.id,
            "📄 Документ получен!\n\n"
            "Для добавления документов в базу знаний:\n"
            "1. Скачайте документ\n"
            "2. Поместите его в папку `data/documents/`\n"
            "3. Выполните /index (использует API эмбеддингов)\n"
            "4. Переключитесь в режим RAG: /mode rag\n\n"
            "⚠️ Автоматическая загрузка документов будет добавлена в следующей версии."
        )
    elif document.mime_type and document.mime_type.startswith("image/"):
        # Handle as image
        await bot.send_message(
            message.chat.id,
            "📸 Получено изображение в виде документа.\n"
            "Отправьте изображение как фото для анализа."
        )
    else:
        await bot.send_message(
            message.chat.id,
            f"ℹ️ Получен файл: {document.file_name}\n"
            f"Тип: {document.mime_type}\n\n"
            "Поддерживаемые типы для анализа:\n"
            "• Изображения (отправляйте как фото)\n"
            "• DOCX/PDF/TXT/MD документы (для базы знаний)"
        )
