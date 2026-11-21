"""
Voice Message Handler.
Handles voice messages with STT and text responses using pyTelegramBotAPI.
"""

from telebot import types
from bot import bot
from services.router import route_voice_request
from utils.logging import logger
from utils.helpers import save_file_async, cleanup_files


@bot.message_handler(content_types=['voice'])
async def handle_voice_message(message: types.Message):
    """Handle voice messages."""
    user_id = message.from_user.id
    
    logger.info(f"Voice message from user {user_id}")
    
    # Show typing indicator
    await bot.send_chat_action(message.chat.id, 'typing')
    
    voice_file_path = None
    
    try:
        # Download voice message
        file_info = await bot.get_file(message.voice.file_id)
        voice_bytes = await bot.download_file(file_info.file_path)
        
        # Save to temporary file
        voice_file_path = await save_file_async(voice_bytes, "ogg")
        
        logger.debug(f"Voice file saved: {voice_file_path}")
        
        # Process voice request
        response = await route_voice_request(user_id, voice_file_path)
        
        # STT may fail before a transcription is available.
        transcription = response.get("transcription")
        if transcription:
            await bot.send_message(
                message.chat.id,
                f"🎤 Распознано:\n{transcription}",
                parse_mode=None
            )

        await bot.send_message(message.chat.id, response["text"])

    except Exception as e:
        logger.error(f"Error handling voice message: {e}", exc_info=True)
        await bot.send_message(
            message.chat.id,
            "❌ Произошла ошибка при обработке голосового сообщения.\n"
            "Попробуйте еще раз."
        )
    
    finally:
        # Cleanup temporary files
        cleanup_files(*(path for path in (voice_file_path,) if path is not None))


@bot.message_handler(content_types=['audio'])
async def handle_audio_message(message: types.Message):
    """Handle audio files (similar to voice)."""
    await bot.send_message(
        message.chat.id,
        "ℹ️ Для обработки аудио используйте голосовые сообщения.\n"
        "Аудиофайлы в виде документов не поддерживаются."
    )
