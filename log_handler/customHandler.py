import logging
import os

import telebot

class TelegramBotHandler(logging.Handler):
    """Handler to send telegram messages using bot"""
    def __init__(self, api_key: str, chat_id: str):
        super().__init__()
        self.api_key = api_key
        self.chat_id = chat_id

    def emit(self, record: logging.LogRecord):
        """Sends message to chat specified in .env file"""
        bot = telebot.TeleBot(self.api_key)

        bot.send_message(
            self.chat_id,
            self.format(record)
        )

class RecognitionScoreFilter(logging.Filter):
    """Filters messages based on recognition score."""

    def filter(self, record):
        return hasattr(record, 'recognition_score') and record.recognition_score > 0.35
