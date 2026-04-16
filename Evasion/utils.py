import sys
import os
import logging

# Убедимся, что папка для логов существует
os.makedirs("logs", exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    handlers=[
                        logging.StreamHandler(sys.stdout),
                        logging.FileHandler("logs/logs_or_text_variant.txt", mode="w", encoding="utf-8")
                    ])
logger = logging.getLogger()

def log_print(*args):
    """Функция для логирования сообщений."""
    logger.info(" ".join(str(arg) for arg in args))
