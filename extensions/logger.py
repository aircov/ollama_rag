# -*- coding: utf-8 -*-
# @Time    : 2025/3/25 10:46
# @Author  : yaomw
# @Desc    :
import os
import sys
import time

import logging
from types import FrameType
from typing import cast

from loguru import logger


basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 定位到log日志文件
log_path = os.path.join(basedir, 'logs')

if not os.path.exists(log_path):
    os.mkdir(log_path)


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:  # noqa: WPS609
            frame = cast(FrameType, frame.f_back)
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage(),
        )


LOGGING_LEVEL = logging.INFO
LOGGERS = ("uvicorn.asgi", "uvicorn.access")

logging.getLogger().handlers = [InterceptHandler()]
for logger_name in LOGGERS:
    logging_logger = logging.getLogger(logger_name)
    logging_logger.handlers = [InterceptHandler(level=LOGGING_LEVEL)]

log_file_path = os.path.join(basedir, 'logs/engine_{}.log'.format(time.strftime("%Y-%m-%d", time.localtime())))
err_log_file_path = os.path.join(basedir, 'logs/engine_{}.err.log'.format(time.strftime("%Y-%m-%d", time.localtime())))

# "rotation": "500 MB"/"00:00"
loguru_config = {
    "handlers": [
        {"sink": sys.stderr, "level": "INFO",
         "format": "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | {thread.name} | <level>{level}</level> | "
                   "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - {message}"},
        {"sink": log_file_path, "rotation": "00:00",
         "retention": "2 weeks", "encoding": 'utf-8'},
        {"sink": err_log_file_path, "serialize": False, "level": 'ERROR',
         "rotation": "00:00", "retention": "2 weeks",
         "encoding": 'utf-8'},
    ],
}

logger.configure(**loguru_config)
