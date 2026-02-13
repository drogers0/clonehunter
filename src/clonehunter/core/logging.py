from __future__ import annotations

import logging

LOGGER_NAME = "clonehunter"


def get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
