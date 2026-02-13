import logging

from clonehunter.core.errors import CloneHunterError, ConfigError
from clonehunter.core.logging import get_logger


def test_errors_are_exceptions():
    assert issubclass(CloneHunterError, Exception)
    assert issubclass(ConfigError, CloneHunterError)


def test_get_logger_singleton():
    logger1 = get_logger()
    logger2 = get_logger()
    assert logger1 is logger2
    assert logger1.name == "clonehunter"
    assert any(isinstance(h, logging.Handler) for h in logger1.handlers)
