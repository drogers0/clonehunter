class CloneHunterError(Exception):
    """Base exception for CloneHunter."""


class ConfigError(CloneHunterError):
    """Raised when configuration is invalid."""
