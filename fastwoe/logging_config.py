"""
Universal logging configuration for fastwoe.

This module provides a standardized logger setup with RichHandler for consistent
logging across the entire codebase.
"""

_HAS_LOGGING = False

try:
    from loguru import logger
    from rich.logging import RichHandler

    _HAS_LOGGING = True
except ImportError:
    # Create a no-op logger when loguru is not available
    class _NullLogger:
        """No-op logger when loguru is not available."""

        def _noop(self, *args, **kwargs):
            pass

        info = debug = warning = error = critical = _noop

    logger = _NullLogger()


def setup_logger(level: str = "INFO") -> None:
    """
    Configure logger with RichHandler for better formatting.

    This function sets up the loguru logger with RichHandler, which provides
    beautiful, formatted output with colors and rich tracebacks. This should
    be called once at the start of your script or test file.

    Parameters
    ----------
    level : str, default="INFO"
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Examples:
    --------
    >>> from fastwoe.logging_config import setup_logger, logger
    >>> setup_logger(level="INFO")
    >>> logger.info("This will be beautifully formatted!")
    """
    if not _HAS_LOGGING:
        return

    # Remove default handler
    logger.remove()

    # Add RichHandler with better formatting
    logger.add(
        RichHandler(markup=True, rich_tracebacks=True),
        format="{message}",
        level=level,
    )


__all__ = ["logger", "setup_logger", "_HAS_LOGGING"]
