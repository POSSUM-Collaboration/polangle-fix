"""Package logger"""

from __future__ import annotations

import logging


def get_logger(name: str = "spice-racs-fix") -> logging.Logger:
    """Logger with one stream handler, added once.

    Args:
        name: Logger name.

    Returns:
        The logger.
    """
    package_logger = logging.getLogger(name)
    if not package_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(levelname)s %(module)s.%(funcName)s: %(message)s")
        )
        package_logger.addHandler(handler)
        package_logger.setLevel(logging.INFO)
    return package_logger


logger = get_logger()
