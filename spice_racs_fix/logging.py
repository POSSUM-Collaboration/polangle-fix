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
    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(levelname)s %(module)s.%(funcName)s: %(message)s")
        )
        log.addHandler(handler)
        log.setLevel(logging.INFO)
    return log


logger = get_logger()
