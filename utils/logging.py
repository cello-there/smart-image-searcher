import logging
import sys

_DEF_FMT = "%(asctime)s %(levelname)s %(name)s | %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter(_DEF_FMT))
    logger.addHandler(h)
    return logger