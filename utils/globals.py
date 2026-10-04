from __future__ import annotations

"""全局日志对象。

训练主流程、数据模块和诊断脚本都复用同一个 logger，
这样所有关键输出都会同时写到终端和 ``scaleimts/logs/train.log``。
"""

import logging
import logging.handlers as handlers
from pathlib import Path


def _build_logger() -> logging.Logger:
    """构造全局 logger，并配置文件输出与终端输出。"""
    logger = logging.getLogger("scaleimts")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_dir = Path("scaleimts") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = handlers.RotatingFileHandler(
        log_dir / "train.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=2,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


logger = _build_logger()
