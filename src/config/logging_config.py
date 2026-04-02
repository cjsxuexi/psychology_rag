from loguru import logger
import os
import sys
import json
from functools import lru_cache
from contextvars import ContextVar
from typing import Optional


request_id_var: ContextVar[str] = ContextVar("request_id", default="")


@lru_cache()
def get_log_dir() -> str:
    """获取日志目录（缓存避免重复计算）"""
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "logs"
    )
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_request_id() -> str:
    """获取当前请求ID"""
    return request_id_var.get()


def set_request_id(request_id: str):
    """设置当前请求ID"""
    request_id_var.set(request_id)


def _json_formatter(record) -> str:
    """JSON格式日志格式化器"""
    log_data = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
        "request_id": get_request_id(),
        "extra": dict(record.get("extra", {}))
    }
    return json.dumps(log_data, ensure_ascii=False, default=str)


def configure_logger(
    level: str = "INFO",
    log_format: str = "simple",
    max_bytes: int = 10*1024*1024,
    backup_count: int = 5,
    retention_days: int = 7
):
    """
    配置日志系统
    
    Args:
        level: 日志级别
        log_format: 日志格式 (simple/json)
        max_bytes: 单个文件最大大小
        backup_count: 保留文件数量
        retention_days: 保留天数
    """
    logger.remove()
    
    if log_format == "json":
        console_format = _json_formatter
        file_format = _json_formatter
    else:
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<yellow>{extra[request_id]}</yellow> - "
            "<level>{message}</level>"
        )
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} | {extra[request_id]} - {message}"
        )
    
    logger.add(
        sys.stdout,
        level=level,
        format=console_format,
        enqueue=True
    )
    
    log_dir = get_log_dir()
    
    logger.add(
        os.path.join(log_dir, "app_{time:YYYY-MM-DD}.log"),
        level="DEBUG",
        format=file_format,
        rotation=max_bytes,
        retention=f"{retention_days} days",
        compression="zip",
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=True
    )
    
    logger.add(
        os.path.join(log_dir, "error_{time:YYYY-MM-DD}.log"),
        level="ERROR",
        format=file_format,
        rotation=max_bytes,
        retention=f"{retention_days} days",
        compression="zip",
        encoding="utf-8",
        enqueue=True
    )
    
    return logger.bind(request_id="")


def get_logger(request_id: Optional[str] = None):
    """获取带上下文的logger"""
    if request_id:
        return logger.bind(request_id=request_id)
    return logger.bind(request_id=get_request_id())


def init_logger(config=None):
    """初始化日志系统"""
    if config is None:
        from src.config.settings import get_settings
        config = get_settings().log
    
    configure_logger(
        level=config.level,
        log_format=config.format,
        max_bytes=config.max_bytes,
        backup_count=config.backup_count,
        retention_days=config.retention_days
    )
    logger.info(f"日志系统初始化完成 | 级别: {config.level} | 格式: {config.format}")


__all__ = ["logger", "get_logger", "init_logger", "set_request_id", "get_request_id"]
