from loguru import logger
import os
import sys

# 日志存储目录
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 移除默认日志配置
logger.remove()

# 添加控制台输出（简洁格式）
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

# 添加文件输出（详细格式，按天分割）
logger.add(
    os.path.join(LOG_DIR, "llamaindex_splitter_{time:YYYY-MM-DD}.log"),
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="00:00",  # 每天凌晨分割
    retention="7 days",  # 保留7天日志
    encoding="utf-8"
)

# 导出logger实例
__all__ = ["logger"]