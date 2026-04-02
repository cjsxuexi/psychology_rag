from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from functools import lru_cache
from typing import Literal, Optional
from enum import Enum
import os


class SplitterType(str, Enum):
    """文本分割器类型枚举"""
    RECURSIVE = "recursive"
    CHARACTER = "character"
    TOKEN = "token"
    LLAMA_SENTENCE = "llama_sentence"
    LLAMA_SENTENCE_WINDOW = "llama_sentence_window"
    LLAMA_SEMANTIC = "llama_semantic"
    LLAMA_COMBINED = "llama_combined"


class LogConfig(BaseSettings):
    """日志配置"""
    model_config = SettingsConfigDict(env_prefix="LOG_")
    
    level: str = Field(default="INFO", description="日志级别")
    format: Literal["simple", "json"] = Field(default="simple", description="日志格式")
    max_bytes: int = Field(default=10*1024*1024, description="单个日志文件最大大小")
    backup_count: int = Field(default=5, description="保留日志文件数量")
    retention_days: int = Field(default=7, description="日志保留天数")
    
    @field_validator("level")
    @classmethod
    def validate_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"无效的日志级别: {v}，可选: {valid_levels}")
        return v.upper()


class DatabaseConfig(BaseSettings):
    """数据库配置"""
    model_config = SettingsConfigDict(env_prefix="DB_")
    
    host: str = Field(default="localhost")
    port: int = Field(default=3306)
    database: str = Field(default="rag_db")
    user: str = Field(default="root")
    password: str = Field(default="", repr=False)
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)
    
    @property
    def connection_string(self) -> str:
        return f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class MilvusConfig(BaseSettings):
    """Milvus向量数据库配置"""
    model_config = SettingsConfigDict(env_prefix="MILVUS_")
    
    host: str = Field(default="localhost")
    port: int = Field(default=19530)
    collection_name: str = Field(default="rag_collection")
    dimension: int = Field(default=768)
    

class SplitterConfig(BaseSettings):
    """文本分割配置"""
    model_config = SettingsConfigDict(env_prefix="SPLITTER_")
    
    type: SplitterType = Field(default=SplitterType.RECURSIVE)
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    
    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v, info):
        chunk_size = info.data.get("chunk_size")
        if chunk_size and v >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        return v


class EmbeddingConfig(BaseSettings):
    """Embedding模型配置"""
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")
    
    model_name: str = Field(default="BAAI/bge-m3")
    device: Literal["cpu", "cuda", "auto"] = Field(default="auto")
    batch_size: int = Field(default=32)


class AppConfig(BaseSettings):
    """应用主配置"""
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    env: Literal["development", "testing", "production"] = Field(
        default="development",
        alias="APP_ENV"
    )
    debug: bool = Field(default=False, alias="APP_DEBUG")
    
    log: LogConfig = Field(default_factory=LogConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    splitter: SplitterConfig = Field(default_factory=SplitterConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    
    @property
    def is_development(self) -> bool:
        return self.env == "development"
    
    @property
    def is_production(self) -> bool:
        return self.env == "production"


@lru_cache()
def get_settings() -> AppConfig:
    """
    获取配置实例（单例模式，缓存结果）
    
    使用 lru_cache 确保配置只加载一次，且线程安全
    """
    return AppConfig()
