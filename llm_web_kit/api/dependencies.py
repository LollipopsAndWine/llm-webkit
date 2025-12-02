"""API 依赖项管理.

包含 FastAPI 应用的依赖项、配置管理和共享服务。
"""

import logging
from contextvars import ContextVar
from functools import lru_cache
from typing import Optional
import os
from logging.handlers import TimedRotatingFileHandler

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# 创建一个 ContextVar 用于存储 request_id，提供默认值
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


class RequestIdFilter(logging.Filter):
    """
    日志过滤器，用于将 request_id 从 ContextVar 注入到日志记录中。
    """
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True


class Settings(BaseSettings):
    """应用配置设置."""

    # API 配置
    api_title: str = "LLM Web Kit API"
    api_version: str = "1.0.0"
    api_description: str = "基于 LLM 的 Web 内容解析和提取 API 服务"

    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # 日志配置
    log_level: str = "INFO"
    log_dir: str = "logs"
    log_filename: str = "api.log"

    # 模型配置
    model_path: Optional[str] = None
    max_content_length: int = 10 * 1024 * 1024  # 10MB
    crawl_url: str = "http://10.140.0.94:9500/crawl"

    # 缓存配置
    cache_ttl: int = 3600  # 1小时

    # 数据库配置
    database_url: Optional[str] = None  # 从环境变量 DATABASE_URL 读取
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # pydantic v2 配置写法
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False
    )


@lru_cache()
def get_settings() -> Settings:
    """获取应用配置单例."""
    return Settings()


def get_logger(name: str = __name__) -> logging.Logger:
    """获取配置好的日志记录器."""
    logger = logging.getLogger(name)
    logger.setLevel(get_settings().log_level)
    logger.addFilter(RequestIdFilter())  # 添加过滤器

    if not logger.handlers:
        # 控制台处理器
        stream_handler = logging.StreamHandler()
        stream_formatter = logging.Formatter(
            '%(asctime)s - %(request_id)s - %(name)s - %(levelname)s - %(message)s'
        )
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

        # 文件处理器 (按天轮换)
        settings = get_settings()
        log_dir = settings.log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file_path = os.path.join(log_dir, settings.log_filename)

        file_handler = TimedRotatingFileHandler(
            log_file_path,
            when="midnight",  # 每天午夜轮换
            interval=1,
            backupCount=30,  # 保留30天的日志
            encoding='utf-8'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(request_id)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


# 全局依赖项
settings = get_settings()

# InferenceService 单例
_inference_service_singleton = None


def get_inference_service():
    """获取 InferenceService 单例."""
    global _inference_service_singleton
    if _inference_service_singleton is None:
        from .services.inference_service import InferenceService
        _inference_service_singleton = InferenceService()
    return _inference_service_singleton
