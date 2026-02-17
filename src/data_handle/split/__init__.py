# -*- coding: utf-8 -*-
"""
文本分割策略模块

企业级文本分割策略框架，提供多种文本分割算法的统一接口。

模块结构：
- base_strategy: 抽象基类定义
- batch_strategy: 批量分割策略（单进程）
- parallel_strategy: 并行分割策略（多进程）
- factory: 策略工厂类

使用示例：
    >>> from src.data_handle.split import TextSplitterStrategyFactory
    >>> from langchain_text_splitters import RecursiveCharacterTextSplitter
    >>> 
    >>> # 创建分割器
    >>> splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    >>> 
    >>> # 创建策略工厂
    >>> factory = TextSplitterStrategyFactory()
    >>> 
    >>> # 创建策略实例
    >>> strategy = factory.create_strategy('batch', splitter, min_length=50)
    >>> 
    >>> # 使用策略分割文本
    >>> texts, metadata = strategy.split_texts(['长文本内容'])

导出的主要类：
- BaseTextSplitterStrategy: 策略基类
- BatchTextSplitterStrategy: 批量分割策略
- ParallelBatchTextSplitterStrategy: 并行分割策略
- TextSplitterStrategyFactory: 策略工厂
"""

from .base_strategy import BaseTextSplitterStrategy
from .batch_strategy import BatchTextSplitterStrategy
from .parallel_strategy import ParallelBatchTextSplitterStrategy
from .factory import TextSplitterStrategyFactory, create_text_splitter_strategy, get_available_strategies

# 便捷导入
__all__ = [
    'BaseTextSplitterStrategy',
    'BatchTextSplitterStrategy', 
    'ParallelBatchTextSplitterStrategy',
    'TextSplitterStrategyFactory',
    'create_text_splitter_strategy',
    'get_available_strategies'
]

# 版本信息
__version__ = '1.0.0'
__author__ = 'RAG Team'