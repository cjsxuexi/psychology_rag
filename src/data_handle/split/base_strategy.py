# -*- coding: utf-8 -*-
"""
文本分割策略基类模块

企业级文本分割策略框架，提供统一的接口和基础功能。
采用策略模式设计，支持多种分割算法的灵活切换。

设计原则：
1. 单一职责：每个策略类只负责一种分割算法
2. 开闭原则：易于扩展新的分割策略
3. 依赖倒置：高层模块不依赖具体策略实现
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)


class BaseTextSplitterStrategy(ABC):
    """
    文本拆分策略抽象基类
    
    定义统一的文本分割接口，所有具体的分割策略必须继承此类。
    提供基础的文本分割功能和配置管理。
    
    Attributes:
        text_splitter (Any): 文本分割器实例（必须具有split_text方法）
        min_length (int): 最小文本长度阈值
        strategy_name (str): 策略名称，用于日志和调试
        
    Example:
        >>> from langchain_text_splitters import RecursiveCharacterTextSplitter
        >>> splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        >>> strategy = ConcreteStrategy(splitter, min_length=50)
        >>> texts, metadata = strategy.split_texts(["长文本内容"], [{"source": "test"}])
    """

    def __init__(self, text_splitter: Any, min_length: int, is_validate: bool = False):
        """
        初始化文本分割策略
        
        Args:
            text_splitter: 文本分割器实例（必须具有split_text方法）
            min_length: 最小文本长度阈值
            
        Raises:
            TypeError: 当text_splitter不具有split_text方法时
            ValueError: 当min_length小于0时
        """
        # 检查text_splitter是否具有split_text方法（支持自定义分割器类）
        if not hasattr(text_splitter, 'split_text') or not callable(getattr(text_splitter, 'split_text')):
            raise TypeError("text_splitter必须具有split_text方法")
        
        if min_length < 0:
            raise ValueError("min_length不能为负数")
            
        self.text_splitter = text_splitter
        self.min_length = min_length
        self.strategy_name = self.__class__.__name__
        self.is_validate = is_validate
        if not self.is_validate :
            logger.info(f"不启用文本验证")
        logger.info(f"初始化文本分割策略: {self.strategy_name}")
        logger.debug(f"配置参数 - 最小长度: {min_length}")

    def split_texts(self, 
                   texts: List[str], 
                   metadata_list: Optional[List[dict]] = None, 
                   batch_size: int = 10_000) -> Tuple[List[str], List[dict]]:
        """
        模板方法：统一的文本分割入口
        
        Args:
            texts: 待分割的文本列表
            metadata_list: 对应的元信息列表，可选
            batch_size: 批处理大小，默认10000
            
        Returns:
            Tuple[List[str], List[dict]]: (分割后的文本列表, 对应的元信息列表)
        """
        # 1. 输入验证
        if self.is_validate:
            self.validate_inputs(texts, metadata_list)
        
        # 2. 过滤过短文本
        filtered_texts, filtered_metadata = self.filter_short_texts(texts, metadata_list)
        
        if not filtered_texts:
            logger.warning("过滤后无有效文本可供处理")
            return [], [] if metadata_list else []
        
        # 3. 调用具体实现
        return self._split_texts(filtered_texts, filtered_metadata, batch_size)
    
    @abstractmethod
    def _split_texts(self, 
                    texts: List[str], 
                    metadata_list: Optional[List[dict]] = None, 
                    batch_size: int = 10_000) -> Tuple[List[str], List[dict]]:
        """
        抽象方法：具体的文本分割实现（需子类实现）
        
        Args:
            texts: 已过滤的文本列表
            metadata_list: 对应的元信息列表，可选
            batch_size: 批处理大小，默认10000
            
        Returns:
            Tuple[List[str], List[dict]]: (分割后的文本列表, 对应的元信息列表)
            
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("子类必须实现_split_texts方法")

    def split_single_text(self, text: str) -> List[str]:
        """
        分割单条文本（通用实现）
        
        Args:
            text: 待分割的单条文本
            
        Returns:
            List[str]: 分割后的文本块列表
            
        Raises:
            TypeError: 当text不是字符串时
            ValueError: 当text为空时
        """
        if not isinstance(text, str):
            raise TypeError("text必须是字符串类型")
        
        if not text.strip():
            logger.warning("检测到空文本，返回空列表")
            return []
            
        try:
            chunks = self.text_splitter.split_text(text)
            logger.debug(f"单条文本分割完成: 输入长度{len(text)} → 生成{len(chunks)}个块")
            return chunks
        except Exception as e:
            logger.error(f"文本分割失败: {str(e)}")
            raise

    def validate_inputs(self, 
                       texts: List[str], 
                       metadata_list: Optional[List[dict]] = None) -> None:
        """
        验证输入参数的有效性
        
        Args:
            texts: 文本列表
            metadata_list: 元信息列表
            
        Raises:
            TypeError: 当参数类型不正确时
            ValueError: 当参数值不符合要求时
        """
        if not isinstance(texts, list):
            raise TypeError("texts必须是列表类型")
        
        # 空列表是允许的，但会在后续处理中返回空结果
        if not texts:
            logger.info("输入文本列表为空，将返回空结果")
            return
            
        # 验证文本列表中的元素
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"texts[{i}]必须是字符串类型")
        
        # 验证元信息列表
        if metadata_list is not None:
            if not isinstance(metadata_list, list):
                raise TypeError("metadata_list必须是列表类型")
                
            if len(metadata_list) != len(texts):
                raise ValueError(f"元信息数量({len(metadata_list)})与文本数量({len(texts)})不匹配")
                
            for i, meta in enumerate(metadata_list):
                if not isinstance(meta, dict):
                    raise TypeError(f"metadata_list[{i}]必须是字典类型")

    def filter_short_texts(self, 
                          texts: List[str], 
                          metadata_list: Optional[List[dict]] = None) -> Tuple[List[str], Optional[List[dict]]]:
        """
        过滤过短的文本
        
        Args:
            texts: 文本列表
            metadata_list: 元信息列表，可选
            
        Returns:
            Tuple[List[str], Optional[List[dict]]]: 过滤后的文本和元信息列表
        """
        filtered_texts = []
        filtered_metadata = [] if metadata_list else None
        
        for i, text in enumerate(texts):
            if len(text.strip()) >= self.min_length:
                filtered_texts.append(text)
                if metadata_list:
                    filtered_metadata.append(metadata_list[i])
            else:
                logger.debug(f"过滤掉过短文本 (长度: {len(text.strip())}, ID: {i})")
        
        logger.info(f"文本过滤完成: 原始{len(texts)}条 → 有效{len(filtered_texts)}条")
        return filtered_texts, filtered_metadata

    def get_strategy_info(self) -> dict:
        """
        获取策略信息
        
        Returns:
            dict: 包含策略配置信息的字典
        """
        return {
            'strategy_name': self.strategy_name,
            'min_length': self.min_length,
            'chunk_size': getattr(self.text_splitter, 'chunk_size', None),
            'chunk_overlap': getattr(self.text_splitter, 'chunk_overlap', None)
        }

    def __repr__(self) -> str:
        """返回策略的字符串表示"""
        return f"{self.strategy_name}(min_length={self.min_length})"