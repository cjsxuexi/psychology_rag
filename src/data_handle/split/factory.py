# -*- coding: utf-8 -*-
"""
文本分割策略工厂模块

提供统一的策略创建接口，实现策略模式的核心工厂功能。
通过配置驱动的方式动态创建不同的文本分割策略实例。

设计模式：
- 工厂模式：封装对象创建过程
- 策略模式：支持运行时策略切换
- 单例模式：工厂类本身可以设计为单例

主要功能：
1. 动态创建策略实例
2. 策略注册和管理
3. 配置验证和默认值处理
4. 策略信息查询
"""
import os
from typing import Dict, Type, Optional, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

from .base_strategy import BaseTextSplitterStrategy
from .batch_strategy import BatchTextSplitterStrategy
from .parallel_strategy import ParallelBatchTextSplitterStrategy

logger = logging.getLogger(__name__)

# 策略类型映射
STRATEGY_REGISTRY: Dict[str, Type[BaseTextSplitterStrategy]] = {
    'batch': BatchTextSplitterStrategy,
    'parallel': ParallelBatchTextSplitterStrategy
}


class TextSplitterStrategyFactory:
    """
    文本分割策略工厂类
    
    负责根据配置参数创建相应的文本分割策略实例。
    提供策略注册、查询和创建的统一接口。
    
    Attributes:
        _registry (Dict[str, Type[BaseTextSplitterStrategy]]): 策略注册表
        
    Example:
        >>> from langchain_text_splitters import RecursiveCharacterTextSplitter
        >>> splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        >>> factory = TextSplitterStrategyFactory()
        >>> strategy = factory.create_strategy('batch', splitter, min_length=50)
        >>> texts, metadata = strategy.split_texts(['文本内容'])
    """

    def __init__(self):
        """初始化策略工厂"""
        self._registry = STRATEGY_REGISTRY.copy()
        logger.info("文本分割策略工厂初始化完成")
        logger.debug(f"已注册策略: {list(self._registry.keys())}")

    def create_strategy(self, 
                       text_splitter: Any,
                       min_length: int,
                        strategy_type: str | None = None,
                       **kwargs) -> BaseTextSplitterStrategy:
        """
        创建指定类型的文本分割策略实例
        
        Args:
            strategy_type: 策略类型 ('batch' | 'parallel')
            text_splitter: 文本分割器实例（必须具有split_text方法）
            min_length: 最小文本长度阈值
            **kwargs: 策略特定的额外参数
            
        Returns:
            BaseTextSplitterStrategy: 策略实例
            
        Raises:
            ValueError: 不支持的策略类型或参数错误
            TypeError: 参数类型错误
            
        Note:
            - 自动验证策略类型是否支持
            - 传递额外参数给具体策略构造函数
        """
        if not strategy_type:
            # 从配置读取拆分策略（默认batch）
            strategy_type = os.getenv("TEXT_SPLIT_STRATEGY", "batch")

        # 验证策略类型
        if strategy_type not in self._registry:
            available_strategies = list(self._registry.keys())
            raise ValueError(
                f"不支持的拆分策略: '{strategy_type}'，"
                f"可选值：{available_strategies}"
            )
        
        # 验证基本参数
        # 检查text_splitter是否具有split_text方法（支持自定义分割器类）
        if not hasattr(text_splitter, 'split_text') or not callable(getattr(text_splitter, 'split_text')):
            raise TypeError("text_splitter必须具有split_text方法")
        
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError("min_length必须是非负整数")
        
        try:
            # 获取策略类
            strategy_class = self._registry[strategy_type]
            
            # 创建策略实例
            strategy_instance = strategy_class(
                text_splitter=text_splitter,
                min_length=min_length,
                **kwargs
            )
            
            logger.info(f"成功创建策略实例: {strategy_type} -> {strategy_instance}")
            return strategy_instance
            
        except Exception as e:
            logger.error(f"创建策略实例失败: {strategy_type}, 错误: {str(e)}")
            raise

    def register_strategy(self, 
                         strategy_type: str, 
                         strategy_class: Type[BaseTextSplitterStrategy]) -> None:
        """
        注册新的策略类型
        
        Args:
            strategy_type: 策略类型名称
            strategy_class: 策略类（必须继承自BaseTextSplitterStrategy）
            
        Raises:
            ValueError: 策略类型已存在或类不符合要求
            TypeError: 参数类型错误
        """
        if not isinstance(strategy_type, str):
            raise TypeError("strategy_type必须是字符串")
            
        if not issubclass(strategy_class, BaseTextSplitterStrategy):
            raise ValueError("strategy_class必须继承自BaseTextSplitterStrategy")
        
        if strategy_type in self._registry:
            raise ValueError(f"策略类型 '{strategy_type}' 已存在")
        
        self._registry[strategy_type] = strategy_class
        logger.info(f"成功注册新策略: {strategy_type} -> {strategy_class.__name__}")

    def unregister_strategy(self, strategy_type: str) -> bool:
        """
        注销策略类型
        
        Args:
            strategy_type: 要注销的策略类型
            
        Returns:
            bool: 是否成功注销
            
        Note:
            不允许注销内置策略类型
        """
        if strategy_type not in self._registry:
            logger.warning(f"策略类型 '{strategy_type}' 不存在")
            return False
        
        # 防止注销内置策略
        if strategy_type in STRATEGY_REGISTRY:
            logger.warning(f"不允许注销内置策略: {strategy_type}")
            return False
        
        removed_class = self._registry.pop(strategy_type)
        logger.info(f"成功注销策略: {strategy_type} -> {removed_class.__name__}")
        return True

    def get_available_strategies(self) -> Dict[str, str]:
        """
        获取所有可用的策略信息
        
        Returns:
            Dict[str, str]: 策略类型到描述信息的映射
        """
        strategies_info = {}
        for strategy_type, strategy_class in self._registry.items():
            doc = strategy_class.__doc__
            if doc:
                strategies_info[strategy_type] = doc.split('\n')[0].strip()
            else:
                strategies_info[strategy_type] = strategy_class.__name__
        return strategies_info

    def get_strategy_info(self, strategy_type: str) -> Optional[Dict[str, Any]]:
        """
        获取指定策略的详细信息
        
        Args:
            strategy_type: 策略类型
            
        Returns:
            Optional[Dict[str, Any]]: 策略信息字典，如果策略不存在则返回None
        """
        if strategy_type not in self._registry:
            return None
            
        strategy_class = self._registry[strategy_type]
        return {
            'type': strategy_type,
            'class_name': strategy_class.__name__,
            'module': strategy_class.__module__,
            'doc': strategy_class.__doc__,
            'is_builtin': strategy_type in STRATEGY_REGISTRY
        }

    def validate_strategy_config(self, 
                               strategy_type: str,
                               config: Dict[str, Any]) -> bool:
        """
        验证策略配置的有效性
        
        Args:
            strategy_type: 策略类型
            config: 配置字典
            
        Returns:
            bool: 配置是否有效
        """
        if strategy_type not in self._registry:
            return False
            
        # 基本必需参数检查
        required_params = ['text_splitter', 'min_length']
        for param in required_params:
            if param not in config:
                logger.error(f"配置缺少必需参数: {param}")
                return False
                
        return True

    @classmethod
    def get_default_strategy(cls) -> str:
        """
        获取默认策略类型
        
        Returns:
            str: 默认策略类型
        """
        return 'batch'

    @classmethod
    def create_with_defaults(cls,
                           strategy_type: Optional[str] = None,
                           **kwargs) -> BaseTextSplitterStrategy:
        """
        使用默认配置创建策略实例（便捷方法）
        
        Args:
            strategy_type: 策略类型，None表示使用默认策略
            **kwargs: 其他参数
            
        Returns:
            BaseTextSplitterStrategy: 策略实例
        """
        factory = cls()
        strategy_type = strategy_type or cls.get_default_strategy()
        
        # 创建默认的文本分割器
        default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=kwargs.get('chunk_size', 1000),
            chunk_overlap=kwargs.get('chunk_overlap', 100),
            length_function=len,
            separators=["\n\n", "\n", "。", "，", " "]
        )
        
        return factory.create_strategy(
            strategy_type=strategy_type,
            text_splitter=default_splitter,
            min_length=kwargs.get('min_length', 50),
            **{k: v for k, v in kwargs.items() 
               if k not in ['chunk_size', 'chunk_overlap', 'min_length']}
        )

    def __repr__(self) -> str:
        """返回工厂的字符串表示"""
        return f"TextSplitterStrategyFactory(registered_strategies={list(self._registry.keys())})"


# 便捷函数
def create_text_splitter_strategy(strategy_type: str,
                                text_splitter: Any,
                                min_length: int,
                                **kwargs) -> BaseTextSplitterStrategy:
    """
    便捷函数：创建文本分割策略实例
    
    Args:
        strategy_type: 策略类型
        text_splitter: 文本分割器实例（必须具有split_text方法）
        min_length: 最小长度
        **kwargs: 其他参数
        
    Returns:
        BaseTextSplitterStrategy: 策略实例
    """
    factory = TextSplitterStrategyFactory()
    return factory.create_strategy(strategy_type, text_splitter, min_length, **kwargs)


def get_available_strategies() -> Dict[str, str]:
    """
    便捷函数：获取可用策略列表
    
    Returns:
        Dict[str, str]: 策略信息
    """
    factory = TextSplitterStrategyFactory()
    return factory.get_available_strategies()