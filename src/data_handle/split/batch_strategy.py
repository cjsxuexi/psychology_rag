# -*- coding: utf-8 -*-
"""
批量文本分割策略模块

提供单进程批量文本分割功能，适用于中小型数据集处理。
该策略通过批处理机制平衡内存使用和处理效率。

主要特性：
1. 内存友好：分批处理避免内存溢出
2. 进度追踪：集成tqdm进度条显示
3. 元信息保持：完整保留原始元信息结构
4. 错误容忍：单条文本处理失败不影响整体流程

适用场景：
- 中小型数据集（万级别以下）
- 内存受限环境
- 对处理时间要求不严格的场景
"""

from typing import List, Optional, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import logging

from .base_strategy import BaseTextSplitterStrategy

logger = logging.getLogger(__name__)


class BatchTextSplitterStrategy(BaseTextSplitterStrategy):
    """
    批量文本分割策略（单进程）
    
    采用分批处理的方式进行文本分割，在保证处理质量的同时
    控制内存使用，适合中小型数据集的处理需求。
    
    Attributes:
        Inherits all attributes from BaseTextSplitterStrategy
        
    Example:
        >>> from langchain_text_splitters import RecursiveCharacterTextSplitter
        >>> splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        >>> strategy = BatchTextSplitterStrategy(splitter, min_length=50)
        >>> texts, metadata = strategy.split_texts(["文本1", "文本2"], batch_size=1000)
    """

    def __init__(self, text_splitter: RecursiveCharacterTextSplitter, min_length: int):
        """
        初始化批量分割策略
        
        Args:
            text_splitter: LangChain文本分割器实例
            min_length: 最小文本长度阈值
            
        Raises:
            TypeError: 参数类型错误
            ValueError: 参数值错误
        """
        super().__init__(text_splitter, min_length)
        logger.info(f"批量分割策略初始化完成: {self.strategy_name}")

    def _split_texts(self, 
                    texts: List[str], 
                    metadata_list: Optional[List[dict]] = None, 
                    batch_size: int = 10_000) -> Tuple[List[str], List[dict]]:
        """
        批量分割文本列表的具体实现
        
        Args:
            texts: 已过滤的文本列表
            metadata_list: 对应的元信息列表，可选
            batch_size: 批处理大小，默认10000
            
        Returns:
            Tuple[List[str], List[dict]]: (分割后的文本列表, 对应的元信息列表)
            
        Note:
            - 为每个chunk复制对应的元信息
            - 保持原始元信息结构不变
        """
        # 初始化结果容器
        split_texts: List[str] = []
        split_metadata: List[dict] = []
        
        # 批量处理
        total_batches = (len(texts) + batch_size - 1) // batch_size
        logger.info(f"开始批量分割: {len(texts)}条文本，{total_batches}个批次")
        
        try:
            for batch_idx in tqdm(range(0, len(texts), batch_size), 
                                desc=f"{self.strategy_name} 分割进度",
                                unit="batch"):
                # 获取当前批次数据
                batch_end = min(batch_idx + batch_size, len(texts))
                batch_texts = texts[batch_idx:batch_end]
                batch_metadata = (metadata_list[batch_idx:batch_end] 
                                if metadata_list else [None] * len(batch_texts))
                
                # 处理当前批次
                batch_results = self._process_batch(batch_texts, batch_metadata)
                batch_split_texts, batch_split_metadata = batch_results
                
                # 合并结果
                split_texts.extend(batch_split_texts)
                split_metadata.extend(batch_split_metadata)
                
                logger.debug(f"批次 {batch_idx//batch_size + 1}/{total_batches} "
                           f"处理完成: 生成{len(batch_split_texts)}个文本块")
            
            logger.info(f"批量分割完成: 输出{len(split_texts)}个文本块")
            return split_texts, split_metadata
            
        except Exception as e:
            logger.error(f"批量分割过程中发生错误: {str(e)}")
            raise

    def _process_batch(self, 
                      batch_texts: List[str], 
                      batch_metadata: List[Optional[dict]]) -> Tuple[List[str], List[dict]]:
        """
        处理单个批次的文本
        
        Args:
            batch_texts: 批次文本列表
            batch_metadata: 批次元信息列表
            
        Returns:
            Tuple[List[str], List[dict]]: 批次处理结果
        """
        batch_split_texts: List[str] = []
        batch_split_metadata: List[dict] = []
        
        # 逐条处理文本
        for text, meta in zip(batch_texts, batch_metadata):
            try:
                # 分割单条文本
                chunks = self.split_single_text(text)
                
                if chunks:  # 只处理有内容的分割结果
                    batch_split_texts.extend(chunks)
                    
                    # 为每个chunk复制元信息
                    if meta:
                        batch_split_metadata.extend([meta.copy() for _ in chunks])
                    else:
                        batch_split_metadata.extend([{} for _ in chunks])
                else:
                    logger.debug("文本分割结果为空")
                    
            except Exception as e:
                logger.error(f"处理单条文本时发生错误: {str(e)}")
                # 继续处理其他文本，不中断整个批次
        
        return batch_split_texts, batch_split_metadata

    def get_processing_stats(self) -> dict:
        """
        获取处理统计信息
        
        Returns:
            dict: 包含处理统计信息的字典
        """
        base_info = self.get_strategy_info()
        return {
            **base_info,
            'processing_mode': 'batch',
            'concurrency': 'single_process'
        }

    def __repr__(self) -> str:
        """返回策略的详细字符串表示"""
        return (f"{self.strategy_name}("
                f"min_length={self.min_length}, "
                f"chunk_size={getattr(self.text_splitter, 'chunk_size', 'unknown')})")