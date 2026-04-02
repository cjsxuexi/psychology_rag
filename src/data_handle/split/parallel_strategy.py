# -*- coding: utf-8 -*-
"""
并行批量文本分割策略模块

提供多进程并行文本分割功能，显著提升大数据集处理效率。
通过进程池并行处理多个文本分割任务，充分利用多核CPU资源。

主要特性：
1. 高性能：多进程并行处理，大幅提升处理速度
2. 资源管理：智能控制进程数量，避免系统过载
3. 元信息丰富：为每个chunk添加详细的处理信息
4. 错误隔离：单个进程失败不影响其他进程

适用场景：
- 大型数据集（十万级别以上）
- CPU多核环境
- 对处理时间敏感的应用
- 可以接受稍高内存消耗的场景

注意：由于使用多进程，需要注意pickle序列化的限制
"""

import logging
import os
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from .base_strategy import BaseTextSplitterStrategy

logger = logging.getLogger(__name__)


class ParallelBatchTextSplitterStrategy(BaseTextSplitterStrategy):
    """
    并行批量文本分割策略（多进程）
    
    利用多进程并行处理文本分割任务，显著提升大规模数据集的处理效率。
    通过进程池管理并行任务，自动平衡负载并控制资源使用。
    
    Attributes:
        Inherits all attributes from BaseTextSplitterStrategy
        max_processes (int): 最大进程数，默认为CPU核心数
        chunk_timeout (int): 单个chunk处理超时时间（秒）
        
    Example:
        >>> from langchain_text_splitters import RecursiveCharacterTextSplitter
        >>> splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        >>> strategy = ParallelBatchTextSplitterStrategy(splitter, min_length=50)
        >>> texts, metadata = strategy.split_texts(["大量文本数据"], batch_size=5000)
    """

    def __init__(self,
                 text_splitter: RecursiveCharacterTextSplitter,
                 min_length: int,
                 max_processes: Optional[int] = None,
                 chunk_timeout: int = 300):
        """
        初始化并行分割策略
        
        Args:
            text_splitter: LangChain文本分割器实例
            min_length: 最小文本长度阈值
            max_processes: 最大进程数，None表示使用CPU核心数
            chunk_timeout: 单个chunk处理超时时间（秒）
            
        Raises:
            TypeError: 参数类型错误
            ValueError: 参数值错误
        """
        super().__init__(text_splitter, min_length)

        # 设置进程数
        if max_processes is not None:
            self.max_processes = max_processes
        else:
            # 从环境变量获取配置
            cpu_percentage = int(os.getenv("PARALLEL_CPU_PERCENTAGE", 75))

            # 计算基于CPU百分比的进程数
            cpu_count_val = cpu_count()
            # 确保至少有1个进程，并且百分比值合理
            cpu_percentage = max(1, min(100, cpu_percentage))
            self.max_processes = max(1, int(cpu_count_val * cpu_percentage / 100))
        self.chunk_timeout = chunk_timeout

        logger.info(f"并行分割策略初始化完成: {self.strategy_name}")
        logger.info(f"配置参数 - 进程数: {self.max_processes}, 超时: {chunk_timeout}秒")

    def _split_texts(self, 
                    texts: List[str], 
                    metadata_list: Optional[List[dict]] = None, 
                    batch_size: int = 500) -> Tuple[List[str], List[dict]]:
        """
        并行批量分割文本列表的具体实现
        
        Args:
            texts: 已过滤的文本列表
            metadata_list: 对应的元信息列表，可选
            batch_size: 批处理大小，默认500
            
        Returns:
            Tuple[List[str], List[dict]]: (分割后的文本列表, 对应的元信息列表)
            
        Note:
            - 使用多进程并行处理
            - 为每个chunk添加详细的处理元信息
            - 自动进行结果一致性验证
        """
        # 初始化结果容器
        split_texts: List[str] = []
        split_metadata: List[dict] = []

        # 计算批次信息
        total_items = len(texts)
        total_batches = (total_items + batch_size - 1) // batch_size
        logger.info(f"开始并行分割: {total_items}条文本，{total_batches}个批次，{self.max_processes}个进程")

        try:
            # 创建进程池
            with Pool(processes=self.max_processes) as pool:
                logger.debug(f"进程池创建成功，进程数: {self.max_processes}")

                # 分批处理
                for batch_idx in tqdm(range(0, total_items, batch_size),
                                      desc=f"{self.strategy_name} 并行分割进度",
                                      unit="batch"):
                    # 获取当前批次数据
                    batch_end = min(batch_idx + batch_size, total_items)
                    batch_texts = texts[batch_idx:batch_end]
                    batch_metadata = (metadata_list[batch_idx:batch_end]
                                      if metadata_list else [None] * len(batch_texts))

                    # 并行处理当前批次
                    batch_results = self._process_batch_parallel(pool, batch_texts, batch_metadata, batch_idx)
                    batch_split_texts, batch_split_metadata = batch_results

                    # 合并结果
                    split_texts.extend(batch_split_texts)
                    split_metadata.extend(batch_split_metadata)

                    logger.debug(f"批次 {batch_idx // batch_size + 1}/{total_batches} "
                                 f"处理完成: 生成{len(batch_split_texts)}个文本块")

            # 验证结果一致性
            self._validate_results_consistency(split_texts, split_metadata)

            logger.info(f"并行分割完成: 输出{len(split_texts)}个文本块")
            return split_texts, split_metadata

        except Exception as e:
            logger.error(f"并行分割过程中发生错误: {str(e)}")
            raise

    def _process_batch_parallel(self,
                                pool: Pool,
                                batch_texts: List[str],
                                batch_metadata: List[Optional[dict]],
                                batch_start_idx: int) -> Tuple[List[str], List[dict]]:
        """
        并行处理单个批次的文本
        
        Args:
            pool: 进程池实例
            batch_texts: 批次文本列表
            batch_metadata: 批次元信息列表
            batch_start_idx: 批次起始索引
            
        Returns:
            Tuple[List[str], List[dict]]: 批次处理结果
        """
        batch_split_texts: List[str] = []
        batch_split_metadata: List[dict] = []
        total_chunks = 0

        try:
            # 准备并行任务参数，只传递文本
            parallel_args = [(text,) for text in batch_texts]

            # 并行执行文本分割
            results = pool.starmap(
                self._parallel_split_worker,
                parallel_args,
                chunksize=max(1, len(batch_texts) // self.max_processes)
            )

            # 处理并行结果
            for item_idx, (chunks, original_text) in enumerate(zip(results, batch_texts)):
                original_item_idx = batch_start_idx + item_idx
                meta = batch_metadata[item_idx] if batch_metadata else None

                # 为每个chunk分配元信息
                for chunk_idx, chunk_text in enumerate(chunks):
                    batch_split_texts.append(chunk_text)
                    total_chunks += 1

                    if meta:
                        # 复制元信息并添加chunk级别的详细信息
                        chunk_meta = meta.copy()
                        chunk_meta.update({
                            'original_item_index': original_item_idx,
                            'chunk_index_in_item': chunk_idx,
                            'chunk_index_global': len(batch_split_texts) - 1,
                            'total_chunks_in_item': len(chunks),
                            'original_text_length': len(original_text),
                            'chunk_length': len(chunk_text),
                            'processing_strategy': self.strategy_name
                        })
                        batch_split_metadata.append(chunk_meta)
                    else:
                        # 如果没有元信息，创建基础结构
                        batch_split_metadata.append({
                            'original_item_index': original_item_idx,
                            'chunk_index_in_item': chunk_idx,
                            'chunk_index_global': len(batch_split_texts) - 1,
                            'total_chunks_in_item': len(chunks),
                            'original_text_length': len(original_text),
                            'chunk_length': len(chunk_text),
                            'processing_strategy': self.strategy_name
                        })

            logger.debug(f"并行批次处理完成: 处理{len(batch_texts)}个原始项，生成{total_chunks}个文本块")

        except Exception as e:
            logger.error(f"并行批次处理失败: {str(e)}")
            raise

        return batch_split_texts, batch_split_metadata

    @staticmethod
    def _parallel_split_worker(text: str) -> List[str]:
        """
        并行工作进程的文本分割函数
        
        注意：这是一个静态方法，因为需要被pickle序列化传递给子进程
        
        Args:
            text: 待分割的文本
            
        Returns:
            List[str]: 分割后的文本块列表
        """
        from src.data_handle.parser import create_splitter

        if not text or not text.strip():
            return []

        # 在子进程中使用create_splitter创建分割器（自动从配置读取）
        splitter = create_splitter()

        # 使用分割器进行文本分割
        chunks = splitter.split_text(text)

        return chunks

    def _validate_results_consistency(self,
                                      split_texts: List[str],
                                      split_metadata: List[dict]) -> None:
        """
        验证处理结果的一致性
        
        Args:
            split_texts: 分割后的文本列表
            split_metadata: 对应的元信息列表
            
        Raises:
            RuntimeError: 当结果不一致时
        """
        if len(split_texts) != len(split_metadata):
            raise RuntimeError(
                f"结果不一致: 文本块数量({len(split_texts)}) ≠ 元信息数量({len(split_metadata)})"
            )

        # 验证元信息结构
        for i, meta in enumerate(split_metadata):
            required_fields = ['original_item_index', 'chunk_index_in_item']
            missing_fields = [field for field in required_fields if field not in meta]
            if missing_fields:
                logger.warning(f"元信息缺失必要字段 {missing_fields} at index {i}")

    def get_processing_stats(self) -> dict:
        """
        获取处理统计信息
        
        Returns:
            dict: 包含处理统计信息的字典
        """
        base_info = self.get_strategy_info()
        return {
            **base_info,
            'processing_mode': 'parallel_batch',
            'concurrency': 'multi_process',
            'max_processes': self.max_processes,
            'chunk_timeout': self.chunk_timeout
        }

    def __repr__(self) -> str:
        """返回策略的详细字符串表示"""
        return (f"{self.strategy_name}("
                f"min_length={self.min_length}, "
                f"processes={self.max_processes}, "
                f"timeout={self.chunk_timeout}s)")