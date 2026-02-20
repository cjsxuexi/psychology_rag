#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基础流处理器抽象类
提供通用的流式数据处理框架，包括检查点管理、资源监控、队列管理等功能
可被具体的数据处理器继承使用
"""

import gc
import json
import time
import os
from abc import ABC, abstractmethod
from typing import Generator, List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
import threading
from loguru import logger

import numpy as np
from tqdm import tqdm

# GPU内存监控（如果可用）
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class BaseStreamProcessor(ABC):
    """
    基础流处理器抽象类
    
    提供通用的流式处理框架，包含：
    1. 检查点管理（断点续传）
    2. 资源监控（内存、CPU、GPU）
    3. 队列管理和流水线协调
    4. 统计信息收集
    5. 错误处理和恢复机制
    
    子类需要实现具体的处理逻辑方法
    """
    
    def __init__(self,
                 batch_size: int = 100,
                 max_workers: int = 2,
                 memory_limit_mb: int = 4000,
                 buffer_size: int = 50,
                 gpu_memory_limit_gb: float = 4.0,
                 checkpoint_interval: int = 1000):
        """
        初始化基础流处理器
        
        Args:
            batch_size: 处理批次大小
            max_workers: 最大工作线程数
            memory_limit_mb: 内存使用上限(MB)
            buffer_size: 流水线缓冲区大小
            gpu_memory_limit_gb: GPU内存限制(GB)
            checkpoint_interval: 检查点保存间隔
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.memory_limit_mb = memory_limit_mb
        self.buffer_size = buffer_size
        self.gpu_memory_limit_gb = gpu_memory_limit_gb
        self.checkpoint_interval = checkpoint_interval
        
        # 运行状态监控
        self.stats = {
            'processed_items': 0,
            'failed_items': 0,
            'start_time': None,
            'memory_peak_mb': 0
        }
        

        
        # 线程安全队列
        self.raw_queue = Queue(maxsize=buffer_size)  # 原始数据队列
        self.processed_queue = Queue(maxsize=buffer_size)  # 处理后数据队列
        
        # 控制信号
        self.stop_event = threading.Event()
        self.pipeline_error = None
        self.pipeline_finished = {'reader': False, 'processor': False}
        
        # GPU监控
        self.gpu_monitor_enabled = TORCH_AVAILABLE and torch.cuda.is_available()
        
        # 检查点管理
        self.checkpoint_file = None
        self.last_checkpoint = {}
        
    def _process_item(self, item: Any) -> Optional[Any]:
        """
        抽象方法：处理单个数据项
        子类必须实现此方法
        
        Args:
            item: 输入数据项
            
        Returns:
            处理后的数据项或None（表示跳过）
        """
        pass
    
    def _data_reader(self, max_items: Optional[int] = None) -> Generator[Any, None, None]:
        """
        抽象方法：数据读取器
        子类必须实现此方法，返回数据生成器
        
        Args:
            max_items: 最大读取条目数
            
        Yields:
            数据项
        """
        pass
    
    def _set_checkpoint_file(self, filename: str):
        """
        设置检查点文件名
        
        Args:
            filename: 检查点文件名
        """
        self.checkpoint_file = filename
        self.last_checkpoint = self._load_checkpoint()
    
    def _save_checkpoint(self, item_id: int):
        """
        保存检查点
        
        Args:
            item_id: 当前处理的数据项ID
        """
        try:
            if not self.checkpoint_file:
                return
                
            checkpoint_data = {
                'last_processed_id': item_id,
                'timestamp': time.time(),
                'stats': self.stats.copy()
            }
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存检查点失败: {e}")
    
    def _load_checkpoint(self) -> Dict:
        """
        加载检查点
        
        Returns:
            检查点数据字典
        """
        try:
            if self.checkpoint_file and os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"加载检查点失败: {e}")
        return {}
    
    def _monitor_resources(self):
        """监控资源使用情况"""
        try:
            import psutil
            process = psutil.Process()
            
            while not self.stop_event.is_set():
                try:
                    # 监控CPU内存
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    self.stats['memory_peak_mb'] = max(self.stats['memory_peak_mb'], memory_mb)
                    
                    # 监控GPU内存（如果可用）
                    gpu_memory_gb = 0
                    if self.gpu_monitor_enabled:
                        try:
                            gpu_memory_gb = torch.cuda.memory_allocated() / 1024**3
                            if gpu_memory_gb > self.gpu_memory_limit_gb * 0.9:
                                logger.warning(f"GPU内存使用接近限制 ({gpu_memory_gb:.1f}GB/{self.gpu_memory_limit_gb}GB)")
                                torch.cuda.empty_cache()
                        except Exception as e:
                            logger.debug(f"GPU监控异常: {e}")
                    
                    # 内存压力管理
                    if memory_mb > max(2000, self.memory_limit_mb * 0.8):  # 固定2000MB阈值
                        logger.info(f"内存使用较高 ({memory_mb:.1f}MB)，执行主动清理")
                        self._aggressive_cleanup()
                    
                    time.sleep(2)  # 每2秒检查一次
                    
                except Exception as e:
                    logger.debug(f"资源监控异常: {e}")
                    break
                    
        except ImportError:
            logger.warning("psutil未安装，无法监控资源使用情况")
        except Exception as e:
            logger.debug(f"资源监控启动失败: {e}")
    
    def _aggressive_cleanup(self):
        """激进的内存清理策略"""
        try:
            # 清理Python垃圾回收
            collected = gc.collect()
            logger.debug(f"垃圾回收清理了 {collected} 个对象")
            
            # 清理PyTorch缓存（如果可用）
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("已清理GPU缓存")
                
            # 强制释放numpy数组引用（如果sys模块支持）
            try:
                import sys
                if hasattr(sys, 'getreferrers'):
                    for obj in list(sys.getreferrers(np.ndarray)):
                        if hasattr(obj, 'shape') and obj.shape[0] > 1000:
                            del obj
                else:
                    logger.debug("sys.getreferrers不可用，跳过numpy数组引用清理")
            except Exception as e:
                logger.debug(f"numpy数组清理异常: {e}")
                    
        except Exception as e:
            logger.debug(f"清理过程异常: {e}")
    
    def _reader_worker(self, max_items: Optional[int] = None):
        """数据读取工作线程"""
        try:
            start_from_id = self.last_checkpoint.get('last_processed_id', 0)
            item_count = 0
            skipped_items = 0
            
            logger.info(f"开始数据读取，起始ID: {start_from_id}")
            
            for item in self._data_reader(max_items):
                if self.stop_event.is_set() or self.pipeline_error:
                    break
                
                item_count += 1
                
                # 断点续传逻辑
                if hasattr(item, 'get'):
                    current_id = item.get('id', item_count)
                else:
                    current_id = item_count
                    
                if current_id <= start_from_id:
                    skipped_items += 1
                    continue
                
                # 等待队列有空间
                while self.raw_queue.full() and not self.stop_event.is_set():
                    time.sleep(0.1)
                
                self.raw_queue.put(item)
                self.stats['processed_items'] += 1
                
                # 定期保存检查点
                if item_count % self.checkpoint_interval == 0:
                    self._save_checkpoint(current_id)
                    logger.info(f"已处理 {item_count} 条数据，检查点已保存 (ID: {current_id})")
                
                # 定期报告进度
                if item_count % 1000 == 0:
                    logger.info(f"已读取 {item_count} 条数据 (跳过: {skipped_items})")
            
            self.pipeline_finished['reader'] = True
            logger.info(f"数据读取完成，总处理: {item_count} 条")
            
        except Exception as e:
            logger.error(f"数据读取线程异常: {e}")
            self.pipeline_error = e
    
    def _processor_worker(self):
        """数据处理工作线程"""
        try:
            while not self.stop_event.is_set():
                try:
                    # 从队列获取批量数据
                    batch_data = []
                    for _ in range(min(self.batch_size, self.raw_queue.qsize())):
                        try:
                            item = self.raw_queue.get(timeout=1)
                            batch_data.append(item)
                            self.raw_queue.task_done()
                        except Empty:
                            break
                    
                    if not batch_data:
                        if self.pipeline_finished.get('reader', False):
                            break
                        continue
                    
                    # 批量处理数据
                    processed_items = []
                    for item in batch_data:
                        try:
                            processed_item = self._process_item(item)
                            if processed_item is not None:
                                processed_items.append(processed_item)
                        except Exception as e:
                            logger.error(f"处理数据项失败: {e}")
                            self.stats['failed_items'] += 1
                    
                    # 将处理结果放入输出队列
                    for processed_item in processed_items:
                        self.processed_queue.put(processed_item)
                    
                except Exception as e:
                    logger.error(f"数据处理工作线程异常: {e}")
                    self.pipeline_error = e
                    break
                    
        except Exception as e:
            logger.error(f"数据处理线程崩溃: {e}")
            self.pipeline_error = e
    
    def process_stream(self, max_items: Optional[int] = None):
        """
        执行流式处理流程
        
        Args:
            max_items: 最大处理条目数
            
        Returns:
            Dict: 处理统计信息
        """
        logger.info("🚀 启动流式数据处理流程")
        logger.info(f"资源配置: CPU核心={self.max_workers}, 内存限制={self.memory_limit_mb}MB")
        
        # 检查是否有检查点需要恢复
        if self.last_checkpoint:
            logger.info(f"从检查点恢复处理: last_id={self.last_checkpoint.get('last_processed_id', 0)}")
            # 合并之前的统计数据
            prev_stats = self.last_checkpoint.get('stats', {})
            self.stats.update({k: v for k, v in prev_stats.items() if k in self.stats})
        
        self.stats['start_time'] = time.time()
        self.pipeline_finished = {'reader': False, 'processor': False}
        
        try:
            # 启动资源监控线程
            monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
            monitor_thread.start()
            
            # 启动处理线程池
            with ThreadPoolExecutor(max_workers=self.max_workers + 1) as executor:
                # 提交工作线程
                futures = []
                futures.append(executor.submit(self._processor_worker))
                
                # 等待工作线程初始化完成
                time.sleep(1)
                
                # 启动读取线程
                self._reader_worker(max_items)
                
                # 等待队列清空
                logger.info("等待原始队列清空...")
                self.raw_queue.join()
                self.pipeline_finished['processor'] = True
                logger.info("数据处理完成")
                
                logger.info("等待处理队列清空...")
                self.processed_queue.join()
                logger.info("输出队列处理完成")
                
                # 等待所有工作线程完成
                for future in as_completed(futures, timeout=300):  # 5分钟超时
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"工作线程异常: {e}")
                
        except Exception as e:
            logger.error(f"流式处理流程异常: {e}")
            self.stop_event.set()
            raise
        finally:
            self.stop_event.set()
            self._print_final_stats()
        

    def _get_final_stats(self) -> Dict:
        """获取最终统计信息"""
        elapsed_time = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        
        return {
            'processed_items': self.stats['processed_items'],
            'failed_items': self.stats['failed_items'],
            'processing_time_seconds': round(elapsed_time, 2),
            'processing_speed_ips': round(self.stats['processed_items'] / elapsed_time, 2) if elapsed_time > 0 else 0,
            'memory_peak_mb': round(self.stats['memory_peak_mb'], 1),
            'success_rate': f"{(self.stats['processed_items'] - self.stats['failed_items']) / max(self.stats['processed_items'], 1) * 100:.1f}%"
        }
    
    def _print_extended_stats(self, stats: Dict):
        """打印扩展统计信息（子类可覆盖此方法提供特定打印逻辑）"""
        # 默认不打印额外信息，子类可根据需要覆盖
        pass
    
    def _print_final_stats(self):
        """打印最终统计信息"""
        stats = self._get_final_stats()
        logger.success("📊 流式处理完成统计:")
        logger.success(f"   处理数据项: {stats['processed_items']}")
        logger.success(f"   失败数据项: {stats['failed_items']}")
        logger.success(f"   处理时间: {stats['processing_time_seconds']}秒")
        logger.success(f"   处理速度: {stats['processing_speed_ips']} 项/秒")
        logger.success(f"   内存峰值: {stats['memory_peak_mb']} MB")
        logger.success(f"   成功率: {stats['success_rate']}")
        
        # 打印扩展统计信息
        self._print_extended_stats(stats)
    
    def close(self):
        """关闭处理器，清理资源"""
        try:
            self.stop_event.set()
            logger.info("流处理器已关闭")
        except Exception as e:
            logger.error(f"关闭处理器失败: {str(e)}")


# 实用工具函数
def estimate_processing_time(total_items: int,
                           avg_processing_speed: float = 50.0,
                           overhead_factor: float = 1.3) -> Dict:
    """
    估算处理时间和资源需求
    
    Args:
        total_items: 总数据项数
        avg_processing_speed: 平均处理速度（项/秒）
        overhead_factor: 开销系数
        
    Returns:
        Dict: 时间估算结果
    """
    estimated_time = (total_items / avg_processing_speed) * overhead_factor
    
    return {
        'total_items': total_items,
        'estimated_seconds': round(estimated_time, 0),
        'estimated_minutes': round(estimated_time / 60, 1),
        'estimated_hours': round(estimated_time / 3600, 2),
        'recommended_batch_size': min(100, max(20, int(total_items / 1000))),
        'memory_estimate_mb': min(4000, max(1000, int(total_items * 0.5)))
    }


def cleanup_checkpoints(pattern: str = "checkpoint_*.json"):
    """清理检查点文件"""
    import glob
    
    checkpoint_files = glob.glob(pattern)
    for file_path in checkpoint_files:
        try:
            os.remove(file_path)
            logger.info(f"已删除检查点文件: {file_path}")
        except Exception as e:
            logger.warning(f"删除检查点文件失败 {file_path}: {e}")


def get_system_recommendations() -> Dict:
    """获取系统推荐配置"""
    try:
        import psutil
        
        cpu_count = psutil.cpu_count(logical=False)  # 物理核心数
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # 根据系统资源推荐配置
        recommendations = {
            'memory_limit_mb': min(6000, max(2000, int(total_memory_gb * 0.6 * 1024))),
            'batch_size': min(100, max(20, int(cpu_count * 10))),
            'max_workers': min(4, max(1, cpu_count - 1))
        }
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            recommendations['gpu_memory_limit_gb'] = min(6.0, max(2.0, gpu_memory_gb * 0.8))
        
        return recommendations
        
    except ImportError:
        logger.warning("psutil未安装，返回默认配置")
        return {
            'memory_limit_mb': 4000,
            'batch_size': 50,
            'max_workers': 2,
            'gpu_memory_limit_gb': 4.0
        }