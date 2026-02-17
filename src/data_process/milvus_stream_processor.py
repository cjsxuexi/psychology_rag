#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Milvus流式数据处理器
使用Milvus向量数据库替换原有的MySQL+FAISS双存储方案
实现边读取边处理边存储的高效数据处理流程
适用于资源受限环境（CPU 4核，内存 6G，GPU 1个）
"""

import gc
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
from typing import Generator, List, Dict, Optional, Tuple

import ijson
from loguru import logger
from pymilvus import (
    Collection, CollectionSchema, FieldSchema, DataType
)

from src.data_process.base_stream_processor import (
    estimate_processing_time,
    cleanup_checkpoints,
    get_system_recommendations
)
from src.common.file_utils import get_resource_path
from src.data_handle.json_handle import JsonHandle
from src.database.milvus_storage import create_milvus_storage
from src.embedding.M3EEmbedding import generate_dense_embeddings_with_m3e
from .base_stream_processor import BaseStreamProcessor

# GPU内存监控（如果可用）
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MilvusStreamProcessor(BaseStreamProcessor):
    """
    Milvus流式数据处理器
    继承自BaseStreamProcessor，专门处理心理学对话数据并存储到Milvus
    
    设计特点：
    1. 流水线处理：JSON读取 → 文本拆分 → 向量化 → Milvus存储
    2. 资源控制：限制并发数、缓冲区大小、内存使用
    3. 错误恢复：支持断点续传和部分失败重试
    4. 监控反馈：实时显示处理进度和资源使用情况
    5. 统一存储：使用Milvus替代MySQL+FAISS的双存储方案
    """

    def __init__(self,
                 json_file: str = "PsyDTCorpus_train_mulit_turn_packing.json",
                 batch_size: int = 100,
                 embedding_batch_size: int = 32,
                 max_workers: int = 2,  # 限制CPU使用
                 memory_limit_mb: int = 5000,  # 内存限制5G
                 buffer_size: int = 50,
                 gpu_memory_limit_gb: float = 4.0,  # GPU内存限制
                 checkpoint_interval: int = 1000,  # 检查点间隔
                 milvus_host: str = "localhost",
                 milvus_port: str = "19530",
                 collection_name: str = "psychology_dialogues"):
        """
        初始化Milvus流式处理器
        
        Args:
            json_file: JSON文件名
            batch_size: 文本处理批次大小
            embedding_batch_size: 向量化批次大小  
            max_workers: 最大工作线程数（控制CPU使用）
            memory_limit_mb: 内存使用上限(MB)
            buffer_size: 流水线缓冲区大小
            gpu_memory_limit_gb: GPU内存限制
            checkpoint_interval: 检查点间隔
            milvus_host: Milvus服务主机
            milvus_port: Milvus服务端口
            collection_name: Milvus集合名称
        """
        # 调用父类构造函数
        super().__init__(
            batch_size=batch_size,
            max_workers=max_workers,
            memory_limit_mb=memory_limit_mb,
            buffer_size=buffer_size,
            gpu_memory_limit_gb=gpu_memory_limit_gb,
            checkpoint_interval=checkpoint_interval
        )

        # Milvus特定属性
        self.json_file = json_file
        self.embedding_batch_size = embedding_batch_size
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = collection_name

        # 初始化组件
        self.json_handle = JsonHandle(json_file)
        # 使用新的MilvusStorage替代原有的连接管理
        self.milvus_storage = create_milvus_storage(
            host=milvus_host,
            port=milvus_port,
            collection_name=collection_name
        )

        # 扩展统计信息
        self.stats.update({
            'generated_chunks': 0,
            'embedded_chunks': 0,
            'stored_chunks': 0
        })

        # Milvus专用队列
        self.chunk_queue = Queue(maxsize=buffer_size)  # 拆分后的文本块
        self.embedding_queue = Queue(maxsize=buffer_size)  # 向量化数据
        self.storage_queue = Queue(maxsize=buffer_size)  # 待存储数据

        # 断点续传支持
        self._set_checkpoint_file(f"checkpoint_milvus_{json_file.replace('.json', '')}.json")

        # Milvus连接已由MilvusStorage自动初始化

    # _init_milvus方法已被移除，使用MilvusStorage自动管理连接

    def _create_collection(self):
        """创建Milvus集合"""
        try:
            # 定义新的字段结构
            fields = [
                FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="file_id", dtype=DataType.INT64),  # 文件ID
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),  # M3E向量维度
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5120),  # 文本内容
                FieldSchema(name="tag", dtype=DataType.VARCHAR, max_length=50),  # 标签
                FieldSchema(name="total_turns", dtype=DataType.INT64)  # 对话轮数
            ]

            schema = CollectionSchema(
                fields=fields,
                description="心理学对话数据集合"
            )

            # 创建集合
            self.milvus_collection = Collection(
                name=self.collection_name,
                schema=schema
            )

            # 创建索引
            index_params = {
                "index_type": "HNSW",
                "metric_type": "IP",  # 内积相似度（适合归一化向量）
                "params": {"M": 32, "efConstruction": 200}
            }

            self.milvus_collection.create_index(
                field_name="embedding",
                index_params=index_params
            )

            # 加载集合
            self.milvus_collection.load()

            logger.success(f"集合 {self.collection_name} 创建并加载成功")

        except Exception as e:
            logger.error(f"创建Milvus集合失败: {str(e)}")
            raise

    def _data_reader(self, max_items: Optional[int] = None) -> Generator[Dict, None, None]:
        """
        流式读取JSON数据（支持断点续传）
        调整为每次读取一个完整的item对象，不校验内部属性
        
        Args:
            max_items: 最大读取条目数
            
        Yields:
            Dict: 完整的JSON item对象
        """
        json_path = get_resource_path(self.json_file)
        item_count = 0

        logger.info(f"开始流式读取JSON文件: {json_path}")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                # 使用ijson.items直接读取数组中的每个完整对象
                items = ijson.items(f, "item")

                for item in items:
                    # 控制最大读取数量
                    if max_items and item_count >= max_items:
                        logger.info(f"达到最大读取数量限制: {max_items}")
                        break

                    # 记录读取进度
                    item_count += 1
                    if item_count % 1000 == 0:  # 每1000条报告一次进度
                        logger.info(f"已读取 {item_count} 个完整item对象")

                    # 直接yield完整的item对象，不进行任何校验
                    yield item

        except FileNotFoundError:
            logger.error(f"JSON文件不存在: {json_path}")
            raise
        except ijson.JSONError as e:
            logger.error(f"JSON格式错误: {e}")
            raise
        except Exception as e:
            logger.error(f"JSON读取失败: {e}")
            raise

    def _process_item(self, item: Dict) -> Optional[Tuple[str, Dict]]:
        """
        处理单个JSON数据项（RAG专用逻辑）
        调整为适配新的数据结构：item包含id、normalizedTag、messages三个属性
        
        Args:
            item: JSON数据项，包含{id, normalizedTag, messages}
            
        Returns:
            Tuple[str, Dict]: (处理后的文本, 元数据) 或 None
        """
        try:
            # 提取核心字段（适配新的数据结构）
            item_id = item.get("id", 0)
            tag = item.get("normalizedTag", "无标签")
            messages = item.get("messages", [])

            # 验证messages字段
            if not isinstance(messages, list) or len(messages) == 0:
                logger.debug(f"跳过无效messages数据项 ID: {item_id}")
                return None

            # 计算content数量，content数量/2作为total_turns
            content_count = sum(1 for msg in messages if msg.get("content", "").strip())
            total_turns = content_count // 2

            # 解析messages中的role和content，构造对话格式
            dialogue_parts = []
            role_mapping = {
                "user": "来访者",
                "assistant": "咨询师"
            }

            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "").strip()

                # 跳过system角色的消息，不纳入content拼接
                if role == "system":
                    continue

                if content:  # 只处理有内容的消息
                    role_cn = role_mapping.get(role, role)
                    dialogue_parts.append(f"{role_cn}：{content}")

            # 拼接全部content内容，不同角色会话用\n分隔
            dialogue_str = "\n".join(dialogue_parts) + "\n"

            # 调整metadata，存储id、tag、total_turns三个值
            metadata = {
                'id': item_id,
                'tag': tag,
                'total_turns': total_turns
            }

            logger.debug(f"已处理数据项 ID: {item_id}, 标签: {tag}, 对话轮数: {total_turns}")
            return dialogue_str, metadata

        except Exception as e:
            logger.error(f"处理JSON项失败: {e}")
            # 注意：这里不再直接修改stats，让BaseStreamProcessor统一处理
            return None

    def _text_splitter_worker(self):
        """文本拆分工作线程"""
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

                    # 批量处理文本拆分
                    texts = [item[0] for item in batch_data]
                    metadatas = [item[1] for item in batch_data]

                    split_texts, split_metadatas = self.json_handle.batch_split_texts(
                        texts,
                        batch_size=self.batch_size,
                        metadata_list=metadatas
                    )

                    # 将拆分结果放入队列
                    for text, meta in zip(split_texts, split_metadatas):
                        self.chunk_queue.put((text, meta))

                    self.stats['generated_chunks'] += len(split_texts)

                except Exception as e:
                    logger.error(f"文本拆分工作线程异常: {e}")
                    self.pipeline_error = e
                    break

        except Exception as e:
            logger.error(f"文本拆分线程崩溃: {e}")
            self.pipeline_error = e

    def _embedding_worker(self):
        """向量化工作线程"""
        try:
            while not self.stop_event.is_set():
                try:
                    # 收集一批文本进行向量化
                    batch_texts = []
                    batch_metadatas = []

                    for _ in range(min(self.embedding_batch_size, self.chunk_queue.qsize())):
                        try:
                            text, meta = self.chunk_queue.get(timeout=1)
                            batch_texts.append(text)
                            batch_metadatas.append(meta)
                            self.chunk_queue.task_done()
                        except Empty:
                            break

                    if not batch_texts:
                        if self.pipeline_finished.get('splitter', False):
                            break
                        continue

                    # 批量向量化
                    embeddings = generate_dense_embeddings_with_m3e(batch_texts)

                    # 将结果放入队列
                    for i, (embedding, meta) in enumerate(zip(embeddings, batch_metadatas)):
                        self.embedding_queue.put((embedding, batch_texts[i], meta))

                    self.stats['embedded_chunks'] += len(batch_texts)

                    # 定期清理内存
                    if self.stats['embedded_chunks'] % 1000 == 0:
                        gc.collect()

                except Exception as e:
                    logger.error(f"向量化工作线程异常: {e}")
                    self.pipeline_error = e
                    break

        except Exception as e:
            logger.error(f"向量化线程崩溃: {e}")
            self.pipeline_error = e

    def _get_max_file_id(self) -> int:
        """
        获取当前集合中的最大file_id
        
        Returns:
            int: 最大file_id，如果集合为空则返回0
        """
        try:
            return self.milvus_storage.get_max_file_id()
        except Exception as e:
            logger.warning(f"获取最大file_id失败: {str(e)}，使用默认值0")
            return 0

    def _storage_worker(self):
        """Milvus存储工作线程 - 使用新的MilvusStorage接口"""
        try:
            texts_buffer = []
            embeddings_buffer = []
            metadata_buffer = []

            while not self.stop_event.is_set():
                try:
                    # 收集数据进行批量存储
                    try:
                        embedding, text, meta = self.embedding_queue.get(timeout=1)
                        texts_buffer.append(text)
                        embeddings_buffer.append(embedding)
                        metadata_buffer.append(meta)
                        self.embedding_queue.task_done()
                    except Empty:
                        pass

                    # 批量插入到Milvus（达到批次大小或队列为空且处理完成）
                    should_flush = (
                            len(texts_buffer) >= self.batch_size or
                            (self.pipeline_finished.get('embedder', False) and self.embedding_queue.empty())
                    )

                    if should_flush and texts_buffer:
                        try:
                            # 使用MilvusStorage的批量插入方法
                            inserted_count = self.milvus_storage.insert_data(
                                texts=texts_buffer,
                                embeddings=embeddings_buffer,
                                metadata_list=metadata_buffer,
                                batch_size=self.batch_size
                            )

                            self.stats['stored_chunks'] += inserted_count
                            logger.info(f"已存储 {inserted_count} 条数据到Milvus")

                            # 清空缓冲区
                            texts_buffer.clear()
                            embeddings_buffer.clear()
                            metadata_buffer.clear()

                        except Exception as e:
                            logger.error(f"Milvus存储失败: {e}")
                            self.pipeline_error = e
                            break

                except Exception as e:
                    logger.error(f"存储工作线程异常: {e}")
                    self.pipeline_error = e
                    break

        except Exception as e:
            logger.error(f"存储线程崩溃: {e}")
            self.pipeline_error = e

    def process_stream(self, max_items: Optional[int] = None):
        """
        执行流式处理流程（RAG专用）
        重写父类方法以适应Milvus的四阶段处理流程
        
        Args:
            max_items: 最大处理条目数
            
        Returns:
            Dict: 处理统计信息
        """
        logger.info("🚀 启动Milvus RAG流式数据处理流程")

        # 检查是否有检查点需要恢复
        if self.last_checkpoint:
            logger.info(f"从检查点恢复处理: last_id={self.last_checkpoint.get('last_processed_id', 0)}")
            # 合并之前的统计数据
            prev_stats = self.last_checkpoint.get('stats', {})
            self.stats.update({k: v for k, v in prev_stats.items() if k in self.stats})

        self.stats['start_time'] = time.time()
        self.pipeline_finished = {
            'reader': False,
            'splitter': False,
            'embedder': False,
            'storage': False
        }

        try:
            # 启动资源监控线程
            monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
            monitor_thread.start()

            # 启动处理线程池
            with ThreadPoolExecutor(max_workers=self.max_workers + 3) as executor:  # +3用于多个处理阶段
                # 提交各阶段工作线程
                futures = []
                futures.append(executor.submit(self._text_splitter_worker))
                futures.append(executor.submit(self._embedding_worker))
                futures.append(executor.submit(self._storage_worker))

                # 等待工作线程初始化完成
                time.sleep(1)

                # 流式读取并分发数据
                logger.info("开始流式读取和处理...")
                reader_generator = self._data_reader(max_items)

                item_count = 0
                for item in reader_generator:
                    if self.stop_event.is_set() or self.pipeline_error:
                        break

                    item_count += 1
                    processed_item = self._process_item(item)
                    if processed_item:
                        # 等待队列有空间再放入数据
                        while self.raw_queue.full() and not self.stop_event.is_set():
                            time.sleep(0.1)

                        self.raw_queue.put(processed_item)
                        self.stats['processed_items'] += 1

                    # 定期报告进度
                    if item_count % 10 == 0:  # 每10条报告一次
                        logger.info(f"已处理 {item_count} 条数据")

                self.pipeline_finished['reader'] = True

                # 等待各个阶段队列清空
                logger.info("等待文本拆分完成...")
                self.raw_queue.join()
                self.pipeline_finished['splitter'] = True
                logger.info("文本拆分完成")

                logger.info("等待向量化完成...")
                self.chunk_queue.join()
                self.pipeline_finished['embedder'] = True
                logger.info("向量化完成")

                logger.info("等待存储完成...")
                self.embedding_queue.join()
                self.pipeline_finished['storage'] = True
                logger.info("数据存储完成")

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

    def search_similar_chunks(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        在Milvus中搜索相似的文本块
        
        Args:
            query_text: 查询文本
            top_k: 返回top-k结果
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        try:
            # 使用MilvusStorage的搜索方法
            results = self.milvus_storage.search_similar(
                query_text=query_text,
                top_k=top_k,
                output_fields=["content", "tag", "total_turns"]
            )

            # 转换结果格式以保持兼容性
            similar_chunks = []
            for result in results:
                similar_chunks.append({
                    'content': result.get('content', ''),
                    'tag': result.get('tag', ''),
                    'total_turns': result.get('total_turns', 0),
                    'distance': result.get('distance', 0.0),
                    'chunk_id': result.get('chunk_id', 0)
                })

            return similar_chunks

        except Exception as e:
            logger.error(f"Milvus搜索失败: {str(e)}")
            raise

    def _print_extended_stats(self, stats: Dict):
        """打印RAG特定的扩展统计信息"""
        logger.success("📊 Milvus RAG处理额外统计:")
        logger.success(f"   生成文本块: {self.stats['generated_chunks']}")
        logger.success(f"   向量化文本: {self.stats['embedded_chunks']}")
        logger.success(f"   存储文本块: {self.stats['stored_chunks']}")

        # 使用MilvusStorage获取集合统计信息
        try:
            collection_stats = self.milvus_storage.get_collection_stats()
            logger.success(f"   Milvus实体数: {collection_stats.get('entity_count', 0)}")
        except Exception as e:
            logger.warning(f"获取Milvus统计信息失败: {e}")

    def close(self):
        """关闭连接"""
        try:
            # 使用MilvusStorage的关闭方法
            if hasattr(self, 'milvus_storage'):
                self.milvus_storage.close()
            logger.info("Milvus连接已关闭")
        except Exception as e:
            logger.error(f"关闭Milvus连接失败: {str(e)}")


# 便捷使用的包装函数
def stream_process_psychology_data_milvus(json_file: str = "PsyDTCorpus_train_mulit_turn_packing.json",
                                          max_items: Optional[int] = None,
                                          auto_configure: bool = True,
                                          milvus_host: str = "localhost",
                                          milvus_port: str = "19530",
                                          collection_name: str = "psychology_dialogues",
                                          **kwargs) -> Dict:
    """
    便捷的Milvus流式处理函数
    
    Args:
        json_file: JSON文件名
        max_items: 最大处理条目数
        auto_configure: 是否自动根据系统资源配置
        milvus_host: Milvus服务主机
        milvus_port: Milvus服务端口
        collection_name: Milvus集合名称
        **kwargs: MilvusStreamProcessor初始化参数
        
    Returns:
        Dict: 处理统计信息
    """
    # 自动配置优化参数
    if auto_configure:
        system_config = get_system_recommendations()
        system_config.update(kwargs)
        kwargs = system_config
        logger.info(f"自动配置参数: {system_config}")

    processor = None
    try:
        processor = MilvusStreamProcessor(
            json_file=json_file,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            collection_name=collection_name,
            **kwargs
        )
        return processor.process_stream(max_items)
    finally:
        if processor:
            processor.close()


def prd_test():
    """生产环境测试"""
    print("场景2: 生产环境处理（自动配置）")
    print("-" * 40)

    # 估算处理时间
    time_estimate = estimate_processing_time(total_items=10000)
    print(f"处理10000条数据预计需要: {time_estimate['estimated_minutes']} 分钟")

    # 获取系统推荐配置
    recommendations = get_system_recommendations()
    print(f"系统推荐配置: {recommendations}")

    # 执行处理（这里用较小的数量演示）
    try:
        production_result = stream_process_psychology_data_milvus(
            max_items=512,  # 实际使用时可以设置为None处理全部
            auto_configure=True
        )

        print(f"\n生产处理结果: {production_result}")

    except Exception as e:
        print(f"生产测试失败: {str(e)}")

    # 清理检查点文件
    cleanup_checkpoints()


if __name__ == "__main__":
    prd_test()
