import time
from typing import Optional, Dict

from langchain_core.runnables import RunnableLambda
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.data_process.base_stream_processor import get_system_recommendations
from src.data_process.milvus_stream_processor import MilvusStreamProcessor
from src.embedding.M3EEmbedding import generate_dense_embeddings_with_m3e


class MilvusLCELStreamProcessor(MilvusStreamProcessor):

    def create_lcel_pipeline(self):
        """
        创建基于 Langchain LCEL 的处理管道
        """

        # 步骤1: JSON解析分块（批量处理）
        def json_parser_step(params):
            """JSON解析分块步骤，每20个chunk返回一次"""
            json_file = params["json_file"]
            max_items = params["max_items"]

            batch_size = 20
            batch = []
            total_count = 0

            # 直接处理JSON数据并批量返回
            for text_chunks, chunk_metadata in self.json_handle.parallel_handle_json_data(
                    json_name=json_file,
                    max_items=max_items
            ):
                for text, meta in zip(text_chunks, chunk_metadata):
                    item = {"text": text, "metadata": meta}
                    batch.append(item)
                    total_count += 1

                    # 当批量达到20个时返回
                    if len(batch) >= batch_size:
                        logger.info(f"Returning batch of {len(batch)} items (total: {total_count})")
                        yield batch
                        batch = []

            # 返回剩余的不足一批的数据
            if batch:
                logger.info(f"Returning final batch of {len(batch)} items (total: {total_count})")
                yield batch

            logger.info(f"Parsed total {total_count} items from JSON")

        # 步骤2: 向量化处理
        def embedder(input_items):
            """向量化处理函数"""
            logger.info(f"Embedder received input: {type(input_items)}")
            # 确保 input_items 是可迭代的
            if isinstance(input_items, dict):
                # 如果是单个字典，转换为列表
                input_items = [input_items]
            elif not hasattr(input_items, '__iter__') or isinstance(input_items, (str, bytes)):
                logger.error(f"Embedder received non-iterable input: {type(input_items)}")
                return []

            # 收集所有有效文本和元数据
            valid_items = []
            texts = []
            metadatas = []

            for input_data in input_items:
                if isinstance(input_data, dict) and "text" in input_data:
                    valid_items.append(input_data)
                    texts.append(input_data["text"])
                    metadatas.append(input_data.get("metadata", {}))
                else:
                    logger.error(f"Embedder input data is not a valid dict: {type(input_data)}")

            # 批量生成嵌入
            results = []
            if texts:
                logger.info(f"Batch embedding {len(texts)} items")
                embeddings = generate_dense_embeddings_with_m3e(texts)

                # 组合结果
                for item, text, metadata, embedding in zip(valid_items, texts, metadatas, embeddings):
                    results.append({"text": text, "metadata": metadata, "embedding": embedding})

            logger.info(f"Successfully embedded {len(results)} items")
            return results

        # 步骤3: 存储处理
        def storage(input_items):
            """存储处理函数"""
            logger.info(f"Storage received input: {type(input_items)}")
            # 确保 input_items 是可迭代的
            if isinstance(input_items, dict):
                # 如果是单个字典，转换为列表
                input_items = [input_items]
            elif not hasattr(input_items, '__iter__') or isinstance(input_items, (str, bytes)):
                logger.error(f"Storage received non-iterable input: {type(input_items)}")
                return []

            # 收集所有数据用于批量处理
            texts = []
            embeddings = []
            metadata_list = []
            valid_items = []

            for input_data in input_items:
                if isinstance(input_data, dict) and all(k in input_data for k in ["text", "metadata", "embedding"]):
                    texts.append(input_data["text"])
                    embeddings.append(input_data["embedding"])
                    metadata_list.append(input_data["metadata"])
                    valid_items.append(input_data)
                else:
                    logger.error(f"Input data is missing required keys: {type(input_data)}")

            # 批量存储所有有效数据
            if texts:
                logger.info(f"Batch storing {len(texts)} items")
                self.milvus_storage.insert_data(
                    texts=texts,
                    embeddings=embeddings,
                    metadata_list=metadata_list,
                    batch_size=len(texts)
                )

            # 生成结果，仅用于测试
            results = []
            for item in valid_items:
                text = item["text"]
                results.append({"status": "success", "text": text[:50] + "..." if len(text) > 50 else text})

            logger.info(f"Successfully stored {len(results)} items")
            return results

        # 在存储步骤添加重试
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
        def storage_with_retry(input_items):
            """带重试的存储处理函数"""
            return storage(input_items)

        # 将函数转换为可运行对象
        json_parser_runnable = RunnableLambda(json_parser_step)
        embedder_runnable = RunnableLambda(embedder)
        storage_runnable = RunnableLambda(storage_with_retry)

        # 使用 LCEL 管道语法创建完整管道
        pipeline = json_parser_runnable | embedder_runnable | storage_runnable

        return pipeline

    def process_stream_with_lcel(self, max_items: Optional[int] = None):
        """
        使用 LCEL 执行流式处理流程
        """
        logger.info("🚀 启动Milvus RAG流式数据处理流程 (LCEL)")

        # 检查是否有检查点需要恢复
        if self.last_checkpoint:
            logger.info(f"从检查点恢复处理: last_id={self.last_checkpoint.get('last_processed_id', 0)}")
            # 合并之前的统计数据
            prev_stats = self.last_checkpoint.get('stats', {})
            self.stats.update({k: v for k, v in prev_stats.items() if k in self.stats})

        self.stats['start_time'] = time.time()

        try:
            # 创建 LCEL 管道
            pipeline = self.create_lcel_pipeline()

            # 执行流式处理
            logger.info("开始数据流式处理...")
            item_count = 0

            # 执行管道
            for batch_result in pipeline.stream({
                "json_file": self.json_file,
                "max_items": max_items
            }):
                # 每个 batch_result 是一个包含多个项目的列表
                batch_size = len(batch_result)
                item_count += batch_size

                if batch_size > 0:
                    logger.info(f"处理完成一批次，共 {batch_size} 个文本块 (累计: {item_count})")
                    # 记录第一个结果作为示例
                    if batch_result:
                        logger.info(f"批次结果示例: {batch_result[0]}")

            self.stats['generated_chunks'] = item_count
            logger.info(f"文本处理完成：共处理 {item_count} 个文本块")

        except Exception as e:
            logger.error(f"流式处理流程异常: {e}")
            self.stop_event.set()
            raise
        finally:
            self.stop_event.set()
            self._print_final_stats()

        return self.stats


def stream_process_psychology_data_milvus_lcel(json_file: str = "PsyDTCorpus_train_mulit_turn_packing.json",
                                               max_items: Optional[int] = None,
                                               auto_configure: bool = True,
                                               milvus_host: str = "localhost",
                                               milvus_port: str = "19530",
                                               collection_name: str = "psychology_dialogues",
                                               **kwargs) -> Dict:
    """
    使用 LCEL 的便捷 Milvus 流式处理函数
    """
    # 自动配置优化参数
    if auto_configure:
        system_config = get_system_recommendations()
        system_config.update(kwargs)
        kwargs = system_config
        logger.info(f"自动配置参数: {system_config}")

    processor = None
    try:
        processor = MilvusLCELStreamProcessor(
            json_file=json_file,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            collection_name=collection_name,
            **kwargs
        )
        return processor.process_stream_with_lcel(max_items)
    finally:
        if processor:
            processor.close()

# def batch_processor(inputs):
#     """批量处理函数"""
#     texts = [item["text"] for item in inputs]
#     metadatas = [item["metadata"] for item in inputs]
#
#     # 批量向量化
#     embeddings = generate_dense_embeddings_with_m3e(texts)
#
#     # 批量存储
#     self.milvus_storage.insert_data(
#         texts=texts,
#         embeddings=embeddings,
#         metadata_list=metadatas,
#         batch_size=len(texts)
#     )
#
#     return [{"status": "success", "text": text[:50] + "..." if len(text) > 50 else text}
#             for text in texts]
