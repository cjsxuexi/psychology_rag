import time
from typing import Optional, Dict

from langchain_core.runnables import Runnable, RunnableParallel, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.data_process.base_stream_processor import BaseStreamProcessor, get_system_recommendations
from src.data_process.milvus_stream_processor import MilvusStreamProcessor
from src.embedding.M3EEmbedding import generate_dense_embeddings_with_m3e


class MilvusLCELStreamProcessor(MilvusStreamProcessor):
    # 现有代码保持不变...

    def create_lcel_pipeline(self):
        """
        创建基于 Langchain LCEL 的处理管道
        """
        # 步骤1: JSON解析分块
        def json_parser(json_file, max_items):
            """JSON解析分块函数"""
            for text_chunks, chunk_metadata in self.json_handle.parallel_handle_json_data(
                json_name=json_file,
                max_items=max_items
            ):
                for text, meta in zip(text_chunks, chunk_metadata):
                    yield {"text": text, "metadata": meta}

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
            
            results = []
            for input_data in input_items:
                logger.info(f"Embedder processing input_data: {type(input_data)}")
                if isinstance(input_data, dict) and "text" in input_data:
                    text = input_data["text"]
                    metadata = input_data.get("metadata", {})

                    # 使用现有的向量化逻辑
                    embedding = generate_dense_embeddings_with_m3e([text])[0]
                    results.append({"text": text, "metadata": metadata, "embedding": embedding})
                else:
                    logger.error(f"Embedder input data is not a valid dict: {type(input_data)}")
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
            
            results = []
            for input_data in input_items:
                logger.info(f"Processing input_data: {type(input_data)}")
                if isinstance(input_data, dict) and all(k in input_data for k in ["text", "metadata", "embedding"]):
                    text = input_data["text"]
                    metadata = input_data["metadata"]
                    embedding = input_data["embedding"]

                    # 使用现有的存储逻辑
                    self.milvus_storage.insert_data(
                        texts=[text],
                        embeddings=[embedding],
                        metadata_list=[metadata],
                        batch_size=1
                    )
                    results.append({"status": "success", "text": text[:50] + "..." if len(text) > 50 else text})
                else:
                    logger.error(f"Input data is missing required keys: {type(input_data)}")
            return results

        # 在存储步骤添加重试
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
        def storage_with_retry(input_items):
            """带重试的存储处理函数"""
            return storage(input_items)



        # 创建 LCEL 管道
        def process_pipeline(params):
            """Complete pipeline processing with proper streaming"""
            # Step 1: Parse JSON
            json_items = list(json_parser(params["json_file"], params["max_items"]))
            logger.info(f"Parsed {len(json_items)} items from JSON")
            
            # Step 2: Embed items
            embedded_items = embedder(json_items)
            logger.info(f"Embedded {len(embedded_items)} items")
            
            # Step 3: Store items
            stored_items = storage_with_retry(embedded_items)
            logger.info(f"Stored {len(stored_items)} items")
            
            return stored_items

        # 创建单个可运行对象
        pipeline = RunnableLambda(process_pipeline)

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
            for result in pipeline.stream({
                "json_file": self.json_file,
                "max_items": max_items
            }):
                item_count += 1
                if item_count % 5 == 0:  # 每5条报告一次
                    logger.info(f"已处理 {item_count} 个文本块")
                    logger.info(f"处理结果: {result}")

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