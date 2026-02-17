from loguru import logger

from src.data_handle.json_handle import batch_split
from src.database.mysql_storage import MySQLStorage
from src.embedding.M3EEmbedding import generate_dense_embeddings_with_m3e, reduce_dimensionality
from storage.Faiss_storage import build_faiss_index


class PsychologyApplication:
    """
    心理咨询应用程序类
    """
    def __init__(self):
        """
        初始化应用程序
        """
        self.key_storage = MySQLStorage()

    def add_data(self):
        """
        运行应用程序
        """
        # 测试时可以调小该值。max_items=10
        self.process_and_store_json_data()
        # 从key_storage读取每条 数据

    def run(self):
        source_document = "psychology_dialogues"
        datas = self.key_storage.get_chunks_by_source_and_indices(source_document)
        dense_embeddings = generate_dense_embeddings_with_m3e(datas)

        # 验证向量格式（确保与后续 PCA、FAISS 兼容）
        logger.info(f"m3e 向量验证：维度 {dense_embeddings.shape[1]}，数据类型 {dense_embeddings.dtype}")
        print(f"\n示例向量形状：{dense_embeddings.shape}")

        # 对稠密向量进行降维
        dense_embeddings_reduced, pca_model, scaler_model = reduce_dimensionality(
            dense_embeddings,
            variance_threshold=0.95  # 保留95%的语义方差，平衡效率与效果
        )

        # 构建百万级（演示10万条）FAISS索引
        faiss_index = build_faiss_index(dense_embeddings_reduced)

    def process_and_store_json_data(self, max_items=None, source_document="psychology_dialogues", batch_size=1000):
        """
        处理JSON数据并批量存储到MySQL

        Args:
            max_items: 最大处理条目数，默认None表示处理全部
            source_document: 源文档标识
            batch_size: 批量入库数量，默认1000条

        Returns:
            dict: 处理结果统计信息
        """
        logger.info(f"开始处理JSON数据并批量存储到MySQL... (批量大小: {batch_size})")

        try:
            # 1. 连接数据库
            if not self.key_storage.connect():
                raise Exception("无法连接到MySQL数据库")

            # 2. 使用JsonHandle.simple()处理数据
            logger.info("正在处理JSON数据...")
            chunks, chunk_metadata = batch_split(max_items)

            logger.info(f"✅ 处理完成，共生成 {len(chunks)} 个文本块")

            # 3. 批量存储到MySQL
            logger.info(f"正在批量存储到MySQL数据库... (每批{batch_size}条)")
            
            all_chunk_ids = []
            total_processed = 0
            
            # 分批处理
            for i in range(0, len(chunks), batch_size):
                batch_end = min(i + batch_size, len(chunks))
                batch_chunks = chunks[i:batch_end]
                batch_metadata = chunk_metadata[i:batch_end] if chunk_metadata else None

                logger.info(f"处理批次 {i//batch_size + 1}: {len(batch_chunks)} 个文本块")

                # 批量存储当前批次
                batch_chunk_ids = self.key_storage.store_chunks(
                    chunks=batch_chunks,
                    source_document=source_document,
                    metadata_list=batch_metadata
                )

                all_chunk_ids.extend(batch_chunk_ids)
                total_processed += len(batch_chunk_ids)

                logger.success(f"✅ 批次 {i//batch_size + 1} 存储完成，本次存储 {len(batch_chunk_ids)} 个文本块")

            logger.success(f"✅ 批量存储完成，总共存储 {total_processed} 个文本块到数据库")

            # 4. 返回统计信息
            stats = {
                'processed_chunks': len(chunks),
                'stored_chunk_ids': all_chunk_ids,
                'source_document': source_document,
                'batch_size': batch_size,
                'total_batches': (len(chunks) + batch_size - 1) // batch_size,
                'metadata_sample': chunk_metadata[0] if chunk_metadata else None
            }

            return stats

        except Exception as e:
            logger.error(f"处理过程中发生错误: {str(e)}")
            raise
        finally:
            # 确保关闭数据库连接
            self.key_storage.disconnect()

if __name__ == "__main__":
    app = PsychologyApplication()
    app.process_and_store_json_data()