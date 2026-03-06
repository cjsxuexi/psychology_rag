#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Milvus向量数据库存储管理模块
提供Milvus连接、集合管理、数据插入和搜索等核心操作
"""

import time
from typing import List, Dict
import numpy as np

from loguru import logger
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType,
    utility, db
)

# 延迟导入，在函数中使用时再导入
# from src.embedding.M3EEmbedding import generate_dense_embeddings_with_m3e


class MilvusStorage:
    """
    Milvus存储管理类
    提供完整的Milvus向量数据库操作接口
    """

    def __init__(self,
                 host: str = "localhost",
                 port: str = "19530",
                 collection_name: str = "psychology_dialogues",
                 database: str = "default",
                 user: str = "",
                 password: str = ""):
        """
        初始化Milvus存储
        
        Args:
            host: Milvus服务主机地址
            port: Milvus服务端口
            collection_name: 集合名称
            database: 数据库名称
            user: Milvus认证用户名
            password: Milvus认证密码
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.database = database
        self.user = user
        self.password = password
        self.collection = None
        self.connected = False

        # 初始化连接
        self._connect()

    def _connect(self):
        """建立Milvus连接"""
        try:
            logger.info(f"正在连接Milvus: {self.host}:{self.port}")

            # 连接Milvus服务
            connect_params = {
                "alias": "default",
                "host": self.host,
                "port": self.port
            }
            
            # 如果提供了用户名和密码，则添加认证信息
            if self.user and self.password:
                connect_params["user"] = self.user
                connect_params["password"] = self.password
            
            connections.connect(**connect_params)

            # 尝试使用指定数据库（如果支持）
            if self.database and self.database != "default":
                try:
                    # 尝试切换到指定数据库
                    db.using_database(self.database)
                    logger.info(f"成功切换到数据库: {self.database}")
                except Exception as db_error:
                    # 如果切换失败，尝试创建数据库
                    try:
                        logger.info(f"数据库 {self.database} 不存在，尝试创建...")
                        db.create_database(self.database)
                        logger.success(f"数据库 {self.database} 创建成功")
                        # 再次尝试切换到新创建的数据库
                        db.using_database(self.database)
                        logger.info(f"成功切换到数据库: {self.database}")
                    except Exception as create_error:
                        # 如果创建也失败，使用默认数据库
                        logger.warning(f"数据库创建失败: {str(create_error)}，使用默认数据库")
                        # 继续使用默认数据库

            self.connected = True
            logger.success("Milvus连接成功")

            # 检查并初始化集合
            self._initialize_collection()

        except Exception as e:
            logger.error(f"Milvus连接失败: {str(e)}")
            raise

    def _initialize_collection(self):
        """初始化集合：检查存在性，不存在则创建"""
        try:
            if utility.has_collection(self.collection_name):
                logger.info(f"集合 {self.collection_name} 已存在，加载集合...")
                self.collection = Collection(self.collection_name)
                self.collection.load()
            else:
                logger.info(f"创建新集合: {self.collection_name}")
                self._create_collection()

        except Exception as e:
            logger.error(f"集合初始化失败: {str(e)}")
            raise

    def _create_collection(self):
        """创建Milvus集合"""
        try:
            # 定义字段结构
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
            self.collection = Collection(
                name=self.collection_name,
                schema=schema
            )

            # 创建索引
            index_params = {
                "index_type": "HNSW",
                "metric_type": "IP",  # 内积相似度（适合归一化向量）
                "params": {"M": 32, "efConstruction": 200}
            }

            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )

            # 加载集合
            self.collection.load()

            logger.success(f"集合 {self.collection_name} 创建并加载成功")

        except Exception as e:
            logger.error(f"创建Milvus集合失败: {str(e)}")
            raise

    def get_max_file_id(self) -> int:
        """
        获取当前集合中的最大file_id
        
        Returns:
            int: 最大file_id，如果集合为空则返回0
        """
        try:
            if not self.connected or not self.collection:
                return 0

            # 查询最大file_id
            expr = "file_id >= 0"  # 查询所有记录
            res = self.collection.query(
                expr=expr,
                output_fields=["file_id"],
                limit=1,
                offset=0,
                consistency_level="Strong"
            )

            if res:
                # 如果有数据，获取最大file_id
                max_id = max([item["file_id"] for item in res])
                return max_id
            else:
                return 0

        except Exception as e:
            logger.warning(f"获取最大file_id失败: {str(e)}，使用默认值0")
            return 0

    def insert_data(self,
                    texts: List[str],
                    embeddings: List[np.ndarray],
                    metadata_list: List[Dict],
                    batch_size: int = 100) -> int:
        """
        批量插入数据到Milvus
        
        Args:
            texts: 文本内容列表
            embeddings: 向量列表
            metadata_list: 元数据列表
            batch_size: 批次大小
            
        Returns:
            int: 成功插入的记录数
        """
        try:
            if not self.connected or not self.collection:
                raise Exception("Milvus未连接或集合未初始化")

            # 获取当前最大的file_id
            current_file_id = self.get_max_file_id() + 1
            logger.info(f"当前文件ID起始值: {current_file_id}")

            total_inserted = 0
            batch_count = 0

            # 分批处理数据
            for i in range(0, len(texts), batch_size):
                batch_end = min(i + batch_size, len(texts))
                batch_texts = texts[i:batch_end]
                batch_embeddings = embeddings[i:batch_end]
                batch_metadata = metadata_list[i:batch_end]

                try:
                    # 准备批量插入数据
                    file_ids = []
                    contents = []
                    tags = []
                    total_turns_list = []

                    for j, (text, meta) in enumerate(zip(batch_texts, batch_metadata)):
                        file_ids.append(int(current_file_id))
                        contents.append(str(text))
                        tags.append(str(meta.get('tag', ''))[:50])  # 限制标签长度
                        total_turns_list.append(int(meta.get('total_turns', 0)))

                    # 转换嵌入向量为列表格式
                    embeddings_list = [emb.tolist() for emb in batch_embeddings]

                    # 执行批量插入
                    insert_result = self.collection.insert([
                        file_ids,  # file_id字段
                        embeddings_list,  # embedding字段
                        contents,  # content字段
                        tags,  # tag字段
                        total_turns_list  # total_turns字段
                    ])

                    inserted_count = len(batch_texts)
                    total_inserted += inserted_count
                    batch_count += 1

                    logger.debug(f"批次 {batch_count} 插入完成: {inserted_count} 条记录")

                except Exception as e:
                    logger.error(f"批次 {batch_count + 1} 插入失败: {str(e)}")
                    raise

            # 刷新数据
            self.collection.flush()
            logger.success(f"数据插入完成，共插入 {total_inserted} 条记录")

            return total_inserted

        except Exception as e:
            logger.error(f"数据插入失败: {str(e)}")
            raise

    def search_similar(self,
                       query_text: str,
                       top_k: int = 5,
                       output_fields: List[str] = None) -> List[Dict]:
        """
        在Milvus中搜索相似的文本块
        
        Args:
            query_text: 查询文本
            top_k: 返回top-k结果
            output_fields: 输出字段列表，默认为["content", "tag", "total_turns"]
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        try:
            if not self.connected or not self.collection:
                raise Exception("Milvus未连接或集合未初始化")

            if output_fields is None:
                output_fields = ["content", "tag", "total_turns"]

            # 生成查询向量
            from src.embedding.M3EEmbedding import generate_dense_embeddings_with_m3e
            query_embedding = generate_dense_embeddings_with_m3e([query_text])[0]

            # 执行搜索
            search_params = {
                "metric_type": "IP",
                "params": {"ef": 64}
            }

            results = self.collection.search(
                data=[query_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=output_fields
            )

            # 处理搜索结果
            similar_chunks = []
            for hits in results:
                for hit in hits:
                    chunk_info = {
                        'distance': hit.distance,
                        'chunk_id': hit.id
                    }

                    # 从实体中提取各字段信息
                    for field in output_fields:
                        chunk_info[field] = hit.entity.get(field, '')

                    similar_chunks.append(chunk_info)

            return similar_chunks

        except Exception as e:
            logger.error(f"Milvus搜索失败: {str(e)}")
            raise

    def get_collection_stats(self) -> Dict:
        """
        获取集合统计信息
        
        Returns:
            Dict: 统计信息字典
        """
        try:
            if not self.connected or not self.collection:
                return {}

            stats = {
                'collection_name': self.collection_name,
                'entity_count': self.collection.num_entities,
                'connected': self.connected,
                'timestamp': time.time()
            }

            return stats

        except Exception as e:
            logger.error(f"获取集合统计信息失败: {str(e)}")
            return {}

    def release_collection(self):
        """释放集合资源"""
        try:
            if self.collection:
                self.collection.release()
                logger.info(f"集合 {self.collection_name} 已释放")
        except Exception as e:
            logger.error(f"释放集合失败: {str(e)}")

    def close(self):
        """关闭Milvus连接"""
        try:
            self.release_collection()
            connections.disconnect("default")
            self.connected = False
            logger.info("Milvus连接已关闭")
        except Exception as e:
            logger.error(f"关闭Milvus连接失败: {str(e)}")


# 便捷使用的包装函数
def create_milvus_storage(host: str = "localhost",
                          port: str = "19530",
                          collection_name: str = "psychology_dialogues",
                          database: str = "default",
                          user: str = "root",
                          password: str = "Milvus") -> MilvusStorage:
    """
    创建Milvus存储实例的便捷函数
    
    Args:
        host: Milvus服务主机
        port: Milvus服务端口
        collection_name: 集合名称
        database: 数据库名称
        user: Milvus认证用户名
        password: Milvus认证密码
        
    Returns:
        MilvusStorage: Milvus存储实例
    """
    return MilvusStorage(
        host=host,
        port=port,
        collection_name=collection_name,
        database=database,
        user=user,
        password=password
    )
