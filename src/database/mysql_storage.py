"""
MySQL存储模块：用于存储文本chunks及其元信息
与FAISS向量索引配合使用，实现向量-文本的映射关系
"""

import mysql.connector
from mysql.connector import Error
import json
import hashlib
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
from loguru import logger
from datetime import datetime
from src.common.file_utils import get_config_path

# ===================== 企业级初始化：加载配置 =====================
# 加载 .env 配置
load_dotenv(dotenv_path=get_config_path('.env'))
DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.getenv("MYSQL_PORT", 5455)),
    "database": os.getenv("MYSQL_DATABASE", "my_rag"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "infini_rag_flow")
}


class MySQLStorage:
    """MySQL存储管理类"""
    
    def __init__(self):
        """
        初始化MySQL连接，使用全局DB_CONFIG配置
        """
        self.host = DB_CONFIG['host']
        self.port = DB_CONFIG['port']
        self.database = DB_CONFIG['database']
        self.user = DB_CONFIG['user']
        self.password = DB_CONFIG['password']
        self.connection = None
        
    def connect(self) -> bool:
        """
        建立数据库连接
        
        Returns:
            bool: 连接是否成功
        """
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                charset='utf8mb4',
                autocommit=True
            )
            
            if self.connection.is_connected():
                logger.success(f"MySQL连接成功: {self.host}:{self.port}/{self.database}")
                # self._initialize_tables()
                return True
                
        except Error as e:
            logger.error(f"MySQL连接失败: {str(e)}")
            return False
    
    def disconnect(self):
        """关闭数据库连接"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("MySQL连接已关闭")
    
    def _initialize_tables(self):
        """初始化数据库表结构"""
        try:
            cursor = self.connection.cursor()
            
            # 创建chunks表
            create_chunks_table = """
            CREATE TABLE IF NOT EXISTS text_chunks (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                chunk_hash VARCHAR(64) UNIQUE NOT NULL COMMENT '文本块哈希值',
                chunk_text TEXT NOT NULL COMMENT '文本块内容',
                source_document VARCHAR(500) COMMENT '源文档标识',
                chunk_index INT COMMENT '在源文档中的索引',
                metadata JSON COMMENT '额外元数据',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_sourceDoc_chunkIndex (source_document,chunk_index),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='文本块存储表';
            """
            
            # 创建向量索引映射表
            create_vector_mapping_table = """
            CREATE TABLE IF NOT EXISTS vector_chunk_mapping (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                chunk_id BIGINT NOT NULL COMMENT '关联的文本块ID',
                vector_index INT NOT NULL COMMENT 'FAISS向量索引',
                faiss_index_name VARCHAR(200) NOT NULL COMMENT 'FAISS索引文件名',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uk_vector_index (faiss_index_name, vector_index),
                INDEX idx_chunk_id (chunk_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='向量-文本块映射表';
            """
            
            cursor.execute(create_chunks_table)
            cursor.execute(create_vector_mapping_table)
            cursor.close()
            
            logger.success("数据库表初始化完成")
            
        except Error as e:
            logger.error(f"数据库表初始化失败: {str(e)}")
            raise
    
    def store_chunks(self, chunks: List[str], source_document: str = None, 
                    metadata_list: List[Dict] = None) -> List[int]:
        """
        存储文本块到数据库
        
        Args:
            chunks: 文本块列表
            source_document: 源文档标识
            metadata_list: 对应的元数据列表
            
        Returns:
            List[int]: 插入的chunk IDs列表
        """
        if not self.connection or not self.connection.is_connected():
            raise Exception("数据库未连接")
        
        chunk_ids = []
        try:
            cursor = self.connection.cursor()
            
            for i, chunk in enumerate(chunks):
                # 生成文本块哈希值用于去重
                chunk_hash = hashlib.md5(chunk.encode('utf-8')).hexdigest()
                
                # 准备元数据
                metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
                metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
                
                # 插入或更新文本块
                insert_query = """
                INSERT INTO text_chunks (chunk_hash, chunk_text, source_document, chunk_index, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                chunk_text = VALUES(chunk_text),
                source_document = VALUES(source_document),
                chunk_index = VALUES(chunk_index),
                metadata = VALUES(metadata)
                """
                
                cursor.execute(insert_query, (
                    chunk_hash,
                    chunk,
                    source_document,
                    i,
                    metadata_json
                ))
                
                # 获取插入的ID
                if cursor.lastrowid:
                    chunk_ids.append(cursor.lastrowid)
                else:
                    # 如果是更新，需要查询获取ID
                    select_query = "SELECT id FROM text_chunks WHERE chunk_hash = %s"
                    cursor.execute(select_query, (chunk_hash,))
                    result = cursor.fetchone()
                    if result:
                        chunk_ids.append(result[0])
            
            cursor.close()
            logger.success(f"成功存储 {len(chunk_ids)} 个文本块")
            return chunk_ids
            
        except Error as e:
            logger.error(f"存储文本块失败: {str(e)}")
            raise
    
    def create_vector_mapping(self, chunk_ids: List[int], vector_indices: List[int], 
                             faiss_index_name: str) -> bool:
        """
        创建向量索引与文本块的映射关系
        
        Args:
            chunk_ids: 文本块IDs
            vector_indices: 对应的FAISS向量索引
            faiss_index_name: FAISS索引文件名
            
        Returns:
            bool: 映射创建是否成功
        """
        if not self.connection or not self.connection.is_connected():
            raise Exception("数据库未连接")
        
        if len(chunk_ids) != len(vector_indices):
            raise ValueError("chunk_ids和vector_indices长度必须相等")
        
        try:
            cursor = self.connection.cursor()
            
            # 批量插入映射关系
            mapping_data = []
            for chunk_id, vector_idx in zip(chunk_ids, vector_indices):
                mapping_data.append((chunk_id, vector_idx, faiss_index_name))
            
            insert_query = """
            INSERT INTO vector_chunk_mapping (chunk_id, vector_index, faiss_index_name)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE
            chunk_id = VALUES(chunk_id),
            faiss_index_name = VALUES(faiss_index_name)
            """
            
            cursor.executemany(insert_query, mapping_data)
            cursor.close()
            
            logger.success(f"成功创建 {len(mapping_data)} 个向量-文本映射关系")
            return True
            
        except Error as e:
            logger.error(f"创建向量映射失败: {str(e)}")
            raise
    
    def get_chunk_by_vector_index(self, vector_index: int, faiss_index_name: str) -> Optional[Dict]:
        """
        根据向量索引获取对应的文本块
        
        Args:
            vector_index: FAISS向量索引
            faiss_index_name: FAISS索引文件名
            
        Returns:
            Optional[Dict]: 文本块信息字典
        """
        if not self.connection or not self.connection.is_connected():
            raise Exception("数据库未连接")
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            query = """
            SELECT tc.*, vcm.vector_index
            FROM text_chunks tc
            JOIN vector_chunk_mapping vcm ON tc.id = vcm.chunk_id
            WHERE vcm.vector_index = %s AND vcm.faiss_index_name = %s
            """
            
            cursor.execute(query, (vector_index, faiss_index_name))
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                # 解析元数据
                if result.get('metadata'):
                    result['metadata'] = json.loads(result['metadata'])
                return result
            
            return None
            
        except Error as e:
            logger.error(f"查询文本块失败: {str(e)}")
            raise
    
    def get_chunks_by_vector_indices(self, vector_indices: List[int], 
                                   faiss_index_name: str) -> List[Dict]:
        """
        根据多个向量索引批量获取对应的文本块
        
        Args:
            vector_indices: FAISS向量索引列表
            faiss_index_name: FAISS索引文件名
            
        Returns:
            List[Dict]: 文本块信息列表
        """
        if not vector_indices:
            return []
            
        if not self.connection or not self.connection.is_connected():
            raise Exception("数据库未连接")
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            placeholders = ','.join(['%s'] * len(vector_indices))
            query = f"""
            SELECT tc.*, vcm.vector_index
            FROM text_chunks tc
            JOIN vector_chunk_mapping vcm ON tc.id = vcm.chunk_id
            WHERE vcm.vector_index IN ({placeholders}) 
            AND vcm.faiss_index_name = %s
            ORDER BY FIELD(vcm.vector_index, {placeholders})
            """
            
            params = vector_indices + [faiss_index_name] + vector_indices
            cursor.execute(query, params)
            
            results = cursor.fetchall()
            cursor.close()
            
            # 解析元数据
            for result in results:
                if result.get('metadata'):
                    result['metadata'] = json.loads(result['metadata'])
            
            return results
            
        except Error as e:
            logger.error(f"批量查询文本块失败: {str(e)}")
            raise
    
    def get_chunks_by_source_and_indices(self, source_document: str, 
                                       chunk_indices: Optional[List[int]] = None) -> List[Dict]:
        """
        根据源文档和chunk索引列表获取除created_at外的全部信息
        
        Args:
            source_document: 源文档标识
            chunk_indices: chunk索引列表，为None或空列表时查询该source_document的全部记录
            
        Returns:
            List[Dict]: 文本块信息列表（不包含created_at字段）
        """
        if not source_document:
            raise ValueError("source_document不能为空")
            
        if not self.connection or not self.connection.is_connected():
            self.connect()
            # raise Exception("数据库未连接")
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            # 构建查询条件
            params = [source_document]
            
            if chunk_indices and len(chunk_indices) > 0:
                # 当提供了chunk_indices时，使用IN查询
                placeholders = ','.join(['%s'] * len(chunk_indices))
                where_clause = f"source_document = %s AND chunk_index IN ({placeholders})"
                params.extend(chunk_indices)
            else:
                # 当chunk_indices为空或None时，查询该source_document的全部记录
                where_clause = "source_document = %s"
            
            # 查询语句，排除created_at字段，使用参数化查询防止SQL注入
            query = f"""
            SELECT id, chunk_hash, chunk_text, source_document, 
                   chunk_index, metadata
            FROM text_chunks 
            WHERE {where_clause}
            ORDER BY chunk_index ASC
            """
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()
            
            # 解析元数据
            for result in results:
                if result.get('metadata'):
                    try:
                        result['metadata'] = json.loads(result['metadata'])
                    except json.JSONDecodeError:
                        logger.warning(f"无法解析metadata JSON: {result['metadata']}")
                        result['metadata'] = {}
                
                # 确保返回字段的一致性和完整性
                result.update({
                    'id': result.get('id'),
                    'chunk_hash': result.get('chunk_hash'),
                    'chunk_text': result.get('chunk_text'),
                    'source_document': result.get('source_document'),
                    'chunk_index': result.get('chunk_index'),
                    'metadata': result.get('metadata', {})
                })
            
            action_desc = "全部记录" if not chunk_indices else f"{len(chunk_indices)}个指定索引"
            logger.success(f"成功查询source_document='{source_document}'的{action_desc}，共{len(results)}条记录")
            return results
            
        except Error as e:
            logger.error(f"根据源文档和索引查询文本块失败: {str(e)}")
            raise
        finally:
            self.disconnect()
    
    def get_statistics(self) -> Dict:
        """
        获取存储统计信息
        
        Returns:
            Dict: 统计信息字典
        """
        if not self.connection or not self.connection.is_connected():
            raise Exception("数据库未连接")
        
        try:
            cursor = self.connection.cursor()
            
            # 总文本块数
            cursor.execute("SELECT COUNT(*) FROM text_chunks")
            total_chunks = cursor.fetchone()[0]
            
            # 总映射关系数
            cursor.execute("SELECT COUNT(*) FROM vector_chunk_mapping")
            total_mappings = cursor.fetchone()[0]
            
            # 不同的FAISS索引数
            cursor.execute("SELECT COUNT(DISTINCT faiss_index_name) FROM vector_chunk_mapping")
            index_count = cursor.fetchone()[0]
            
            cursor.close()
            
            return {
                'total_chunks': total_chunks,
                'total_mappings': total_mappings,
                'faiss_index_count': index_count,
                'timestamp': datetime.now().isoformat()
            }
            
        except Error as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            raise


# 上下文管理器支持
class MySQLStorageManager:
    """MySQL存储管理器（支持上下文管理）"""
    
    def __init__(self, config: Dict = None):
        self.storage = MySQLStorage(config)
        self.connected = False
    
    def __enter__(self):
        self.connected = self.storage.connect()
        if not self.connected:
            raise Exception("无法建立数据库连接")
        return self.storage
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.storage.disconnect()