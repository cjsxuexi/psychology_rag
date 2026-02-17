import os
from typing import List, Dict, Optional, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from loguru import logger

from src.common.file_utils import get_config_path, get_storage_path
from src.database.mysql_storage import MySQLStorage


def build_faiss_index(embeddings, 
                     index_type="hnsw_pq", 
                     save_path="faiss_million_index.index",
                     text_chunks: List[str] = None,
                     source_document: str = None,
                     metadata_list: List[Dict] = None) -> Tuple[faiss.Index, Optional[List[int]]]:
    """
    构建FAISS索引（增强版）：
    - hnsw_pq：HNSW（ANN）+ PQ（乘积量化），适合百万级数据
    - 支持保存索引，后续可直接加载使用
    - 可选：同步存储文本块到MySQL数据库
    
    Args:
        embeddings: 向量数组
        index_type: 索引类型
        save_path: 索引保存路径
        text_chunks: 对应的文本块列表（可选）
        source_document: 源文档标识（可选）
        metadata_list: 文本块元数据列表（可选）
        
    Returns:
        Tuple[faiss.Index, Optional[List[int]]]: (FAISS索引对象, 存储的chunk IDs)
    """
    dim = embeddings.shape[1]
    index = None
    chunk_ids = None
    
    # 如果提供了文本块和MySQL配置，则存储到数据库
    if text_chunks and len(embeddings) == len(text_chunks):
        try:
            logger.info("检测到文本块和MySQL配置，开始同步存储...")
            mysql_storage = MySQLStorage()
            if mysql_storage.connect():
                # 存储文本块
                chunk_ids = mysql_storage.store_chunks(
                    chunks=text_chunks,
                    source_document=source_document,
                    metadata_list=metadata_list
                )
                logger.success(f"成功存储 {len(chunk_ids)} 个文本块到MySQL")
                mysql_storage.disconnect()
            else:
                logger.warning("MySQL连接失败，跳过文本块存储")
        except Exception as e:
            logger.error(f"存储文本块到MySQL失败: {str(e)}")

    if index_type == "hnsw_pq":
        # 1. 配置HNSW参数（平衡检索速度与召回率）
        hnsw_m = 32  # 每个节点的邻居数，越大召回率越高，速度越慢
        hnsw_ef_construction = 200  # 构建索引时的探索范围，越大索引质量越高，构建越慢

        # 2. 动态计算有效的PQ参数（确保维度兼容性）
        # 获取有效的子量化器数量M（满足 dim % M == 0）
        valid_ms = [m for m in [32, 16, 8, 4, 2, 1] if dim % m == 0 and m > 0]
        if not valid_ms:
            # 如果没有合适的M值，选择能整除dim的最大正整数
            valid_ms = [m for m in range(1, dim + 1) if dim % m == 0]
        
        pq_m = max(valid_ms)  # 选择最大的M值以获得更好的压缩效果
        pq_nbits = 8  # 每个子向量量化为8比特
        
        logger.info(f"PQ参数计算完成：维度={dim}, M={pq_m}, 满足条件: {dim}%{pq_m}={dim % pq_m}")

        # 3. 初始化索引：HNSW + PQ
        quantizer = faiss.IndexHNSWFlat(dim, hnsw_m)
        quantizer.hnsw.efConstruction = hnsw_ef_construction
        index = faiss.IndexPQ(dim, pq_m, pq_nbits, faiss.METRIC_L2)
        index.train(len(embeddings), embeddings)  # PQ需要先训练量化器
        index.add(embeddings)  # 添加降维后的向量

        # 4. 关联HNSW量化器，提升检索速度
        index = faiss.IndexIVFFlat(quantizer, dim, 100, faiss.METRIC_L2)
        index.train(len(embeddings), embeddings)
        index.add(embeddings)

    elif index_type == "flat":
        # 暴力检索索引（对比用，百万级数据速度极慢，不推荐）
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

    # 5. 保存索引到本地
    path = get_storage_path(save_path.lstrip('/'))  # 移除开头的斜杠
    
    # 确保目录存在
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    faiss.write_index(index, path)
    print(f"FAISS索引构建完成，已保存至{path}")
    
    # 如果存储了文本块，创建向量映射关系
    if chunk_ids:
        try:
            mysql_storage = MySQLStorage()
            if mysql_storage.connect():
                vector_indices = list(range(len(chunk_ids)))
                mysql_storage.create_vector_mapping(
                    chunk_ids=chunk_ids,
                    vector_indices=vector_indices,
                    faiss_index_name=save_path
                )
                logger.success("向量-文本块映射关系创建完成")
                mysql_storage.disconnect()
        except Exception as e:
            logger.error(f"创建向量映射关系失败: {str(e)}")

    return index, chunk_ids


# ===================== 加载配置（企业级解耦） =====================
load_dotenv(dotenv_path=get_config_path('.env'))
FAISS_CONFIG = {
    "pq_m_candidates": list(map(int, os.getenv("PQ_M_CANDIDATES").split(","))),
    "pq_nbits": int(os.getenv("PQ_NBITS")),
    "quantize_type": os.getenv("FAISS_QUANTIZE_TYPE").upper(),
    "hnsw_m": 32,
    "hnsw_ef_construction": 200
}


# ===================== 动态获取有效 PQ M 值（保留核心逻辑） =====================
def get_valid_pq_m(dim: int, candidates: List[int]) -> int:
    """
    企业级：动态获取有效的PQ子量化器数量M（满足 d % M == 0，适配 FAISS 1.13.2）
    优先选择候选值中较大的M（提升压缩比，减少存储）
    """
    try:
        # 校验输入参数（兼容 NumPy 2.4.1 数值类型）
        dim_int = int(dim)
        if dim_int <= 0:
            raise ValueError(f"向量维度无效：{dim}（转换后{dim_int}），必须为正整数")
        if not candidates or not isinstance(candidates, list):
            raise ValueError("PQ子量化器候选值不能为空，且必须为列表类型")

        # 筛选满足 d % M == 0 的候选值
        valid_ms = [m for m in candidates if dim_int % m == 0 and m > 0]

        if not valid_ms:
            # 若无匹配候选值，选择最大的能整除dim的正整数（兜底方案）
            logger.warning(f"无匹配的PQ候选值，自动选择能整除维度{dim_int}的最大子量化器数量")
            valid_ms = [m for m in range(1, dim_int + 1) if dim_int % m == 0]

        # 优先选择最大的M（提升压缩效率，企业级最佳实践）
        best_m = max(valid_ms)
        logger.success(f"获取有效PQ子量化器数量：M={best_m}，向量维度d={dim_int}，满足d%M={dim_int % best_m}")
        return best_m
    except Exception as e:
        logger.error(f"获取有效PQ子量化器数量失败：{str(e)}")
        raise Exception(f"PQ参数计算异常：{e}")


# ===================== 企业级 FAISS 索引构建（适配 1.13.2 纯位置参数） =====================
def build_faiss_index_enterprise(embeddings: np.ndarray,
                                 index_save_path: str = "faiss_million_m3e.index") -> faiss.Index:
    """
    企业级 FAISS 索引构建（适配 FAISS 1.13.2 + NumPy 2.4.1，纯位置参数传递）
    修复：IndexPQ 无关键字参数，严格按固定位置传递
    """
    try:
        # 1. NumPy 2.4.1 数组兼容性处理（核心：float32 + C 连续 + 二维数组）
        if embeddings.ndim != 2:
            raise ValueError(f"向量数组必须为二维数组，当前维度：{embeddings.ndim}")
        # 转换为 float32（FAISS 1.13.2 强制要求）
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
            logger.warning("向量格式转换为 np.float32（FAISS 1.13.2 要求）")
        # 确保 C 连续存储（NumPy 2.4.1 优化，提升 FAISS 处理速度）
        if not embeddings.flags.c_contiguous:
            embeddings = np.ascontiguousarray(embeddings)
            logger.warning("向量数组转换为 C 连续存储（NumPy 2.4.1 优化）")

        # 提取向量维度和数量
        dim_int = embeddings.shape[1]
        embeddings_count = len(embeddings)
        if dim_int <= 0 or embeddings_count <= 0:
            raise ValueError("向量数据无效（维度/数量≤0），无法构建 FAISS 索引")

        logger.info(
            f"开始构建企业级 FAISS 索引（1.13.2），向量维度：{dim_int}，向量数量：{embeddings_count}，量化类型：{FAISS_CONFIG['quantize_type']}")
        index = None

        # 2. 构建索引（根据量化类型选择逻辑，核心修复：IndexPQ 纯位置参数）
        if FAISS_CONFIG["quantize_type"] == "PQ":
            # 步骤1：动态获取有效的 PQ M 值
            pq_m = get_valid_pq_m(
                dim=dim_int,
                candidates=FAISS_CONFIG["pq_m_candidates"]
            )
            pq_nbits = FAISS_CONFIG["pq_nbits"]
            metric = faiss.METRIC_L2  # 距离度量（固定为 L2，与位置4对应）

            # 步骤2：初始化 HNSW 量化器（FAISS 1.13.2，纯位置参数）
            quantizer = faiss.IndexHNSWFlat(
                dim_int,  # 位置1：向量维度（纯位置，无关键字）
                FAISS_CONFIG["hnsw_m"],  # 位置2：HNSW 邻居数（纯位置，无关键字）
                metric  # 位置3：距离度量（纯位置，无关键字）
            )
            quantizer.hnsw.efConstruction = FAISS_CONFIG["hnsw_ef_construction"]

            # 步骤3：核心修复：IndexPQ 初始化（FAISS 1.13.2 纯位置参数，无任何关键字）
            # 位置顺序严格遵循：dim → M → nbits → metric（不可调整，缺一不可）
            logger.info(f"开始初始化 PQ 索引（1.13.2 纯位置参数），维度：{dim_int}，M={pq_m}，nbits={pq_nbits}，度量：L2")
            index = faiss.IndexPQ(
                dim_int,  # 位置1：向量维度
                pq_m,  # 位置2：子量化器数量 M（核心：移除 M= 关键字）
                pq_nbits,  # 位置3：量化比特数 nbits（核心：移除 nbits= 关键字）
                metric  # 位置4：距离度量（核心：移除 metric= 关键字，可选但建议显式传递）
            )

            # 步骤4：训练 PQ 量化器并添加向量（FAISS 1.13.2 要求训练数据量 ≥ 1000，否则警告）
            if embeddings_count < 1000:
                logger.warning(f"PQ 训练数据量不足（{embeddings_count} < 1000），可能影响量化精度，建议补充数据")
            logger.info(f"开始训练 PQ 量化器（M={pq_m}，nbits={pq_nbits}）")
            index.train(len(embeddings), embeddings)
            index.add(embeddings)
            logger.success("PQ 量化器训练完成，向量已添加至索引（FAISS 1.13.2）")

        elif FAISS_CONFIG["quantize_type"] == "SQ":
            # 兜底：标量量化（SQ，FAISS 1.13.2 纯位置参数）
            logger.info("使用标量量化（SQ）构建索引（1.13.2），无向量维度均分约束")
            index = faiss.IndexScalarQuantizer(
                dim_int,  # 位置1：向量维度（纯位置）
                faiss.ScalarQuantizer.QT_8bit,  # 位置2：量化类型（纯位置）
                faiss.METRIC_L2  # 位置3：距离度量（纯位置）
            )
            index.train(len(embeddings), embeddings)
            index.add(embeddings)
            logger.success("SQ 标量量化索引构建完成（FAISS 1.13.2）")

        else:
            # 兜底：无量化索引（IndexHNSWFlat，FAISS 1.13.2 纯位置参数）
            logger.warning("未知量化类型，使用无量化 IndexHNSWFlat 索引（1.13.2）")
            index = faiss.IndexHNSWFlat(
                dim_int,  # 位置1：向量维度（纯位置）
                FAISS_CONFIG["hnsw_m"],  # 位置2：HNSW 邻居数（纯位置）
                faiss.METRIC_L2  # 位置3：距离度量（纯位置）
            )
            index.hnsw.efConstruction = FAISS_CONFIG["hnsw_ef_construction"]
            index.add(embeddings)
            logger.success("无量化 HNSW 索引构建完成（FAISS 1.13.2）")

        # 3. 保存索引（FAISS 1.13.2 支持持久化，兼容 NumPy 2.4.1）
        # 使用 get_storage_path 确保路径正确且目录存在
        path = get_storage_path(index_save_path.lstrip('/'))  # 移除开头的斜杠
        
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        faiss.write_index(index, path)
        logger.success(f"FAISS 索引构建完成（1.13.2），已保存至：{path}，索引维度：{index.d}")
        return index

    except Exception as e:
        logger.error(f"构建 FAISS 索引失败（1.13.2）：{str(e)}")
        raise Exception(f"FAISS 索引构建异常：{e}")


# ===================== 测试修复效果（企业级闭环验证，适配目标版本） =====================
# if __name__ == "__main__":
#     try:
#         # 模拟 m3e-base 降维后的向量（维度 256，满足 32 的整数倍，兼容 NumPy 2.4.1）
#         demo_dim = 256
#         demo_embeddings = np.random.rand(1000, demo_dim).astype(np.float32)
#         # 确保 C 连续存储（NumPy 2.4.1 优化）
#         demo_embeddings = np.ascontiguousarray(demo_embeddings)
#
#         # 构建修复后的 FAISS 索引（适配 1.13.2 + 2.4.1）
#         faiss_index = build_faiss_index_enterprise(
#             embeddings=demo_embeddings,
#             index_save_path="/faiss_fixed_pq_1.13.2.index"
#         )
#
#         # 验证索引有效性（FAISS 1.13.2 接口）
#         logger.info(f"索引验证成功（1.13.2），索引维度：{faiss_index.d}，包含向量数：{faiss_index.ntotal}")
#     except Exception as e:
#         logger.critical(f"测试 FAISS 索引修复失败（1.13.2）：{str(e)}")
#         exit(1)
