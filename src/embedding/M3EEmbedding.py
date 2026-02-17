import numpy as np
import os

import torch
from dotenv import load_dotenv
from loguru import logger
from sentence_transformers import SentenceTransformer
from src.common import load_model
from src.common.file_utils import get_config_path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import threading
# ===================== 企业级初始化：加载配置、日志、单例模型 =====================
# 加载 .env 配置
load_dotenv(dotenv_path=get_config_path('.env'))
CONFIG = {
    "m3e_model_name": os.getenv("M3E_MODEL_NAME"),
    "m3e_batch_size": int(os.getenv("M3E_BATCH_SIZE")),
    "embedding_dim": int(os.getenv("EMBEDDING_DIM")),
    "text_chunk_size": int(os.getenv("TEXT_CHUNK_SIZE")),
    "text_chunk_overlap": int(os.getenv("TEXT_CHUNK_OVERLAP")),
    "min_text_length": int(os.getenv("MIN_TEXT_LENGTH")),
    "faiss_index_path": os.getenv("FAISS_INDEX_SAVE_PATH"),
    "pca_variance": float(os.getenv("PCA_VARIANCE_THRESHOLD"))
}


# 单例模式加载 m3e-base 模型（避免重复加载，节省内存）
class M3EEmbeddingSingleton:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # 双重检查锁定模式
                if cls._instance is None:
                    cls._instance = super(M3EEmbeddingSingleton, cls).__new__(cls)
                    try:
                        cls._load_model()
                        cls._initialized = True
                    except Exception as e:
                        # 清理失败的实例
                        cls._instance = None
                        cls._initialized = False
                        raise RuntimeError(f"Failed to initialize M3E embedding model: {str(e)}")
        return cls._instance

    @classmethod
    def _load_model(cls):
        """加载 m3e-base 模型，自动适配 GPU/CPU"""
        try:
            logger.info(f"开始加载 m3e 模型：{CONFIG['m3e_model_name']}")
            # 自动检测 GPU（有 GPU 自动使用，无 GPU 降级到 CPU）
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._model = SentenceTransformer(
                load_model(CONFIG["m3e_model_name"]),
                device=device
            )
            logger.success(f"m3e 模型加载成功，运行设备：{device}，向量维度：{CONFIG['embedding_dim']}")
        except Exception as e:
            logger.error(f"m3e 模型加载失败：{str(e)}")
            raise Exception(f"模型加载异常：{e}")

    def generate_embeddings(self, texts, batch_size=None):
        """生成稠密向量，企业级批量处理，带容错"""
        if batch_size is None:
            batch_size = CONFIG["m3e_batch_size"]

        if not texts:
            logger.warning("无有效文本数据，无法生成向量")
            return np.array([], dtype=np.float32)

        try:
            logger.info(f"开始生成向量，文本总数：{len(texts)}，批次大小：{batch_size}")
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # 向量归一化，提升检索效果
            )
            # 转换数据类型以兼容 FAISS（SentenceTransformer.encode 不接受 dtype 参数）
            embeddings = embeddings.astype(np.float32)
            logger.success(f"向量生成完成，向量形状：{embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"向量生成失败：{str(e)}")
            raise Exception(f"向量生成异常：{e}")


# 初始化 m3e 单例模型
m3e_embedding = M3EEmbeddingSingleton()


# ===================== 步骤 2：生成 m3e-base 稠密向量 =====================
def generate_dense_embeddings_with_m3e(texts):
    """
    企业级向量生成：调用 m3e 单例模型，批量生成，格式兼容后续流程
    """
    return m3e_embedding.generate_embeddings(texts)


def reduce_dimensionality(embeddings, target_dim=256, variance_threshold=0.95):
    """
    PCA降维：自动选择保留指定方差的维度，或固定目标维度
    """
    # 1. 数据标准化（PCA对数据尺度敏感，必须标准化）
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # 2. 初始化PCA
    if variance_threshold is not None:
        pca = PCA(n_components=variance_threshold)
    else:
        pca = PCA(n_components=target_dim)

    # 3. 训练PCA并降维
    print(f"开始降维：原始维度{embeddings.shape[1]} → 目标（保留{variance_threshold * 100}%方差）")
    embeddings_reduced = pca.fit_transform(embeddings_scaled)

    # 4. 输出实际降维后的维度
    actual_dim = embeddings_reduced.shape[1]
    print(f"降维完成：实际维度{actual_dim}，保留方差{pca.explained_variance_ratio_.sum():.4f}")

    return embeddings_reduced, pca, scaler

# ===================== 主执行流程 =====================
import logging
import traceback
import src.data_handle.json_handle as rag  # 将导入语句移至顶部


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    try:
        # 步骤 1：获取有效文本块
        valid_text_chunks = rag.batch_split()

        # 步骤 2：生成 m3e-base 稠密向量（替换原 OpenAI 向量生成）
        # 演示：取前 1000 条文本块（百万级可直接使用 valid_text_chunks）
        demo_text_chunks = valid_text_chunks[:1000]
        dense_embeddings = generate_dense_embeddings_with_m3e(demo_text_chunks)

        # 验证向量格式（确保与后续 PCA、FAISS 兼容）
        logger.info(f"m3e 向量验证：维度 {dense_embeddings.shape[1]}，数据类型 {dense_embeddings.dtype}")
        print(f"\n示例文本块：\n{demo_text_chunks[0][:200]}...")
        print(f"\n示例向量形状：{dense_embeddings.shape}")

        # 对稠密向量进行降维
        dense_embeddings_reduced, pca_model, scaler_model = reduce_dimensionality(
            dense_embeddings,
            variance_threshold=0.95  # 保留95%的语义方差，平衡效率与效果
        )

    except ImportError as e:
        logger.critical(f"模块导入失败：{str(e)}")
        logger.debug(traceback.format_exc())
        exit(1)
    except RuntimeError as e:
        logger.critical(f"运行时错误：{str(e)}")
        logger.debug(traceback.format_exc())
        exit(1)
    except Exception as e:
        logger.critical(f"未知错误：{str(e)}")
        logger.debug(traceback.format_exc())
        exit(1)
