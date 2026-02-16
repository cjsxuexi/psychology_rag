import torch
from typing import List, Dict, Any, Optional
from loguru import logger
from sentence_transformers import CrossEncoder

# 注意：如果common.model_util不存在，需确保load_model方法能正确返回模型路径
# 这里为了代码可运行性，临时定义一个mock的load_model（实际使用时替换为你的真实实现）
try:
    from common.model_util import load_model
except ImportError:
    def load_model(model_name: str) -> str:
        """模拟load_model，直接返回模型名（实际场景请替换为真实逻辑）"""
        logger.warning("使用mock的load_model，实际使用时请导入common.model_util中的真实实现")
        return model_name


class RetrievalEnhanced:
    """
    基于LlamaIndex思想的检索增强类
    实现对Milvus检索结果的重排序功能

    设计特点：
    1. 解耦设计：通过参数传递接收Milvus检索结果，不直接依赖MilvusStreamProcessor
    2. 模块化：使用model_util.py中的load_model方法加载bge-reranker模型
    3. 灵活性：支持不同的rerank策略和参数配置
    """

    def __init__(self,
                 model_name: str = "BAAI/bge-reranker-v2-m3",
                 use_fp16: bool = True,
                 batch_size: int = 32,
                 max_length: int = 512):
        """
        初始化LlamaIndex风格的检索器

        Args:
            model_name: bge-reranker模型名称
            use_fp16: 是否使用半精度推理（仅在CUDA可用时生效）
            batch_size: 批处理大小
            max_length: 最大序列长度（设置到tokenizer）
        """
        self.model_name = model_name
        self.use_fp16 = use_fp16 and torch.cuda.is_available()  # 仅CUDA可用时启用半精度
        self.batch_size = batch_size
        self.max_length = max_length
        self.reranker_model = None

        # 初始化模型
        self._load_reranker_model()

    def _load_reranker_model(self):
        """加载bge-reranker模型（适配新版sentence-transformers）"""
        try:
            logger.info(f"开始加载bge-reranker模型: {self.model_name}")

            # 使用model_util.py中的load_model方法加载模型
            model_path = load_model(self.model_name)

            # 初始化CrossEncoder（移除不支持的use_fp16参数）
            self.reranker_model = CrossEncoder(
                model_path,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

            # 设置tokenizer的最大序列长度（核心修复：将max_length配置到tokenizer）
            self.reranker_model.tokenizer.model_max_length = self.max_length
            self.reranker_model.tokenizer.padding_side = "right"  # 优化padding方式
            logger.info(f"已设置tokenizer最大序列长度为: {self.max_length}")

            # 手动设置半精度（适配新版sentence-transformers）
            if self.use_fp16 and torch.cuda.is_available():
                self.reranker_model.model = self.reranker_model.model.half()
                logger.info("已启用半精度推理（FP16）")
            else:
                logger.info("使用默认精度推理（FP32）")

            logger.success(f"bge-reranker模型加载成功: {self.model_name}")

        except Exception as e:
            logger.error(f"bge-reranker模型加载失败: {str(e)}")
            raise RuntimeError(f"Failed to load bge-reranker model: {str(e)}")

    def rerank_results(self,
                       query_text: str,
                       search_results: List[Dict[str, Any]],
                       top_k: Optional[int] = None,
                       return_scores: bool = True) -> List[Dict[str, Any]]:
        """
        对Milvus检索结果进行重排序

        Args:
            query_text: 查询文本
            search_results: MilvusStreamProcessor.search_similar_chunks的返回结果
            top_k: 返回top-k结果，如果为None则返回全部
            return_scores: 是否返回重排序分数

        Returns:
            List[Dict]: 重排序后的结果列表
        """
        try:
            if not search_results:
                logger.warning("检索结果为空，直接返回空列表")
                return []

            logger.info(f"开始对{len(search_results)}个检索结果进行重排序")

            # 准备rerank输入对
            pairs = []
            valid_indices = []  # 记录有效结果的索引，避免空内容导致的分数错位
            for idx, result in enumerate(search_results):
                content = result.get('content', '')
                if content:
                    pairs.append([query_text, content])
                    valid_indices.append(idx)
                else:
                    logger.warning(f"发现空内容的检索结果，跳过: {result}")

            if not pairs:
                logger.warning("没有有效的检索结果用于重排序")
                return []

            # 执行重排序：移除predict中不支持的max_length参数（核心修复）
            logger.info(f"使用bge-reranker对{len(pairs)}个文本对进行评分")
            scores = self.reranker_model.predict(
                pairs,
                batch_size=self.batch_size,
                convert_to_numpy=True  # 确保返回numpy数组，方便后续处理
                # 移除max_length参数，该参数已在tokenizer层面设置
            )

            # 将分数添加到原始结果中（修复分数与结果的索引对应问题）
            reranked_results = []
            for idx, score in zip(valid_indices, scores):
                original_result = search_results[idx]
                result_copy = original_result.copy()
                result_copy['rerank_score'] = float(score)
                if 'distance' in result_copy:
                    result_copy['original_distance'] = result_copy['distance']
                reranked_results.append(result_copy)

            # 按重排序分数降序排列
            reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)

            # 截取top-k结果
            if top_k is not None and top_k > 0:
                reranked_results = reranked_results[:top_k]

            logger.success(f"重排序完成，返回{len(reranked_results)}个结果")

            # 如果不需要返回分数，则移除分数字段
            if not return_scores:
                for result in reranked_results:
                    result.pop('rerank_score', None)

            return reranked_results

        except Exception as e:
            logger.error(f"重排序过程失败: {str(e)}")
            raise RuntimeError(f"Rerank failed: {str(e)}")

    def hybrid_search(self,
                      query_text: str,
                      search_results: List[Dict[str, Any]],
                      top_k: int = 5,
                      alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        混合搜索：结合原始相似度分数和重排序分数

        Args:
            query_text: 查询文本
            search_results: 检索结果
            top_k: 返回结果数量
            alpha: 混合权重 (0=完全依赖原始分数, 1=完全依赖rerank分数)

        Returns:
            List[Dict]: 混合排序后的结果
        """
        try:
            if not search_results:
                return []

            logger.info(f"执行混合搜索，alpha={alpha}")

            # 先进行重排序获取分数
            reranked_results = self.rerank_results(
                query_text=query_text,
                search_results=search_results,
                top_k=None,  # 先获取全部结果
                return_scores=True
            )

            if not reranked_results:
                logger.warning("重排序结果为空，直接返回空列表")
                return []

            # 归一化分数
            rerank_scores = [result['rerank_score'] for result in reranked_results]
            original_distances = [result.get('original_distance', result.get('distance', 0))
                                  for result in reranked_results]

            # 归一化到0-1范围
            max_rerank = max(rerank_scores) if rerank_scores else 1
            min_rerank = min(rerank_scores) if rerank_scores else 0
            max_distance = max(original_distances) if original_distances else 1
            min_distance = min(original_distances) if original_distances else 0

            # 计算混合分数
            hybrid_results = []
            for result in reranked_results:
                # 归一化rerank分数 (越高越好)
                norm_rerank = (result['rerank_score'] - min_rerank) / (max_rerank - min_rerank) \
                    if max_rerank != min_rerank else 0.5

                # 归一化距离分数 (越低越好，所以1-归一化值)
                orig_dist = result.get('original_distance', result.get('distance', 0))
                norm_distance = 1 - (orig_dist - min_distance) / (max_distance - min_distance) \
                    if max_distance != min_distance else 0.5

                # 计算混合分数
                hybrid_score = alpha * norm_rerank + (1 - alpha) * norm_distance

                result_copy = result.copy()
                result_copy['hybrid_score'] = hybrid_score
                result_copy['norm_rerank_score'] = norm_rerank
                result_copy['norm_distance_score'] = norm_distance
                hybrid_results.append(result_copy)

            # 按混合分数排序
            hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)

            # 返回top-k结果
            final_results = hybrid_results[:top_k]

            logger.success(f"混合搜索完成，返回{len(final_results)}个结果")
            return final_results

        except Exception as e:
            logger.error(f"混合搜索失败: {str(e)}")
            raise RuntimeError(f"Hybrid search failed: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            Dict: 模型配置信息
        """
        return {
            'model_name': self.model_name,
            'use_fp16': self.use_fp16,
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'device': "cuda" if torch.cuda.is_available() else "cpu",
            'precision': "FP16" if self.use_fp16 else "FP32",
            'tokenizer_max_length': self.reranker_model.tokenizer.model_max_length if self.reranker_model else self.max_length
        }

    def __str__(self) -> str:
        """字符串表示"""
        info = self.get_model_info()
        return f"RetrievalEnhanced(model={info['model_name']}, device={info['device']}, precision={info['precision']}, max_length={info['max_length']})"

    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()


if __name__ == '__main__':
    # 测试代码：初始化并简单验证模型加载
    reranker = RetrievalEnhanced()
    print(reranker)

    # 测试重排序功能
    test_query = "什么是大语言模型？"
    test_results = [
        {"content": "大语言模型是基于Transformer架构的大型语言模型，能够理解和生成人类语言。", "distance": 0.1},
        {"content": "机器学习是人工智能的一个分支，专注于数据驱动的模型训练。", "distance": 0.8},
        {"content": "", "distance": 0.5},  # 空内容测试
        {"content": "GPT-4是一种先进的大语言模型，由OpenAI开发。", "distance": 0.2}
    ]

    # 执行重排序
    reranked = reranker.rerank_results(test_query, test_results, top_k=2)
    print("\n重排序结果：")
    for res in reranked:
        print(f"内容: {res['content'][:50]} | 重排序分数: {res['rerank_score']:.4f}")

    # 执行混合搜索
    hybrid = reranker.hybrid_search(test_query, test_results, top_k=2, alpha=0.7)
    print("\n混合搜索结果：")
    for res in hybrid:
        print(f"内容: {res['content'][:50]} | 混合分数: {res['hybrid_score']:.4f}")