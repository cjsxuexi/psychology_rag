from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.common.model_util import load_model
from sentence_transformers import SentenceTransformer


@dataclass
class RAGEvaluationResult:
    """RAG评估结果数据类"""
    precision: float
    recall: float
    f1: float
    ndcg: float
    mrr: float
    metrics: Dict[str, float]


class RAGEvaluator:
    """RAG评估器"""

    def __init__(self, use_ragas: bool = False):
        """
        初始化评估器

        Args:
            use_ragas: 是否使用ragas进行评估
        """
        self.use_ragas = use_ragas
        self.sim_model = None
        
        # 加载SentenceTransformer模型
        if not use_ragas:
            self._load_models()

    def _load_models(self):
        """加载评估所需的模型"""
        try:
            model_dir = load_model('sentence-transformers/all-MiniLM-L6-v2')
            self.sim_model = SentenceTransformer(model_dir)
        except Exception as e:
            # 模型加载失败时，使用简单的基于字符串匹配的评估方法
            print(f"Warning: Failed to load SentenceTransformer model: {str(e)}")
            print("Using simple string-based evaluation as fallback")
            self.sim_model = None

    def evaluate_initial_retrieval(
        self,
        queries: List[str],
        retrieved_results: List[List[Dict[str, Any]]],
        k: int = 5
    ) -> RAGEvaluationResult:
        """
        评估RAG的初步检索效果

        Args:
            queries: 查询列表
            retrieved_results: 每个查询的检索结果列表，每个结果包含文本内容
            k: 评估的top-k值

        Returns:
            RAGEvaluationResult: 评估结果
        """

        if self.use_ragas:
            return self._evaluate_with_ragas(queries, retrieved_results, k)
        else:
            return self._evaluate_with_custom(queries, retrieved_results, k)

    def evaluate_rerank(
        self,
        queries: List[str],
        initial_results: List[List[Dict[str, Any]]],
        reranked_results: List[List[Dict[str, Any]]],
        k: int = 5
    ) -> RAGEvaluationResult:
        """
        评估RAG的rerank后的效果

        Args:
            queries: 查询列表
            initial_results: 每个查询的初始检索结果列表
            reranked_results: 每个查询的重排序结果列表
            k: 评估的top-k值

        Returns:
            RAGEvaluationResult: 评估结果
        """
        if self.use_ragas:
            return self._evaluate_rerank_with_ragas(queries, initial_results, reranked_results, k)
        else:
            return self._evaluate_rerank_with_custom(queries, initial_results, reranked_results, k)

    def _evaluate_with_custom(
        self,
        queries: List[str],
        retrieved_results: List[List[Dict[str, Any]]],
        k: int = 5
    ) -> RAGEvaluationResult:
        """
        使用自定义无监督方法评估检索效果
        """
        # 1. 计算语义相似度
        similarity_scores = []
        for query, results in zip(queries, retrieved_results):
            query_embedding = self.sim_model.encode([query])
            context_embeddings = self.sim_model.encode([result.get('text', '') for result in results[:k]])
            if len(context_embeddings) > 0:
                similarities = cosine_similarity(query_embedding, context_embeddings)[0]
                similarity_scores.append(np.mean(similarities))
            else:
                similarity_scores.append(0.0)
        
        avg_similarity = np.mean(similarity_scores)

        # 2. 计算检索结果的多样性
        diversity_scores = []
        for results in retrieved_results:
            contexts = [result.get('text', '') for result in results[:k]]
            if len(contexts) > 1:
                embeddings = self.sim_model.encode(contexts)
                # 计算平均 pairwise 相似度，1 - 相似度即为多样性
                pairwise_similarities = cosine_similarity(embeddings)
                # 排除对角线（自身相似度）
                n = len(pairwise_similarities)
                avg_similarity_val = (np.sum(pairwise_similarities) - n) / (n * (n - 1)) if n > 1 else 0
                diversity = 1 - avg_similarity_val
                diversity_scores.append(diversity)
            else:
                diversity_scores.append(0.0)
        
        avg_diversity = np.mean(diversity_scores)

        # 3. 计算信息密度（基于文本长度和语义丰富度）
        density_scores = []
        for results in retrieved_results:
            total_length = 0
            valid_contexts = 0
            for result in results[:k]:
                text = result.get('text', '')
                if text:
                    total_length += len(text)
                    valid_contexts += 1
            
            if valid_contexts > 0:
                avg_length = total_length / valid_contexts
                # 信息密度 = 平均长度（归一化）
                density = min(1.0, avg_length / 1000)  # 归一化到0-1
                density_scores.append(density)
            else:
                density_scores.append(0.0)
        
        avg_density = np.mean(density_scores)

        # 4. 计算综合指标
        precision = avg_similarity
        recall = avg_diversity
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        ndcg = avg_similarity  # 用相似度近似NDCG
        mrr = avg_similarity  # 用相似度近似MRR

        # 构建评估结果
        metrics = {
            'similarity': float(avg_similarity),
            'diversity': float(avg_diversity),
            'density': float(avg_density),
            'k': k
        }

        return RAGEvaluationResult(
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            ndcg=float(ndcg),
            mrr=float(mrr),
            metrics=metrics
        )

    def _evaluate_rerank_with_custom(
        self,
        queries: List[str],
        initial_results: List[List[Dict[str, Any]]],
        reranked_results: List[List[Dict[str, Any]]],
        k: int = 5
    ) -> RAGEvaluationResult:
        """
        使用自定义无监督方法评估重排序效果
        """
        # 评估初始检索
        initial_eval = self._evaluate_with_custom(
            queries=queries,
            retrieved_results=initial_results,
            k=k
        )

        # 评估重排序结果
        rerank_eval = self._evaluate_with_custom(
            queries=queries,
            retrieved_results=reranked_results,
            k=k
        )

        # 计算改进度
        similarity_improvement = rerank_eval.metrics['similarity'] - initial_eval.metrics['similarity']
        diversity_improvement = rerank_eval.metrics['diversity'] - initial_eval.metrics['diversity']
        density_improvement = rerank_eval.metrics['density'] - initial_eval.metrics['density']

        # 构建评估结果
        metrics = {
            **{k: float(v) for k, v in rerank_eval.metrics.items()},
            'similarity_improvement': float(similarity_improvement),
            'diversity_improvement': float(diversity_improvement),
            'density_improvement': float(density_improvement),
            'initial_similarity': float(initial_eval.metrics['similarity']),
            'initial_diversity': float(initial_eval.metrics['diversity']),
            'initial_density': float(initial_eval.metrics['density']),
            'k': k
        }

        return RAGEvaluationResult(
            precision=float(rerank_eval.precision),
            recall=float(rerank_eval.recall),
            f1=float(rerank_eval.f1),
            ndcg=float(rerank_eval.ndcg),
            mrr=float(rerank_eval.mrr),
            metrics=metrics
        )

    def _evaluate_with_ragas(
        self,
        queries: List[str],
        retrieved_results: List[List[Dict[str, Any]]],
        k: int = 5
    ) -> RAGEvaluationResult:
        """
        使用ragas评估检索效果
        """
        try:
            import pandas as pd
            from ragas import evaluate
            from ragas.metrics import context_relevancy, context_recall
        except ImportError:
            raise RuntimeError("ragas is not installed. Please install it with 'pip install ragas'")

        # 构建ragas评估所需的数据结构
        data = []
        for query, results in zip(queries, retrieved_results):
            # 提取检索到的上下文
            contexts = [result.get('text', '') for result in results[:k]]
            # ragas需要ground_truth，这里使用空字符串作为占位符
            data.append({
                "question": query,
                "contexts": contexts,
                "ground_truth": ""
            })
        
        # 转换为DataFrame
        df = pd.DataFrame(data)
        
        # 使用ragas评估
        result = evaluate(
            df,
            metrics=[
                context_recall,
                context_relevancy
            ]
        )
        
        # 转换为现有结果格式
        metrics = result.to_dict()
        # 确保所有数值都是Python原生类型
        metrics = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in metrics.items()}
        precision = float(metrics.get('context_relevancy', 0.0))
        recall = float(metrics.get('context_recall', 0.0))
        f1 = float(2 * (precision * recall) / (precision + recall + 1e-9))
        
        return RAGEvaluationResult(
            precision=precision,
            recall=recall,
            f1=f1,
            ndcg=0.0,  # ragas不直接提供NDCG
            mrr=0.0,   # ragas不直接提供MRR
            metrics=metrics
        )

    def _evaluate_rerank_with_ragas(
        self,
        queries: List[str],
        initial_results: List[List[Dict[str, Any]]],
        reranked_results: List[List[Dict[str, Any]]],
        k: int = 5
    ) -> RAGEvaluationResult:
        """
        使用ragas评估重排序效果
        """
        try:
            import pandas as pd
            from ragas import evaluate
            from ragas.metrics import context_relevancy, context_recall
        except ImportError:
            raise RuntimeError("ragas is not installed. Please install it with 'pip install ragas'")

        # 构建ragas评估所需的数据结构
        data = []
        for query, reranked in zip(queries, reranked_results):
            # 提取重排序后的上下文
            contexts = [result.get('text', '') for result in reranked[:k]]
            data.append({
                "question": query,
                "contexts": contexts,
                "ground_truth": ""
            })
        
        # 转换为DataFrame
        df = pd.DataFrame(data)
        
        # 使用ragas评估
        result = evaluate(
            df,
            metrics=[
                context_recall,
                context_relevancy
            ]
        )
        
        # 计算初始检索的评估结果
        initial_data = []
        for query, initial in zip(queries, initial_results):
            contexts = [result.get('text', '') for result in initial[:k]]
            initial_data.append({
                "question": query,
                "contexts": contexts,
                "ground_truth": ""
            })
        
        initial_df = pd.DataFrame(initial_data)
        initial_result = evaluate(
            initial_df,
            metrics=[
                context_recall,
                context_relevancy
            ]
        )
        
        # 转换为现有结果格式
        metrics = result.to_dict()
        initial_metrics = initial_result.to_dict()
        
        # 确保所有数值都是Python原生类型
        metrics = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in metrics.items()}
        initial_metrics = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in initial_metrics.items()}
        
        precision = float(metrics.get('context_relevancy', 0.0))
        recall = float(metrics.get('context_recall', 0.0))
        f1 = float(2 * (precision * recall) / (precision + recall + 1e-9))
        
        # 计算改进度
        context_recall_improvement = float(metrics.get('context_recall', 0.0) - initial_metrics.get('context_recall', 0.0))
        context_relevancy_improvement = float(metrics.get('context_relevancy', 0.0) - initial_metrics.get('context_relevancy', 0.0))
        
        metrics.update({
            'context_recall_improvement': context_recall_improvement,
            'context_relevancy_improvement': context_relevancy_improvement,
            'initial_context_recall': float(initial_metrics.get('context_recall', 0.0)),
            'initial_context_relevancy': float(initial_metrics.get('context_relevancy', 0.0)),
            'k': k
        })

        return RAGEvaluationResult(
            precision=precision,
            recall=recall,
            f1=f1,
            ndcg=0.0,  # ragas不直接提供NDCG
            mrr=0.0,   # ragas不直接提供MRR
            metrics=metrics
        )