from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Any

from src.retrieval_augment.eval import RAGEvaluator, RAGEvaluationResult

# 创建FastAPI应用
app = FastAPI(
    title="RAG评估接口",
    description="提供RAG系统的评估功能，支持自定义无监督评估和ragas评估",
    version="1.0.0"
)

# 评估器实例
rag_evaluator = None

# 请求模型
class InitialRetrievalRequest(BaseModel):
    """初步检索评估请求模型"""
    queries: List[str]
    retrieved_results: List[List[Dict[str, Any]]]
    k: int = 5
    use_ragas: bool = False

class RerankRequest(BaseModel):
    """重排序评估请求模型"""
    queries: List[str]
    initial_results: List[List[Dict[str, Any]]]
    reranked_results: List[List[Dict[str, Any]]]
    k: int = 5
    use_ragas: bool = False

# 响应模型
class EvaluationResponse(BaseModel):
    """评估响应模型"""
    precision: float
    recall: float
    f1: float
    ndcg: float
    mrr: float
    metrics: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化评估器"""
    global rag_evaluator
    # 延迟初始化，避免启动时加载模型
    pass

@app.post("/evaluate/initial", response_model=EvaluationResponse)
async def evaluate_initial_retrieval(
    request: InitialRetrievalRequest = Body(...)
):
    """
    评估RAG的初步检索效果
    
    Args:
        request: 评估请求参数
        
    Returns:
        评估结果
    """
    try:
        # 初始化评估器
        evaluator = RAGEvaluator(use_ragas=request.use_ragas)
        
        # 执行评估
        result = evaluator.evaluate_initial_retrieval(
            queries=request.queries,
            retrieved_results=request.retrieved_results,
            k=request.k
        )
        
        # 转换为响应模型
        return EvaluationResponse(
            precision=result.precision,
            recall=result.recall,
            f1=result.f1,
            ndcg=result.ndcg,
            mrr=result.mrr,
            metrics=result.metrics
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"评估失败: {str(e)}")

@app.post("/evaluate/rerank", response_model=EvaluationResponse)
async def evaluate_rerank(
    request: RerankRequest = Body(...)
):
    """
    评估RAG的重排序效果
    
    Args:
        request: 评估请求参数
        
    Returns:
        评估结果
    """
    try:
        # 初始化评估器
        evaluator = RAGEvaluator(use_ragas=request.use_ragas)
        
        # 执行评估
        result = evaluator.evaluate_rerank(
            queries=request.queries,
            initial_results=request.initial_results,
            reranked_results=request.reranked_results,
            k=request.k
        )
        
        # 转换为响应模型
        return EvaluationResponse(
            precision=result.precision,
            recall=result.recall,
            f1=result.f1,
            ndcg=result.ndcg,
            mrr=result.mrr,
            metrics=result.metrics
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"评估失败: {str(e)}")

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "RAG评估接口",
        "endpoints": {
            "evaluate_initial": "/evaluate/initial",
            "evaluate_rerank": "/evaluate/rerank"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Any

from src.retrieval_augment.eval import RAGEvaluator, RAGEvaluationResult

# 创建FastAPI应用
app = FastAPI(
    title="RAG评估接口",
    description="提供RAG系统的评估功能，支持自定义无监督评估和ragas评估",
    version="1.0.0"
)

# 评估器实例
rag_evaluator = None

# 请求模型
class InitialRetrievalRequest(BaseModel):
    """初步检索评估请求模型"""
    queries: List[str]
    retrieved_results: List[List[Dict[str, Any]]]
    k: int = 5
    use_ragas: bool = False

class RerankRequest(BaseModel):
    """重排序评估请求模型"""
    queries: List[str]
    initial_results: List[List[Dict[str, Any]]]
    reranked_results: List[List[Dict[str, Any]]]
    k: int = 5
    use_ragas: bool = False

# 响应模型
class EvaluationResponse(BaseModel):
    """评估响应模型"""
    precision: float
    recall: float
    f1: float
    ndcg: float
    mrr: float
    metrics: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化评估器"""
    global rag_evaluator
    # 延迟初始化，避免启动时加载模型
    pass

@app.post("/evaluate/initial", response_model=EvaluationResponse)
async def evaluate_initial_retrieval(
    request: InitialRetrievalRequest = Body(...)
):
    """
    评估RAG的初步检索效果
    
    Args:
        request: 评估请求参数
        
    Returns:
        评估结果
    """
    try:
        # 初始化评估器
        evaluator = RAGEvaluator(use_ragas=request.use_ragas)
        
        # 执行评估
        result = evaluator.evaluate_initial_retrieval(
            queries=request.queries,
            retrieved_results=request.retrieved_results,
            k=request.k
        )
        
        # 转换为响应模型
        return EvaluationResponse(
            precision=result.precision,
            recall=result.recall,
            f1=result.f1,
            ndcg=result.ndcg,
            mrr=result.mrr,
            metrics=result.metrics
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"评估失败: {str(e)}")

@app.post("/evaluate/rerank", response_model=EvaluationResponse)
async def evaluate_rerank(
    request: RerankRequest = Body(...)
):
    """
    评估RAG的重排序效果
    
    Args:
        request: 评估请求参数
        
    Returns:
        评估结果
    """
    try:
        # 初始化评估器
        evaluator = RAGEvaluator(use_ragas=request.use_ragas)
        
        # 执行评估
        result = evaluator.evaluate_rerank(
            queries=request.queries,
            initial_results=request.initial_results,
            reranked_results=request.reranked_results,
            k=request.k
        )
        
        # 转换为响应模型
        return EvaluationResponse(
            precision=result.precision,
            recall=result.recall,
            f1=result.f1,
            ndcg=result.ndcg,
            mrr=result.mrr,
            metrics=result.metrics
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"评估失败: {str(e)}")

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "RAG评估接口",
        "endpoints": {
            "evaluate_initial": "/evaluate/initial",
            "evaluate_rerank": "/evaluate/rerank"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)