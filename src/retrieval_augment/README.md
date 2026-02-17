# RetrievalEnhanced 检索增强模块

## 简介

RetrievalEnhanced是一个基于LlamaIndex设计理念的检索增强模块，专门用于对Milvus检索结果进行重排序优化。该模块实现了bge-reranker-v2-m3模型的集成，提供灵活的重排序和混合搜索功能。

## 核心特性

### 1. 解耦设计
- 通过参数传递接收Milvus检索结果，不直接依赖MilvusStreamProcessor
- 模块化设计，易于集成到现有系统中

### 2. 智能重排序
- 使用bge-reranker-v2-m3模型对检索结果进行语义相关性重排序
- 支持本地模型缓存，首次下载后自动使用本地模型

### 3. 混合搜索策略
- 结合向量相似度和语义相关性进行混合排序
- 可调节alpha参数平衡两种排序策略

### 4. 企业级特性
- 完善的日志记录和错误处理
- 支持批量处理和性能优化
- 灵活的配置参数

## 安装依赖

```bash
# 添加FlagEmbedding依赖
uv add FlagEmbedding
```

## 快速开始

### 基本使用

```python
from src.retrieval_augment import RetrievalEnhanced
from src.data_process.milvus_stream_processor import MilvusStreamProcessor

# 1. 初始化检索器
retriever = RetrievalEnhanced(
   model_name="BAAI/bge-reranker-v2-m3",
   use_fp16=True,
   batch_size=16
)

# 2. 执行Milvus检索
processor = MilvusStreamProcessor(collection_name="psychology_dialogues")
query = "情侣之间因为金钱问题产生矛盾怎么办？"
milvus_results = processor.search_similar_chunks(query, top_k=10)

# 3. 执行重排序
reranked_results = retriever.rerank_results(
   query_text=query,
   search_results=milvus_results,
   top_k=5,
   return_scores=True
)

# 查看结果
for i, result in enumerate(reranked_results):
   print(f"{i + 1}. 分数: {result['rerank_score']:.4f}")
   print(f"   内容: {result['content'][:100]}...")
```

### 混合搜索

```python
# 使用混合搜索策略
hybrid_results = retriever.hybrid_search(
    query_text=query,
    search_results=milvus_results,
    top_k=5,
    alpha=0.5  # 0.5表示平衡向量相似度和语义相关性
)

for i, result in enumerate(hybrid_results):
    print(f"{i+1}. 混合分数: {result['hybrid_score']:.4f}")
    print(f"   Rerank分数: {result['norm_rerank_score']:.4f}")
    print(f"   距离分数: {result['norm_distance_score']:.4f}")
```

## API参考

### RetrievalEnhanced类

#### 构造函数
```python
RetrievalEnhanced(
    model_name: str = "BAAI/bge-reranker-v2-m3",
    use_fp16: bool = True,
    batch_size: int = 32,
    max_length: int = 512
)
```

**参数说明:**
- `model_name`: bge-reranker模型名称
- `use_fp16`: 是否使用半精度推理（节省内存）
- `batch_size`: 批处理大小
- `max_length`: 最大序列长度

#### 主要方法

##### `rerank_results()`
对检索结果进行重排序

```python
def rerank_results(
    query_text: str,
    search_results: List[Dict[str, Any]],
    top_k: Optional[int] = None,
    return_scores: bool = True
) -> List[Dict[str, Any]]
```

**参数:**
- `query_text`: 查询文本
- `search_results`: Milvus检索结果列表
- `top_k`: 返回前k个结果
- `return_scores`: 是否返回重排序分数

**返回值:**
重排序后的结果列表，每个结果包含原始字段和新增的`rerank_score`字段

##### `hybrid_search()`
混合搜索：结合向量相似度和语义相关性

```python
def hybrid_search(
    query_text: str,
    search_results: List[Dict[str, Any]],
    top_k: int = 5,
    alpha: float = 0.5
) -> List[Dict[str, Any]]
```

**参数:**
- `alpha`: 混合权重 (0=完全依赖向量距离, 1=完全依赖rerank分数)

**返回值:**
混合排序后的结果，包含`hybrid_score`、`norm_rerank_score`、`norm_distance_score`等字段

##### `get_model_info()`
获取模型配置信息

```python
def get_model_info() -> Dict[str, Any]
```

## 配置说明

### model_util.py集成

模块使用`common/model_util.py`中的`load_model()`方法加载模型：

```python
def load_model(model_name: str) -> str:
    """
    加载模型的公共方法，支持本地缓存
    首次下载后自动使用本地模型
    """
```

### 参数调优建议

1. **batch_size**: 根据GPU内存调整，一般16-32较为合适
2. **use_fp16**: 在GPU环境下建议开启以节省内存
3. **max_length**: 根据文本长度调整，过长会影响性能
4. **alpha**: 混合搜索参数，可根据具体场景调整

## 使用示例

完整的使用示例请参考：`retrieval_augment/example_usage.py`

```bash
# 运行示例
python retrieval_augment/example_usage.py
```

## 测试

### 运行核心逻辑测试

相关功能已整合到现有测试框架中，请参考项目根目录下的测试文件。

### 运行完整功能测试（需要安装FlagEmbedding）



## 注意事项

1. **依赖兼容性**: FlagEmbedding可能与某些transformers版本存在兼容性问题
2. **模型下载**: 首次使用会自动下载模型到`resource_package/models/`目录
3. **内存使用**: bge-reranker模型较大，建议在GPU环境下使用
4. **性能优化**: 可通过调整batch_size和use_fp16参数优化性能

## 故障排除

### 常见问题

1. **ModuleNotFoundError: No module named 'FlagEmbedding'**
   ```bash
   uv add FlagEmbedding
   ```

2. **ImportError: cannot import name 'is_torch_fx_available'**
   - 这是FlagEmbedding与transformers版本的兼容性问题
   - 可以使用核心逻辑测试文件进行功能验证

3. **模型下载失败**
   - 检查网络连接
   - 确保`HF_ENDPOINT`环境变量设置正确

## 架构设计

### 设计理念
借鉴LlamaIndex的模块化设计思想，将检索增强功能独立封装，便于复用和扩展。

### 核心组件
1. **RetrievalEnhanced**: 主检索器类
2. **model_util集成**: 模型加载和缓存管理
3. **解耦设计**: 通过参数传递实现与Milvus的松耦合

### 扩展性
- 易于集成其他reranker模型
- 支持自定义混合策略
- 可扩展为分布式部署

## 版本历史

- v1.0.0: 初始版本，实现基本重排序和混合搜索功能