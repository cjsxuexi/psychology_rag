# GraphRAG+传统RAG混合检索插件

## 简介

本插件实现了GraphRAG与传统RAG相结合的混合检索功能，参照Dify的知识库检索插件设计，提供更精准、更全面的信息检索能力。

## 核心特性

### 1. GraphRAG功能
- 自动构建知识图谱，捕捉文档间的关系和语义结构
- 基于图的检索算法，利用实体关系提升检索质量
- 支持实体提取和关系构建

### 2. 混合检索策略
- 融合传统向量检索和基于图的检索结果
- 可配置的权重参数，平衡两种检索方式
- 智能结果融合和重排序

### 3. Dify兼容接口
- 与Dify知识库检索插件接口兼容
- 支持标准的插件调用方式
- 提供符合Dify格式的返回结果

### 4. 性能优化
- 支持批处理和并行计算
- 高效的知识图谱构建算法
- 优化的检索流程，确保响应时间<1秒

## 安装依赖

```bash
# 基本依赖
uv add torch sentence-transformers networkx

# 其他依赖
uv add jieba  # 用于文本处理
```

## 快速开始

### 1. 初始化混合检索器

```python
from src.retrieval_augment.hybrid_retrieval import HybridRetrieval

# 初始化混合检索器
hybrid = HybridRetrieval(
    collection_name="psychology_dialogues",  # Milvus集合名称
    milvus_host="localhost",              # Milvus服务主机
    milvus_port="19530",                  # Milvus服务端口
    rag_weight=0.5,                        # 传统RAG权重
    graph_weight=0.5                       # GraphRAG权重
)

print(hybrid)
```

### 2. 执行混合检索

```python
# 执行混合检索
query = "情侣之间因为金钱问题产生矛盾怎么办？"
results = hybrid.hybrid_search(
    query=query,
    top_k=5,          # 返回结果数量
    search_top_k=20    # 每个检索器返回的结果数
)

# 查看结果
print("混合检索结果：")
for i, result in enumerate(results):
    print(f"{i+1}. 混合分数: {result['hybrid_score']:.4f}")
    print(f"   RAG分数: {result['rag_score']:.4f}, Graph分数: {result['graph_score']:.4f}")
    print(f"   内容: {result['content'][:100]}...")
```

### 3. 使用Dify兼容接口

```python
from src.retrieval_augment.dify_plugin import Tool

# 初始化Dify工具
tool = Tool()

# 执行搜索
inputs = {
    "query": "如何处理工作压力？",
    "top_k": 3
}
result = tool.run(inputs)

print("Dify工具运行结果：")
print(result)
```

## 模块说明

### 1. GraphRAG

`src/retrieval_augment/graphrag.py`

- **功能**: 实现知识图谱构建和基于图的检索
- **核心方法**:
  - `build_knowledge_graph()`: 构建知识图谱
  - `search_graph()`: 基于图的检索
  - `get_graph_info()`: 获取知识图谱信息

### 2. HybridRetrieval

`src/retrieval_augment/hybrid_retrieval.py`

- **功能**: 集成传统RAG和GraphRAG的检索结果
- **核心方法**:
  - `hybrid_search()`: 执行混合检索
  - `update_weights()`: 更新权重参数
  - `rebuild_graph()`: 重新构建知识图谱

### 3. DifyKnowledgeBasePlugin

`src/retrieval_augment/dify_plugin.py`

- **功能**: 提供与Dify知识库检索插件兼容的接口
- **核心方法**:
  - `search()`: 搜索接口
  - `get_plugin_info()`: 获取插件信息
  - `update_config()`: 更新插件配置

## 配置参数

### HybridRetrieval配置

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| collection_name | str | "psychology_dialogues" | Milvus集合名称 |
| milvus_host | str | "localhost" | Milvus服务主机 |
| milvus_port | str | "19530" | Milvus服务端口 |
| rag_weight | float | 0.5 | 传统RAG的权重 |
| graph_weight | float | 0.5 | GraphRAG的权重 |
| model_name | str | "BAAI/bge-reranker-v2-m3" | 重排序模型名称 |

### 检索参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| query | str | - | 查询文本 |
| top_k | int | 5 | 返回结果数量 |
| search_top_k | int | 20 | 每个检索器返回的结果数 |

## 性能优化建议

1. **知识图谱构建**: 对于大规模知识库，可适当调整`build_knowledge_graph()`中的`top_k`参数，控制构建的文档数量

2. **权重调整**: 根据具体场景调整`rag_weight`和`graph_weight`，例如：
   - 对于结构化知识，增加`graph_weight`
   - 对于语义相似度搜索，增加`rag_weight`

3. **批处理优化**: 调整`RetrievalEnhanced`中的`batch_size`参数，根据硬件资源优化性能

4. **并行计算**: 对于大规模检索，可考虑使用多线程或异步处理

## 故障排除

### 常见问题

1. **ModuleNotFoundError: No module named 'jieba'**
   ```bash
   uv add jieba
   ```

2. **Milvus连接失败**
   - 检查Milvus服务是否运行
   - 确认主机和端口配置正确

3. **知识图谱构建失败**
   - 检查Milvus中是否有数据
   - 确认embedding模型能正常加载

## 版本历史

- v1.0.0: 初始版本，实现GraphRAG+传统RAG混合检索功能

## 贡献

欢迎提交issue和pull request，帮助改进这个插件。

## 许可证

本项目采用MIT许可证。