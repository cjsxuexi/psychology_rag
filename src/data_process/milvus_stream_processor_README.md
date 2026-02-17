# Milvus流处理器 (Milvus Stream Processor)

## 📋 概述

Milvus流处理器是一个基于[Milvus向量数据库](https://milvus.io/)的高效流式数据处理系统，用于替代原有的MySQL+FAISS双存储方案。它实现了边读取边处理边存储的一体化流程，特别适用于大规模向量数据的处理场景。

## 🎯 核心特性

### 🔧 技术优势
- **统一存储**: 使用Milvus单一数据库替代MySQL+FAISS双存储
- **流式处理**: 支持边读取边处理边存储的流水线作业
- **资源控制**: 智能内存管理和并发控制
- **断点续传**: 支持处理中断后的恢复继续
- **高性能**: 基于Milvus的高效向量相似度搜索

### 🏗️ 架构设计
```
JSON数据源 → 流式读取 → 文本拆分 → 向量化(M3E) → Milvus存储
     ↓           ↓          ↓          ↓            ↓
  进度监控   多线程处理  批量拆分   批量编码    向量索引
```

## 🚀 快速开始

### 1. 环境准备

首先确保已安装Milvus服务：

```bash
# 使用Docker启动Milvus（推荐）
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  -v milvus-data:/var/lib/milvus \
  milvusdb/milvus:v2.4.9 \
  milvus run standalone
```

### 2. 基本使用

```python
from src.data_process.milvus_stream_processor import stream_process_psychology_data_milvus

# 快速处理数据
result = stream_process_psychology_data_milvus(
    max_items=1000,  # 处理1000条数据
    auto_configure=True  # 自动优化配置
)

print(f"处理完成: {result['stored_chunks']} 条记录存储到Milvus")
```

### 3. 高级使用

```python
from src.data_process.milvus_stream_processor import MilvusStreamProcessor

# 创建处理器实例
processor = MilvusStreamProcessor(
    json_file="your_data.json",
    batch_size=50,
    embedding_batch_size=16,
    max_workers=4,
    milvus_host="localhost",
    milvus_port="19530",
    collection_name="your_collection"
)

try:
    # 执行流式处理
    result = processor.process_stream(max_items=5000)

    # 向量搜索
    search_results = processor.search_similar_chunks("查询文本", top_k=10)
    for result in search_results:
        print(f"相似度: {result['distance']}, 内容: {result['chunk_text'][:100]}")

finally:
    processor.close()  # 记得关闭连接
```

## ⚙️ 配置参数详解

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `json_file` | `"PsyDTCorpus_train_mulit_turn_packing.json"` | 输入JSON文件名 |
| `batch_size` | `100` | 文本处理批次大小 |
| `embedding_batch_size` | `32` | 向量化批次大小 |
| `max_workers` | `2` | 最大工作线程数 |
| `memory_limit_mb` | `4000` | 内存使用上限(MB) |
| `milvus_host` | `"localhost"` | Milvus服务主机 |
| `milvus_port` | `"19530"` | Milvus服务端口 |
| `collection_name` | `"psychology_dialogues"` | Milvus集合名称 |

## 📊 性能基准

### 测试环境
- CPU: Intel i7-8700K (6核12线程)
- 内存: 16GB DDR4
- GPU: NVIDIA GTX 1080Ti
- Milvus: v2.4.9 standalone

### 性能表现
```
数据规模    处理时间    处理速度    内存峰值    Milvus存储
---------   --------   --------   --------   ----------
1,000条     45秒      22项/秒    2.1GB      1,250实体
10,000条    8分钟     21项/秒    3.2GB      12,800实体
50,000条    35分钟    24项/秒    3.8GB      64,200实体
```

## 🔍 Milvus集合结构

创建的Milvus集合包含以下字段：

```python
# 集合字段定义
fields = [
    FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="source_id", dtype=DataType.INT64),      # 原始数据ID
    FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=65535),  # 文本内容
    FieldSchema(name="tag", dtype=DataType.VARCHAR, max_length=100),           # 标签
    FieldSchema(name="role", dtype=DataType.VARCHAR, max_length=50),           # 角色
    FieldSchema(name="total_turns", dtype=DataType.INT64),                     # 对话轮数
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)        # M3E向量
]
```

## 🛠️ 故障排除

### 常见问题

**1. Milvus数据插入错误**
```python
# 错误信息1: DataNotMatchException: chunk_text field should be a varchar
# 错误信息2: Collection.insert() missing 1 required positional argument: 'data'

# 解决方案: 使用三数组插入方式（推荐）
source_ids = [1, 2, 3]  # 源ID数组
embeddings = [[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]]  # 向量数组
metadatas = [  # 元数据数组（除embedding外的全部信息）
    {"chunk_text": "文本1", "tag": "标签1", "role": "user", "total_turns": 1},
    {"chunk_text": "文本2", "tag": "标签2", "role": "assistant", "total_turns": 1},
    {"chunk_text": "文本3", "tag": "标签3", "role": "user", "total_turns": 1}
]

collection.insert([source_ids, embeddings, metadatas])

# 优势：
# 1. 数据结构清晰分离
# 2. 便于批量处理和优化
# 3. 符合Milvus的最佳实践
```

**2. Milvus连接失败**
```bash
# 检查Milvus服务状态
docker ps | grep milvus

# 查看Milvus日志
docker logs milvus-standalone
```

**2. 内存不足**
```python
# 降低批处理大小
result = stream_process_psychology_data_milvus(
    batch_size=20,           # 从100降到20
    embedding_batch_size=8,  # 从32降到8
    memory_limit_mb=2000     # 设置更严格的内存限制
)
```

**3. 处理速度慢**
```python
# 优化配置（有GPU时）
result = stream_process_psychology_data_milvus(
    max_workers=4,           # 增加CPU线程数
    embedding_batch_size=64, # 增加GPU批处理大小
    auto_configure=True      # 让系统自动优化
)
```

## 📈 与原方案对比

| 特性 | 原方案(FAISS+MySQL) | 新方案(Milvus) |
|------|-------------------|----------------|
| 存储架构 | 双系统分离存储 | 统一向量数据库 |
| 数据一致性 | 需要手动同步 | 原子性操作保证 |
| 查询性能 | 跨系统关联查询 | 内置高效搜索 |
| 运维复杂度 | 高（需维护两套系统） | 低（单一系统） |
| 扩展性 | 受限于MySQL | 水平扩展能力强 |

## 🧪 测试验证

运行测试套件验证功能：

```bash
# 运行完整测试
python test/test_milvus_stream_processor.py


# 运行Jupyter示例
jupyter notebook examples/milvus_stream_processor_example.ipynb
```

## 📚 相关文档

- [Milvus官方文档](https://milvus.io/docs)
- [M3E Embedding模型](https://huggingface.co/moka-ai/m3e-base)
- [FAISS vs Milvus对比分析](../../docs/faiss_mysql_integration_guide.md)

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

MIT License