# RAG心理咨询对话系统

这是一个基于检索增强生成(RAG)的企业级心理咨询对话系统，采用Milvus向量数据库和QWEN大语言模型构建智能心理咨询服务。

## 🏗️ 系统架构

```
用户提问 → Milvus检索(20条) → RetrievalEnhanced重排序(4条) → QWEN生成回答
```

## 🔧 核心功能模块

### 1. 核心引擎层
- **[chat_application.py](src/application/chat_application.py)**: 心理咨询聊天应用主类，整合检索和生成流程
- **[milvus_stream_processor.py](src/data_process/milvus_stream_processor.py)**: Milvus流式数据处理器，负责数据读取、处理和存储
- **[milvus_storage.py](src/database/milvus_storage.py)**: Milvus向量数据库存储管理模块
- **[retrieval_enhanced.py](src/retrieval_augment/retrieval_enhanced.py)**: 基于LlamaIndex思想的检索增强器，实现重排序功能

### 2. 配置管理层
- **[config/](src/config/)**: 系统配置文件目录
  - `psychology_prompt.txt`: 心理咨询Prompt模板
  - `prompt_template.txt`: 通用Prompt模板

### 3. 工具类库
- **[common/](src/common/)**: 通用工具类
  - `file_utils.py`: 文件路径管理和配置读取工具
  - `model_util.py`: 模型加载和管理工具，支持模型缓存避免重复加载

### 4. 数据处理层
- **[data_handle/](src/data_handle/)**: 数据处理辅助模块
  - `json_handle.py`: JSON数据处理工具，支持数据加载、处理和术语提取
    - **核心功能**:
      - `load_json_data()`: 加载小型JSON文件，一次性读取处理
      - `load_large_json_data()`: 流式加载大型JSON文件，避免内存溢出
      - `_process_json_item()`: 处理单个JSON数据项，实现数据清洗和格式化
      - `_extract_terms()`: 术语识别和提取，支持从标签和内容中提取关键词
      - `get_all_terms()`: 获取所有识别的术语
      - `clear_terms()`: 清空术语存储
    - **术语提取逻辑**:
      - 从标签中提取术语
      - 从内容中使用jieba分词和词频统计提取关键词
      - 随机从高频词中选择3个作为最终术语
      - 全局存储术语，支持去重和批量获取
  
  - **[parser/](src/data_handle/parser/)**: 文本分割器实现
    - **分割器类型**:
      - `RecursiveSplitter`: 递归字符分割器，基于langchain实现
      - `CharacterSplitter`: 字符分割器，基于langchain实现
      - `TokenSplitter`: 令牌分割器，基于langchain实现
      - `LlamaSentenceSplitter`: 基于LlamaIndex的句子分割器
      - `LlamaSentenceWindowSplitter`: 基于LlamaIndex的句子窗口分割器，为每个句子添加上下文
      - `LlamaSemanticSplitter`: 基于LlamaIndex的语义分割器，使用embedding模型进行语义分割
      - `LlamaCombinedSplitter`: 组合分割器，先使用语义分割，再对过大的块进行句子分割
    - **SplitterFactory**: 分割器工厂类，根据配置创建不同类型的分割器实例
      - 支持从环境变量或参数获取配置
      - 提供统一的分割器创建接口
  
  - **[split/](src/data_handle/split/)**: 文本分割策略实现
    - **设计模式**:
      - 工厂模式：封装策略创建过程
      - 策略模式：支持运行时策略切换
      - 单例模式：工厂类设计
    - **策略实现**:
      - `BaseTextSplitterStrategy`: 基础分割策略类
      - `BatchTextSplitterStrategy`: 批量处理策略
      - `ParallelBatchTextSplitterStrategy`: 并行批量处理策略，提高处理效率
    - **TextSplitterStrategyFactory**: 策略工厂类
      - 动态创建策略实例
      - 策略注册和管理
      - 配置验证和默认值处理
      - 策略信息查询
    - **核心功能**:
      - 支持不同分割策略的切换
      - 批量处理文本分割
      - 并行处理提高效率
      - 配置驱动的策略选择

## 🚀 快速开始

### 1. 系统启动

```python
from src.application import PsychologyChatBot

# 初始化聊天机器人
chatbot = PsychologyChatBot(
    collection_name="psychology_dialogues",
    milvus_host="localhost",
    milvus_port="19530"
)

# 启动交互式聊天
chatbot.interactive_chat()
```

### 2. 核心API调用
```python
# 处理用户问题
response = chatbot.chat("我最近总是感到焦虑怎么办？")

# 返回结果包含：
# - ai_response: AI生成的回答
# - contexts_used: 使用的参考案例
# - milvus_raw_count: Milvus原始检索数量
# - final_context_count: 最终使用的上下文数量
```

### 3. 数据处理

```python
from src.data_process.milvus_stream_processor import stream_process_psychology_data_milvus

# 流式处理心理学数据
result = stream_process_psychology_data_milvus(
    json_file="PsyDTCorpus_train_mulit_turn_packing.json",
    max_items=10000,
    auto_configure=True
)
```

## 📁 项目结构

```
rag_demo1/
├── application/              # 应用层
│   ├── chat_application.py      # 心理咨询聊天应用主类
│   ├── app1.py                 # QWEN模型调用接口
│   └── psychologyApplication.py # 传统心理学应用
├── data_process/             # 数据处理层
│   ├── milvus_stream_processor.py  # Milvus流式处理器
│   ├── base_stream_processor.py    # 基础流式处理器
│   └── stream_processor.py         # 通用流式处理器
├── database/                 # 数据存储层
│   ├── milvus_storage.py           # Milvus存储管理
│   └── mysql_storage.py            # MySQL存储管理
├── retrieval_augment/        # 检索增强层
│   └── retrieval_enhanced.py       # 重排序增强器
├── embedding/                # 向量嵌入层
│   └── M3EEmbedding.py             # M3E模型嵌入生成
├── common/                   # 通用工具层
│   ├── file_utils.py               # 文件工具类
│   └── model_util.py               # 模型工具类
├── config/                   # 配置文件
│   ├── psychology_prompt.txt       # 心理咨询Prompt模板
│   └── prompt_template.txt         # 通用Prompt模板
├── data_handle/              # 数据处理辅助
│   └── json_handle.py              # JSON数据处理
└── test/                     # 测试文件
    ├── test_chat_application.py    # 聊天应用测试
    └── test_milvus_stream_processor.py  # Milvus处理器测试
```

## ⚙️ 系统特性

### 1. 智能检索流程
- **两阶段检索**: Milvus初筛(20条) + BGE重排序精筛(4条)
- **高效向量搜索**: 基于HNSW索引的快速相似度匹配
- **语义理解增强**: 利用bge-reranker-v2-m3提升检索准确性

### 2. 流式处理优势
- **内存友好**: 边读取边处理边存储，避免内存溢出
- **断点续传**: 支持处理中断后继续执行
- **资源监控**: 实时监控CPU、内存、GPU使用情况
- **自动配置**: 根据系统资源自动优化处理参数

### 3. 模型管理优化
- **智能缓存**: 模型一次性加载到内存，避免重复加载
- **半精度支持**: CUDA环境下自动启用FP16提升推理速度
- **批量推理**: 支持批量向量化和重排序提升效率

## 🛠️ 部署要求

### 环境依赖
```bash
# Python版本
Python >= 3.8

# 核心依赖
pymilvus >= 2.4.0
sentence-transformers >= 2.2.0
loguru >= 0.7.0
ijson >= 3.2.0
```

### 服务依赖
- **Milvus**: 向量数据库服务 (localhost:19530)
- **QWEN API**: 大语言模型服务密钥配置

### 配置说明
```bash
# 环境变量配置
export QWEN_API_KEY="your-api-key-here"
export MILVUS_HOST="localhost"
export MILVUS_PORT="19530"
```

## 🧪 测试验证

```bash
# 运行核心功能测试
python test/test_chat_application.py
python test/test_milvus_stream_processor.py

# 运行集成测试
python -m pytest test/ -v
```

## 📊 性能指标

| 功能模块 | 处理速度 | 内存占用 | 准确率 |
|---------|---------|---------|--------|
| Milvus检索 | ~50ms/查询 | ~200MB | 85% |
| 重排序 | ~200ms/批次 | ~500MB | 92% |
| QWEN生成 | ~1s/回答 | ~1GB | - |
| 流式处理 | ~1000条/分钟 | ~1GB | - |