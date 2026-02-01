# RAG项目说明

## 1. 项目简介

Python 项目，使用M3E embedding，用Faiss生成向量索引，将索引存入resource_package/storage文件夹下，后续直接使用本地索引调用qwen-plus回答。

## 2. 项目目录结构

（已排除 .venv 虚拟环境和 resource_package/models 目录）

```text
rag_demo1/  # 项目根目录
├── application/  # 基于大模型问答
│   ├── __init__.py
│   ├── app1.py  # 基于本地索引回答问题
│   └── rag_prompt_template.txt  
├── common/
│   ├── __init__.py
│   ├── file_utils.py  # 读取本地配置、资源的工具类
│   ├── model_test.ipynb
│   └── model_util.py  # 加载本地模型的工具类。model不存在则下载到本地
├── config/
├── data_handle/
│   ├── __init__.py
│   └── json_handle.py   # 解析json并spilt，返回chunks
├── embedding/
│   ├── __init__.py
│   └── M3EEmbedding.py  #对chunks用M3E embedding
├── storage/
│   ├── __init__.py
│   └── Faiss_storage.py  # 将embedding后的内容存储本地并生成索引。后续使用本地索引检索
├── test/
│   └── demo1.ipynb     # 验证整个功能
├── resource_package/
│   ├── PsyDTCorpus_train_mulit_turn_packing.json
│   └── storage/
│       └── faiss_fixed_pq_1.13.2.index  #生成的本地索引
├── .gitignore
├── .python-version
├── README.md
├── main.py
├── pyproject.toml
├── rag_demo1.rar
└── uv.lock
```
