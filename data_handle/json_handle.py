#%%
import pandas as pd
import numpy as np
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from tqdm import tqdm
import os
import common.file_utils as fu
import ijson
from multiprocessing import Pool

# ===================== 核心修改：加载并处理 json 数据 =====================
def load_json_data(json_name="PsyDTCorpus_train_mulit_turn_packing.json"):
    """
    加载PsyDTCorpus_train_mulit_turn_packing.json数据，处理逻辑：
    1. 读取JSON文件，支持多条目（即使只有1条也兼容）
    2. 拼接单条数据的多轮对话为结构化文本（带角色标识）
    3. 关联normalizedTag标签，增强文本语义信息
    """
    # 1. 读取JSON文件
    path = fu.get_resource_path(json_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到数据文件：{path}，请确认路径正确")

    with open(path, "r", encoding="utf-8") as f:
        data_list = json.load(f)  # 读取为列表，支持多条对话数据

    processed_texts = []
    for item in tqdm(data_list, desc="处理JSON数据条目"):
        # 提取核心字段
        tag = item.get("normalizedTag", "无标签")
        messages = item.get("messages", [])

        # 2. 拼接多轮对话：按 角色:内容 格式拼接，保留对话上下文
        dialogue_str = f"【标签】{tag}\n【对话】\n"
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "").strip()
            if role and content:
                role_cn = {
                    "system": "系统提示",
                    "user": "来访者",
                    "assistant": "咨询师"
                }.get(role, role)
                dialogue_str += f"{role_cn}：{content}\n"

        # 3. 加入结果列表
        processed_texts.append(dialogue_str)

    return pd.DataFrame({"text": processed_texts})

def load_large_json_data(json_name="PsyDTCorpus_train_mulit_turn_packing.json"):
    processed_texts = []
    json_path = fu.get_resource_path(json_name)
    with open(json_path, "r", encoding="utf-8") as f:
        # 逐行读取JSON数组中的每个item
        for item in ijson.items(f, "item"):
            tag = item.get("normalizedTag", "无标签")
            messages = item.get("messages", [])
            # 拼接逻辑同上
            dialogue_str = f"【标签】{tag}\n【对话】\n"
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "").strip()
                if role and content:
                    role_cn = {"system":"系统提示","user":"来访者","assistant":"咨询师"}.get(role, role)
                    dialogue_str += f"{role_cn}：{content}\n"
            processed_texts.append(dialogue_str)
    return pd.DataFrame({"text": processed_texts})

# ===================== 数据筛选（复用原逻辑，适配真实数据） =====================
def filter_low_quality_data(df, min_length=50):
    """
    筛选低质量数据：
    - 去除重复文本
    - 去除过短文本（避免无意义语义）
    """
    # a. 去除重复文本
    df = df.drop_duplicates(subset=["text"], keep="first")
    # b. 去除长度过短的文本（对话文本至少保留50字符，确保有语义价值）
    df = df[df["text"].str.len() >= min_length]
    # c. 重置索引
    df = df.reset_index(drop=True)
    return df

# ===================== 文本分割（复用原分批处理逻辑，适配对话文本） =====================
def batch_split_texts(texts, batch_size=10_000, chunk_size=300, chunk_overlap=30):
    """
    分批分割文本：
    - 对话文本较长，chunk_size调整为300（适配多轮对话语义）
    - chunk_overlap调整为30（保留对话上下文关联）
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "，", " "]  # 优先按语义分割
    )

    split_texts = []
    for i in tqdm(range(0, len(texts), batch_size), desc="分批分割对话文本"):
        batch = texts[i:i+batch_size]
        for text in batch:
            chunks = text_splitter.split_text(text)
            split_texts.extend(chunks)
    return split_texts

def split_single_text(text, splitter):
    return splitter.split_text(text)

def parallel_batch_split_texts(texts, batch_size=10_000, chunk_size=300, chunk_overlap=30):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    split_texts = []
    with Pool(processes=os.cpu_count()) as pool:
        for i in tqdm(range(0, len(texts), batch_size), desc="并行分割文本"):
            batch = texts[i:i+batch_size]
            results = pool.starmap(split_single_text, [(text, splitter) for text in batch])
            for res in results:
                split_texts.extend(res)
    return split_texts


def simple():
    # 1. 加载json数据
    df = load_json_data(json_name="PsyDTCorpus_train_mulit_turn_packing.json")  # 替换为你的JSON文件实际路径
    print(f"原始JSON数据条目数：{len(df)}")

    # 2. 数据筛选：去除重复、过短文本
    df_filtered = filter_low_quality_data(df, min_length=50)
    print(f"数据筛选完成：原始{len(df)}条 → 有效{len(df_filtered)}条")

    # 3. 提取文本并执行分割
    raw_texts = df_filtered["text"].tolist()
    valid_text_chunks = batch_split_texts(
        raw_texts,
        batch_size=10000,  # 百万级数据可保持此批次大小
        chunk_size=300,  # 对话文本推荐300字符/块
        chunk_overlap=30
    )

    print(f"文本分割完成：有效文本→{len(valid_text_chunks)}个文本块")
    print(f"示例文本块：\n{valid_text_chunks[0]}")

def batch_test():
    # 1. 加载json数据
    df = load_large_json_data(json_name="PsyDTCorpus_train_mulit_turn_packing.json")  # 替换为你的JSON文件实际路径
    print(f"原始JSON数据条目数：{len(df)}")

    # 2. 数据筛选：去除重复、过短文本
    df_filtered = filter_low_quality_data(df, min_length=50)
    print(f"数据筛选完成：原始{len(df)}条 → 有效{len(df_filtered)}条")

    # 3. 提取文本并执行分割
    raw_texts = df_filtered["text"].tolist()
    valid_text_chunks = parallel_batch_split_texts(
        raw_texts,
        batch_size=10000,  # 百万级数据可保持此批次大小
        chunk_size=500,  # 对话文本推荐300字符/块
        chunk_overlap=50
    )

    print(f"文本分割完成：有效文本→{len(valid_text_chunks)}个文本块")
    print(f"示例文本块：\n{valid_text_chunks[0]}")
    return valid_text_chunks

# ===================== 主执行流程 =====================
if __name__ == "__main__":
    batch_test()