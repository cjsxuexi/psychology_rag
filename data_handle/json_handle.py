# %%
import pandas as pd
import numpy as np
import json
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from tqdm import tqdm
import os
import common.file_utils as fu
import ijson
from multiprocessing import Pool

load_dotenv(dotenv_path=fu.get_config_path('.env'))


class JsonHandle:
    """
    处理json数据，并返回处理后的文本块列表
    """

    def __init__(self, json_name="PsyDTCorpus_train_mulit_turn_packing.json"):
        self.json_name = json_name
        self.min_length = int(os.getenv("MIN_TEXT_LENGTH"))
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("TEXT_CHUNK_SIZE")),
            chunk_overlap=int(os.getenv("TEXT_CHUNK_OVERLAP")),
            length_function=len,
            separators=["\n\n", "\n", "。", "，", " "]  # 优先按语义分割
        )

    # ===================== 数据筛选（复用原逻辑，适配真实数据） =====================
    def filter_low_quality_data(self, df):
        """
        筛选低质量数据：
        - 去除重复文本
        - 去除过短文本（避免无意义语义）
        """
        # a. 去除重复文本
        df = df.drop_duplicates(subset=["text"], keep="first")
        # b. 去除长度过短的文本（对话文本至少保留50字符，确保有语义价值）
        df = df[df["text"].str.len() >= self.min_length]
        # c. 重置索引
        df = df.reset_index(drop=True)
        return df

    # ===================== 文本分割（复用原分批处理逻辑，适配对话文本） =====================
    def batch_split_texts(self, texts, batch_size=10_000, metadata_list=None):
        """
        分批分割文本：
        - 对话文本较长，chunk_size调整为300（适配多轮对话语义）
        - chunk_overlap调整为30（保留对话上下文关联）
        - 支持将元信息嵌入到每个chunk中
        
        参数:
            texts: 待分割的文本列表
            batch_size: 批处理大小
            metadata_list: 对应的元信息列表，每个元素包含{'tag': str, 'id': int, 'total_turns': int}
        """

        split_texts = []
        split_metadata = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="分批分割对话文本"):
            batch = texts[i:i + batch_size]
            batch_metadata = metadata_list[i:i + batch_size] if metadata_list else [None] * len(batch)
            
            for text, meta in zip(batch, batch_metadata):
                chunks = self.text_splitter.split_text(text)
                split_texts.extend(chunks)
                
                # 为每个chunk复制元信息
                if meta:
                    split_metadata.extend([meta.copy() for _ in chunks])
                else:
                    split_metadata.extend([None for _ in chunks])
        
        return split_texts, split_metadata

    def split_single_text(self, text):
        return self.text_splitter.split_text(text)

    def parallel_batch_split_texts(self, texts, batch_size=10_000, metadata_list=None):
        """
        并行分批分割文本
        
        参数:
            texts: 待分割的文本列表
            batch_size: 批处理大小
            metadata_list: 对应的元信息列表
        """
        split_texts = []
        split_metadata = []
        
        with Pool(processes=os.cpu_count()) as pool:
            for i in tqdm(range(0, len(texts), batch_size), desc="并行分割文本"):
                batch = texts[i:i + batch_size]
                batch_metadata = metadata_list[i:i + batch_size] if metadata_list else [None] * len(batch)
                
                # 并行处理文本分割
                results = pool.starmap(self.split_single_text, [(text,) for text in batch])
                
                # 处理结果和元信息
                for res, meta in zip(results, batch_metadata):
                    split_texts.extend(res)
                    if meta:
                        split_metadata.extend([meta.copy() for _ in res])
                    else:
                        split_metadata.extend([None for _ in res])
        
        return split_texts, split_metadata


# ===================== 核心修改：加载并处理 json 数据 =====================
def load_json_data(json_name="PsyDTCorpus_train_mulit_turn_packing.json", max_items=3):
    """
    加载PsyDTCorpus_train_mulit_turn_packing.json数据，处理逻辑：
    1. 读取JSON文件，支持多条目（即使只有1条也兼容）
    2. 拼接单条数据的多轮对话为结构化文本（带角色标识）
    3. 关联normalizedTag标签，增强文本语义信息
    
    参数:
        json_name (str): JSON文件名
        max_items (int): 最大提取条目数，默认3个，None表示提取全部
    """
    # 1. 读取JSON文件
    path = fu.get_resource_path(json_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到数据文件：{path}，请确认路径正确")

    with open(path, "r", encoding="utf-8") as f:
        data_list = json.load(f)  # 读取为列表，支持多条对话数据

    # 控制提取条目数量
    if max_items is not None:
        data_list = data_list[:max_items]
        print(f"提取前 {len(data_list)} 个数据片段")
    else:
        print(f"提取全部 {len(data_list)} 个数据片段")

    processed_texts = []
    metadata_list = []
    
    for item in tqdm(data_list, desc="处理JSON数据条目"):
        # 提取核心字段
        item_id = item.get("id", 0)
        tag = item.get("normalizedTag", "无标签")
        messages = item.get("messages", [])
        total_turns = len(messages)

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
        metadata_list.append({
            'id': item_id,
            'tag': tag,
            'total_turns': total_turns
        })

    return pd.DataFrame({
        "text": processed_texts,
        "metadata": metadata_list
    })


def load_large_json_data(json_name="PsyDTCorpus_train_mulit_turn_packing.json", max_items=3):
    processed_texts = []
    json_path = fu.get_resource_path(json_name)
    item_count = 0
    
    with open(json_path, "r", encoding="utf-8") as f:
        # 逐行读取JSON数组中的每个item
        for item in ijson.items(f, "item"):
            # 控制提取条目数量
            if max_items is not None and item_count >= max_items:
                break
            item_count += 1
            tag = item.get("normalizedTag", "无标签")
            messages = item.get("messages", [])
            # 拼接逻辑同上
            dialogue_str = f"【标签】{tag}\n【对话】\n"
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "").strip()
                if role and content:
                    role_cn = {"system": "系统提示", "user": "来访者", "assistant": "咨询师"}.get(role, role)
                    dialogue_str += f"{role_cn}：{content}\n"
            processed_texts.append(dialogue_str)
    return pd.DataFrame({"text": processed_texts})


def simple_split(max_items=3):
    """
    简单测试函数
    
    参数:
        max_items (int): 最大提取条目数，默认3个
    
    返回:
        tuple: (chunks列表, metadata列表)
    """
    handle = JsonHandle("PsyDTCorpus_train_mulit_turn_packing.json")
    # 1. 加载json数据（默认提取前3个片段）
    df = load_json_data(json_name=handle.json_name, max_items=max_items)
    print(f"原始JSON数据条目数：{len(df)}")

    # 2. 数据筛选：去除重复、过短文本
    df_filtered = handle.filter_low_quality_data(df)
    print(f"数据筛选完成：原始{len(df)}条 → 有效{len(df_filtered)}条")

    # 3. 提取文本和元信息并执行分割
    raw_texts = df_filtered["text"].tolist()
    metadata_list = df_filtered["metadata"].tolist()
    
    valid_text_chunks, chunk_metadata = handle.batch_split_texts(
        raw_texts,
        batch_size=200,  # 百万级数据可保持此批次大小
        metadata_list=metadata_list
    )

    print(f"文本分割完成：有效文本→{len(valid_text_chunks)}个文本块")
    print(f"示例文本块：\n{valid_text_chunks[0][:200]}...")
    
    # 显示元信息示例
    if chunk_metadata and chunk_metadata[0]:
        print(f"示例元信息：{chunk_metadata[0]}")
    
    return valid_text_chunks, chunk_metadata


def batch_split(max_items=None, file_name="PsyDTCorpus_train_mulit_turn_packing.json"):
    handle = JsonHandle(file_name)
    # 1. 加载json数据
    df = load_large_json_data(json_name=handle.json_name, max_items=max_items)  # 替换为你的JSON文件实际路径
    print(f"原始JSON数据条目数：{len(df)}")

    # 2. 数据筛选：去除重复、过短文本
    df_filtered = handle.filter_low_quality_data(df)
    print(f"数据筛选完成：原始{len(df)}条 → 有效{len(df_filtered)}条")

    # 3. 提取文本并执行分割
    raw_texts = df_filtered["text"].tolist()
    valid_text_chunks = handle.parallel_batch_split_texts(
        raw_texts,
        batch_size=200,  # 百万级数据可保持此批次大小
    )

    print(f"文本分割完成：有效文本→{len(valid_text_chunks)}个文本块")
    print(f"示例文本块：\n{valid_text_chunks[0]}")
    return valid_text_chunks


def demo_extract_control():
    """
    演示不同提取控制方式的使用示例
    """
    print("=== JSON数据提取控制演示 ===\n")
    
    # 示例1: 默认提取前3个片段
    print("1. 默认提取前3个片段:")
    df1 = load_json_data(max_items=3)
    print(f"   提取到 {len(df1)} 条数据\n")
    
    # 示例2: 提取前5个片段
    print("2. 提取前5个片段:")
    df2 = load_json_data(max_items=5)
    print(f"   提取到 {len(df2)} 条数据\n")
    
    # 示例3: 提取全部数据
    print("3. 提取全部数据:")
    # df3 = load_json_data(max_items=None)
    # print(f"   提取到 {len(df3)} 条数据\n")
    
    # 示例4: 使用simple函数（默认3个）
    print("4. 使用simple函数（默认提取3个）:")
    # simple()
    
    print("\n=== 演示完成 ===")


# ===================== 主执行流程 =====================
if __name__ == "__main__":
    # 运行演示函数
    demo_extract_control()
    
    # 或者运行原来的测试函数
    # batch_test()
