# %%
import json
import os

import ijson
import pandas as pd
from loguru import logger
from tqdm import tqdm

from src import common as fu
from src.data_handle.parser import SplitterFactory
from src.data_handle.split import TextSplitterStrategyFactory


# ===================== 核心JSON处理类 =====================
class JsonHandle:
    """
    处理json数据，并返回处理后的文本块列表
    支持通过配置切换拆分策略和分割器类型
    """

    def __init__(self, json_name="PsyDTCorpus_train_mulit_turn_packing.json"):
        self.json_name = json_name
        self.min_length = int(os.getenv("MIN_TEXT_LENGTH", 50))

        # 初始化文本分割器（使用工厂创建，支持配置切换）
        splitter_factory = SplitterFactory()
        self.text_splitter = splitter_factory.create_splitter()

        # 创建策略实例
        factory = TextSplitterStrategyFactory()
        self.split_strategy = factory.create_strategy(
            text_splitter=self.text_splitter,
            min_length=self.min_length
        )

    # ===================== 数据筛选 =====================
    def filter_low_quality_data(self, df):
        """
        筛选低质量数据：
        - 去除重复文本
        - 去除过短文本（避免无意义语义）
        """
        # a. 去除重复文本
        df = df.drop_duplicates(subset=["text"], keep="first")
        # b. 去除长度过短的文本
        df = df[df["text"].str.len() >= self.min_length]
        # c. 重置索引
        df = df.reset_index(drop=True)
        return df

    # ===================== 对外统一拆分接口 =====================
    def split_texts(self, texts, metadata_list=None, batch_size=10_000):
        """
        统一的文本拆分入口（自动使用配置的策略）
        """
        return self.split_strategy.split_texts(
            texts=texts,
            metadata_list=metadata_list,
            batch_size=batch_size
        )

    def split_single_text(self, text):
        """拆分单条文本"""
        return self.text_splitter.split_text(text)


# ===================== JSON数据加载函数 =====================
def load_json_data(json_name="PsyDTCorpus_train_mulit_turn_packing.json", max_items=3):
    """
    加载JSON文件（一次性读取，适合小文件）
    1. 读取JSON文件，支持多条目
    2. 拼接单条数据的多轮对话为结构化文本（带角色标识）
    3. 关联normalizedTag标签，增强文本语义信息
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
        logger.info(f"提取前 {len(data_list)} 个数据片段")
    else:
        logger.info(f"提取全部 {len(data_list)} 个数据片段")

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
    """
    加载大型JSON数据文件（流式读取，避免内存溢出）
    处理逻辑与load_json_data保持一致
    """
    processed_texts = []
    metadata_list = []
    json_path = fu.get_resource_path(json_name)
    item_count = 0

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"未找到数据文件：{json_path}，请确认路径正确")

    # 使用ijson进行流式读取
    with open(json_path, "r", encoding="utf-8") as f:
        # 逐行读取JSON数组中的每个item
        for item in ijson.items(f, "item"):
            # 控制提取条目数量
            if max_items is not None and item_count >= max_items:
                break
            item_count += 1

            # 提取核心字段
            item_id = item.get("id", 0)
            tag = item.get("normalizedTag", "无标签")
            messages = item.get("messages", [])
            total_turns = len(messages)

            # 拼接多轮对话
            dialogue_str = f"【标签】{tag}\n【对话】\n"
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "").strip()
                if role and content:
                    role_cn = {"system": "系统提示", "user": "来访者", "assistant": "咨询师"}.get(role, role)
                    dialogue_str += f"{role_cn}：{content}\n"

            # 添加到结果列表
            processed_texts.append(dialogue_str)
            metadata_list.append({
                'id': item_id,
                'tag': tag,
                'total_turns': total_turns
            })

    logger.info(f"流式加载完成，共提取 {item_count} 个数据片段")
    return pd.DataFrame({
        "text": processed_texts,
        "metadata": metadata_list
    })


# ===================== 测试/执行函数 =====================
def simple_split(max_items=3):
    """简单测试函数（使用配置的拆分策略）"""
    handle = JsonHandle("PsyDTCorpus_train_mulit_turn_packing.json")
    # 1. 加载json数据
    df = load_json_data(json_name=handle.json_name, max_items=max_items)
    logger.info(f"原始JSON数据条目数：{len(df)}")

    # 2. 数据筛选
    df_filtered = handle.filter_low_quality_data(df)
    logger.info(f"数据筛选完成：原始{len(df)}条 → 有效{len(df_filtered)}条")

    # 3. 统一调用拆分接口（自动使用配置的策略）
    raw_texts = df_filtered["text"].tolist()
    metadata_list = df_filtered["metadata"].tolist()

    valid_text_chunks, chunk_metadata = handle.split_texts(
        raw_texts,
        batch_size=200,
        metadata_list=metadata_list
    )

    logger.info(f"文本分割完成：有效文本→{len(valid_text_chunks)}个文本块")
    logger.info(f"示例文本块：\n{valid_text_chunks[0][:200]}...")

    # 显示元信息示例
    if chunk_metadata and chunk_metadata[0]:
        logger.info(f"示例元信息：{chunk_metadata[0]}")

    return valid_text_chunks, chunk_metadata


def batch_split(max_items=None, file_name="PsyDTCorpus_train_mulit_turn_packing.json"):
    """批量处理函数（使用配置的拆分策略）"""
    handle = JsonHandle(file_name)
    # 1. 加载json数据（大型文件使用流式加载）
    df = load_large_json_data(json_name=handle.json_name, max_items=max_items)
    logger.info(f"原始JSON数据条目数：{len(df)}")

    # 2. 数据筛选
    df_filtered = handle.filter_low_quality_data(df)
    logger.info(f"数据筛选完成：原始{len(df)}条 → 有效{len(df_filtered)}条")

    # 3. 统一调用拆分接口
    raw_texts = df_filtered["text"].tolist()
    metadata_list = df_filtered["metadata"].tolist()
    valid_text_chunks, chunk_metadata = handle.split_texts(
        raw_texts,
        batch_size=200,
        metadata_list=metadata_list
    )

    logger.info(f"文本分割完成：有效文本→{len(valid_text_chunks)}个文本块")
    logger.info(f"示例文本块：\n{valid_text_chunks[0]}")

    if chunk_metadata and chunk_metadata[0]:
        logger.info(f"示例元信息：{chunk_metadata[0]}")

    return valid_text_chunks, chunk_metadata


def demo_extract_control():
    """演示不同提取控制方式的使用示例"""
    logger.info("=== JSON数据提取控制演示 ===")

    # 示例1: 默认提取前3个片段
    logger.info("1. 默认提取前3个片段:")
    df1 = load_json_data(max_items=3)
    logger.info(f"   提取到 {len(df1)} 条数据")

    # 示例2: 使用配置策略拆分
    logger.info("3. 使用配置的拆分策略处理前3条数据:")
    simple_split(max_items=3)

    logger.info("\n=== 演示完成 ===")


# ===================== 主执行流程 =====================
if __name__ == "__main__":
    # 运行演示函数
    demo_extract_control()

    # 或者运行批量处理
    # batch_split(max_items=10)
