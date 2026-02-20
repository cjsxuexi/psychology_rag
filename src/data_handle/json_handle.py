# %%
import json
import os
from collections import Counter

import ijson
import jieba
import pandas as pd
from loguru import logger
from tqdm import tqdm

from src import common as fu
from src.data_handle.parser import create_splitter
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
        self.text_splitter = create_splitter()

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

    def parallel_handle_json_data(self, json_name="PsyDTCorpus_train_mulit_turn_packing.json", max_items=3):
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

                # 使用本地方法处理数据项
                dialogue_str, metadata = _process_json_item(item)

                # 添加到结果列表
                processed_texts.append(dialogue_str)
                metadata_list.append(metadata)

                df = pd.DataFrame({
                    "text": processed_texts,
                    "metadata": metadata_list
                })
                logger.info(f"原始JSON数据条目数：{len(df)}")

                # 2. 数据筛选
                df_filtered = self.filter_low_quality_data(df)
                logger.info(f"数据筛选完成：原始{len(df)}条 → 有效{len(df_filtered)}条")

                # 3. 统一调用拆分接口
                raw_texts = df_filtered["text"].tolist()
                metadata_list = df_filtered["metadata"].tolist()
                yield self.split_texts(
                    raw_texts,
                    batch_size=20,
                    metadata_list=metadata_list
                )

        logger.info(f"流式加载完成，共提取 {item_count} 个数据片段")

    # ===================== 对外统一拆分接口 =====================
    def split_texts(self, texts, metadata_list=None, batch_size=500):
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


# 全局术语存储
_all_terms = set()


# ===================== 数据处理辅助函数 =====================
def _process_json_item(item):
    """
    处理单个JSON数据项
    
    Args:
        item: JSON数据项
        
    Returns:
        tuple: (processed_text, metadata)
    """
    # 提取核心字段
    item_id = item.get("id", 0)
    tag = item.get("normalizedTag", "无标签")
    messages = item.get("messages", [])
    # 计算content数量，content数量/2作为total_turns
    content_count = sum(1 for msg in messages if msg.get("content", "").strip())
    total_turns = content_count // 2

    # 拼接多轮对话：按 角色:内容 格式拼接，保留对话上下文
    dialogue_parts = []
    content_text = ""
    role_mapping = {
        "user": "来访者",
        "assistant": "咨询师"
    }

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "").strip()
        content_text += content + " "
        # 跳过system角色的消息，不纳入content拼接
        if role == "system":
            continue

        if content:  # 只处理有内容的消息
            role_cn = role_mapping.get(role, role)
            dialogue_parts.append(f"{role_cn}：{content}")

    # 拼接全部content内容，不同角色会话用\n分隔
    dialogue_str = "\n".join(dialogue_parts) + "\n"

    # 识别术语
    terms = _extract_terms(tag, content_text)

    # 将术语添加到全局存储
    global _all_terms
    _all_terms.update(terms)

    # 构建元数据
    metadata = {
        'id': item_id,
        'tag': tag,
        'total_turns': total_turns,
        'terms': terms
    }

    return dialogue_str, metadata


def _extract_terms(tag, content):
    """
    用jieba+自定义规则，提取术语
    多语言可以用spaCy
    Args:
        tag: 标签
        content: 内容文本
        
    Returns:
        list[str]: 术语列表
    """
    terms = []

    # 1. 添加标签作为术语
    if tag and tag != "无标签":
        terms.append(tag)

    # 2. 从内容中提取关键词
    # 这里使用简单的基于规则的方法，实际应用中可以使用更复杂的模型
    # 推荐的开源模型：keybert、textrank4zh、HanLP
    # 推荐的大模型：Qwen1.5、ChatGLM3

    # 简单的关键词提取（基于分词和频率）
    # 分词
    words = jieba.cut(content)

    # 过滤停用词和短词
    stop_words = set(
        ['你们', '来访者', '帮助', '感到', '行为', '自己', "的", "了", "和", "是", "在", "有", "我", "他", "她", "它",
         "这", "那", "你", "您", "我们", "他们", "她们",
         "它们", "然后", "但是", "所以", "因为", "如果", "虽然", "然而", "而且", "或者", "还是", "只是", "已经", "曾经",
         "将会", "可以", "应该", "必须", "可能", "也许", "大概", "几乎", "完全", "非常", "特别", "十分", "很", "太",
         "更", "最", "比较", "稍微", "一点", "一些", "许多", "很多", "不少", "大量", "大部分", "全部", "所有", "每个",
         "各个", "任何", "某", "某个", "某些", "这个", "那个", "这些", "那些", "这里", "那里", "这儿", "那儿", "这里",
         "那里", "现在", "过去", "未来", "今天", "明天", "昨天", "今年", "去年", "明年", "上午", "下午", "晚上", "早上",
         "中午", "傍晚", "深夜", "凌晨", "时候", "时间", "时刻", "时期", "阶段", "过程", "经历", "经验", "体验", "感受",
         "感觉", "感情", "情绪", "心情", "心理", "心态", "态度", "看法", "观点", "意见", "建议", "想法", "思考", "思想",
         "思维", "逻辑", "道理", "原因", "结果", "目的", "目标", "方向", "方式", "方法", "办法", "手段", "工具", "设备",
         "机器", "仪器", "装置", "系统", "体系", "制度", "机制", "结构", "组织", "机构", "部门", "单位", "公司", "企业",
         "工厂", "学校", "医院", "政府", "国家", "社会", "世界", "地球", "宇宙", "自然", "环境", "生态", "资源", "能源",
         "材料", "物质", "元素", "成分", "组成", "结构", "功能", "作用", "效果", "影响", "意义", "价值", "重要性",
         "必要性", "可能性", "可行性", "合理性", "合法性", "道德性", "伦理性", "科学性", "艺术性", "技术性", "专业性",
         "业余性", "普遍性", "特殊性", "共性", "个性", "一般性", "特殊性", "普遍性", "特殊性", "整体", "部分", "全局",
         "局部", "全部", "部分", "整体", "部分", "宏观", "微观", "长期", "短期", "长期", "短期", "现在", "未来", "现在",
         "过去", "主观", "客观", "主观", "客观", "积极", "消极", "正面", "负面", "优点", "缺点", "优势", "劣势", "机会",
         "威胁", "机遇", "挑战", "成功", "失败", "胜利", "失败", "进步", "退步", "发展", "倒退", "增长", "减少", "增加",
         "减少", "上升", "下降", "提高", "降低", "增强", "减弱", "扩大", "缩小", "扩张", "收缩", "前进", "后退", "向前",
         "向后", "向上", "向下", "向左", "向右", "对内", "对外", "对内", "对外", "直接", "间接", "直接", "间接", "主要",
         "次要", "主要", "次要", "根本", "表面", "本质", "现象", "内容", "形式", "内容", "形式", "原因", "结果", "原因",
         "结果", "目的", "手段", "目的", "手段", "理论", "实践", "理论", "实践", "认识", "实践", "知识", "实践", "真理",
         "谬误", "正确", "错误", "真实", "虚假", "诚实", "虚伪", "善良", "邪恶", "美好", "丑陋", "光明", "黑暗", "正义",
         "邪恶", "公平", "不公平", "平等", "不平等", "自由", "不自由", "民主", "专制", "法治", "人治", "文明", "野蛮",
         "先进", "落后", "现代", "传统", "创新", "守旧", "开放", "封闭", "包容", "排斥", "合作", "竞争", "团结", "分裂",
         "和谐", "冲突", "和平", "战争", "稳定", "动荡", "安全", "危险", "健康", "疾病", "快乐", "痛苦", "幸福", "不幸",
         "成功", "失败", "胜利", "失败", "进步", "退步", "发展", "倒退", "增长", "减少", "增加", "减少", "上升", "下降",
         "提高", "降低", "增强", "减弱", "扩大", "缩小", "扩张", "收缩", "前进", "后退", "向前", "向后", "向上", "向下",
         "向左", "向右", "对内", "对外", "对内", "对外", "直接", "间接", "直接", "间接", "主要", "次要", "主要", "次要",
         "根本", "表面", "本质", "现象", "内容", "形式", "内容", "形式", "原因", "结果", "原因", "结果", "目的", "手段",
         "目的", "手段", "理论", "实践", "理论", "实践", "认识", "实践", "知识", "实践", "真理", "谬误", "正确", "错误",
         "真实", "虚假", "诚实", "虚伪", "善良", "邪恶", "美好", "丑陋", "光明", "黑暗", "正义", "邪恶", "公平",
         "不公平", "平等", "不平等", "自由", "不自由", "民主", "专制", "法治", "人治", "文明", "野蛮", "先进", "落后",
         "现代", "传统", "创新", "守旧", "开放", "封闭", "包容", "排斥", "合作", "竞争", "团结", "分裂", "和谐", "冲突",
         "和平", "战争", "稳定", "动荡", "安全", "危险", "健康", "疾病", "快乐", "痛苦", "幸福", "不幸"])

    filtered_words = [word for word in words if word not in stop_words and len(word) > 1]

    # 统计词频并提取高频词
    word_counts = Counter(filtered_words)
    top_words = [word for word, _ in word_counts.most_common(10)]  # 提取前10个高频词

    # 随机从10个词中提取3个作为高频词
    import random
    if len(top_words) > 3:
        selected_words = random.sample(top_words, 3)
    else:
        selected_words = top_words

    terms.extend(selected_words)

    return list(set(terms))  # 去重


def get_all_terms():
    """
    获取所有识别的术语
    
    Returns:
        list[str]: 所有术语的列表
    """
    global _all_terms
    return list(_all_terms)


def clear_terms():
    """
    清空术语存储
    """
    global _all_terms
    _all_terms.clear()


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
        # 使用本地方法处理数据项
        dialogue_str, metadata = _process_json_item(item)

        # 加入结果列表
        processed_texts.append(dialogue_str)
        metadata_list.append(metadata)

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

            # 使用本地方法处理数据项
            dialogue_str, metadata = _process_json_item(item)

            # 添加到结果列表
            processed_texts.append(dialogue_str)
            metadata_list.append(metadata)

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
        batch_size=20,
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
        batch_size=20,
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
    # logger.info("1. 默认提取前3个片段:")
    # df1 = load_json_data(max_items=3)
    # logger.info(f"   提取到 {len(df1)} 条数据")

    # 示例2: 使用配置策略拆分
    logger.info("3. 使用配置的拆分策略处理前3条数据:")
    # simple_split(max_items=3)

    batch_split(max_items=3)
    logger.info("\n=== 演示完成 ===")


# ===================== 主执行流程 =====================
if __name__ == "__main__":
    # 运行演示函数
    # demo_extract_control()
    # logger.info(get_all_terms())
    # get_all_terms()
    # 或者运行批量处理
    # batch_split(max_items=10)
    handle = JsonHandle("PsyDTCorpus_train_mulit_turn_packing.json")
    # 迭代生成器并打印结果
    logger.info("=== 测试 parallel_handle_json_data 方法 ===")
    for i, (text_chunks, chunk_metadata) in enumerate(handle.parallel_handle_json_data(max_items=3)):
        logger.info(f"第 {i+1} 批处理结果:")
        logger.info(f"文本块数量: {len(text_chunks)}")
        logger.info(f"元数据数量: {len(chunk_metadata)}")
        if text_chunks:
            logger.info(f"示例文本块: {text_chunks[0][:200]}...")
        if chunk_metadata:
            logger.info(f"示例元数据: {chunk_metadata[0]}")
    logger.info("=== 测试完成 ===")