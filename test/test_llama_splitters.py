#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试基于LlamaIndex的新分割器
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_handle.parser.splitter_factory import SplitterFactory
from src.data_handle.split.factory import TextSplitterStrategyFactory

test_text = """
人工智能（AI）是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。
人工智能的发展历史可以追溯到20世纪50年代，当时计算机科学家开始探索让机器模拟人类思考的可能性。
机器学习是人工智能的一个重要分支，它使计算机能够从数据中学习并做出预测或决策。
深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的学习过程。
自然语言处理（NLP）是人工智能的另一个重要领域，它关注计算机与人类语言之间的交互。
你是一位精通理情行为疗法（Rational Emotive Behavior Therapy，简称REBT）的心理咨询师，能够合理地采用理情行为疗法给来访者提供专业地指导和支持，缓解来访者的负面情绪和行为反应，帮助他们实现个人成长和心理健康。理情行为治疗主要包括以下几个阶段，下面是对话阶段列表，并简要描述了各个阶段的重点。
（1）**检查非理性信念和自我挫败式思维**：理情行为疗法把认知干预视为治疗的“生命”，因此，几乎从治疗一开始，在问题探索阶段，咨询师就以积极的、说服教导式的态度帮助来访者探查隐藏在情绪困扰后面的原因，包括来访者理解事件的思维逻辑，产生情绪的前因后果，借此来明确问题的所在。咨询师坚定地激励来访者去反省自己在遭遇刺激事件后，在感到焦虑、抑郁或愤怒前对自己“说”了些什么。
（2）**与非理性信念辩论**：咨询师运用多种技术（主要是认知技术）帮助来访者向非理性信念和思维质疑发难，证明它们的不现实、不合理之处，认识它们的危害进而产生放弃这些不合理信念的愿望和行为。
（3）**得出合理信念，学会理性思维**：在识别并驳倒非理性信念的基础上，咨询师进一步诱导、帮助来访者找出对于刺激情境和事件的适宜的、理性的反应，找出理性的信念和实事求是的、指向问题解决的思维陈述，以此来替代非理性信念和自我挫败式思维。为了巩固理性信念，咨询师要向来访者反复教导，证明为什么理性信念是合情合理的，它与非理性信念有什么不同，为什么非理性信念导致情绪失调，而理性信念导致较积极、健康的结果。
（4）**迁移应用治疗收获**：积极鼓励来访者把在治疗中所学到的客观现实的态度，科学合理的思维方式内化成个人的生活态度，并在以后的生活中坚持不懈地按理情行为疗法的教导来解决新的问题。
LlamaIndex 是一个用于构建 LLM 应用程序的数据框架。         它提供了一套工具，可以帮助开发者将私有数据与大型语言模型（LLMs）连接起来，实现包括问答、检索增强生成（RAG）等功能。         LlamaIndex 支持多种数据源，包括 PDF、数据库、API 等。其核心概念包括文档加载器、节点解析器、索引和查询引擎。
         文档加载器负责将各种格式和来源的数据摄取到 LlamaIndex 中。         节点解析器随后将这些加载的文档分解成更小、更易于管理的单元，称为节点。         这些节点通常是句子或段落，具体取决于解析策略。索引是构建在这些节点之上的数据结构，旨在实现高效存储和检索，         通常涉及向量嵌入以进行语义搜索。         最后，查询引擎促进了与索引数据的交互，允许用户提出问题并利用 LLM 和检索到的信息合成答案。
         --- 以下是与 LlamaIndex 主题不太直接相关的内容 ---
         此外，Python 作为一门通用编程语言，其简洁性和丰富的库生态使其在 AI 领域广受欢迎。例如，NumPy 和 Pandas 是数据处理的基础，它们提供了强大的工具用于数值操作和结构化数据。
		 Scikit-learn 则提供了全面的机器学习算法套件，适用于分类、回归和聚类等任务。
		 这些工具共同构成了数据科学家和 AI 从业者的强大工具箱，使他们能够高效地开发和部署复杂的 AI 模型。
         --- 以下是另一个相关但概念上独立的部分 ---
         句子窗口切片是一种高级的切片策略，它在每个切片中包含一个目标句子，并在其前后添加一定数量的“窗口”句子作为上下文。这种方法旨在检索时为 LLM 提供丰富的局部上下文，从而提高生成答案的连贯性。         语义切片则尝试根据文本的语义内容来划分段落，而不是仅仅依靠固定的字符数或句子数量。它利用嵌入模型计算句子或短语之间的语义相似度，识别出主题或含义发生自然转变的断点。这两种高级方法都能有效提升 RAG 应用的召回和生成质量。选择正确的切片策略通常取决于数据的具体特征和预期的查询类型。

"""


def test_llama_splitters():
    """测试基于LlamaIndex的分割器"""
    print("=== 测试基于LlamaIndex的分割器 ===")
    
    # 测试文本
    test_text = """这是一个测试文本。它包含多个句子，用于测试不同的分割器。
    分割器应该能够根据不同的策略将文本分割成合适的块。
    例如，句子分割器会根据句子边界进行分割，而语义分割器会尝试保持语义的完整性。
    句子窗口分割器会为每个句子创建一个窗口，包含前后的上下文。
    混合分割器则会结合多种分割策略，以获得更好的效果。
    """
    
    # 测试分割器类型
    splitter_types = [
        "llama_sentence",
        "llama_sentence_window",
        "llama_semantic",
        "llama_combined"
    ]
    
    factory = TextSplitterStrategyFactory()
    
    for splitter_type in splitter_types:
        print(f"\n测试 {splitter_type}:")
        
        try:
            # 创建分割器
            splitter = SplitterFactory.create_splitter(
                splitter_type=splitter_type,
                chunk_size=200,
                chunk_overlap=50
            )
            print(f"  分割器创建成功: {type(splitter).__name__}")
            
            # 测试split_text方法
            chunks = splitter.split_text(test_text)
            print(f"  分割结果: {len(chunks)} 个块")
            for i, chunk in enumerate(chunks[:2]):  # 只显示前2个块
                print(f"    块 {i+1}: {chunk[:100]}...")
            
            # 测试策略创建
            strategy = factory.create_strategy(
                text_splitter=splitter,
                min_length=50
            )
            print(f"  策略创建成功: {type(strategy).__name__}")
            
        except Exception as e:
            print(f"  测试失败: {str(e)}")
    
    print("\n=== 测试完成 ===")


def test_llama_semantic_splitter():
    """专门测试LlamaSemanticSplitter的效果"""
    print("\n=== 专门测试LlamaSemanticSplitter ===")
    
    # 测试文本（包含多个语义相关的段落）
    try:
        # 创建语义分割器
        splitter = SplitterFactory.create_splitter(
            splitter_type="llama_semantic",
            chunk_size=100,
            chunk_overlap=10
        )
        print(f"分割器创建成功: {type(splitter).__name__}")
        
        # 测试split_text方法
        chunks = splitter.split_text(test_text)
        print(f"分割结果: {len(chunks)} 个块")
        
        # 显示所有分割结果
        for i, chunk in enumerate(chunks):
            print(f"\n块 {i+1}:")
            print(f"{chunk.strip()}")
        
        # 测试策略创建
        factory = TextSplitterStrategyFactory()
        strategy = factory.create_strategy(
            text_splitter=splitter,
            min_length=50
        )
        print(f"\n策略创建成功: {type(strategy).__name__}")
        
        print("\n=== 语义分割器测试完成 ===")
        return True
    except Exception as e:
        print(f"测试失败: {str(e)}")
        return False


def test_llama_combined_splitter():
    """专门测试LlamaCombinedSplitter的效果"""
    print("\n=== 专门测试LlamaCombinedSplitter ===")
    
    try:
        # 创建组合分割器
        splitter = SplitterFactory.create_splitter(
            splitter_type="llama_combined",
            chunk_size=200,
            chunk_overlap=30
        )
        print(f"分割器创建成功: {type(splitter).__name__}")
        
        # 测试split_text方法
        chunks = splitter.split_text(test_text)
        print(f"分割结果: {len(chunks)} 个块")
        
        # 显示所有分割结果
        for i, chunk in enumerate(chunks):
            print(f"\n块 {i+1}:")
            print(f"{chunk.strip()}")
        
        # 测试策略创建
        factory = TextSplitterStrategyFactory()
        strategy = factory.create_strategy(
            text_splitter=splitter,
            min_length=50
        )
        print(f"\n策略创建成功: {type(strategy).__name__}")
        
        print("\n=== 组合分割器测试完成 ===")
        return True
    except Exception as e:
        print(f"测试失败: {str(e)}")
        return False


if __name__ == "__main__":
    # test_llama_splitters()
    # test_llama_semantic_splitter()
    test_llama_combined_splitter()