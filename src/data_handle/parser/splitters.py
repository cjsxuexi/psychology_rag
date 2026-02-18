import re
from abc import ABC, abstractmethod

from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
from llama_index.core.node_parser import (
    SentenceSplitter,
    SentenceWindowNodeParser,
    SemanticSplitterNodeParser
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.common.model_util import load_model
from src.utils.logger import logger


class BaseSplitter(ABC):
    """基础分割器接口"""

    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        """分割文本"""
        pass

    @abstractmethod
    def split_documents(self, documents: list) -> list:
        """分割文档"""
        pass


class RecursiveSplitter(BaseSplitter):
    """递归字符分割器"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "，", " "]  # 优先按语义分割
        )

    def split_text(self, text: str) -> list[str]:
        return self.splitter.split_text(text)

    def split_documents(self, documents: list) -> list:
        return self.splitter.split_documents(documents)


class CharacterSplitter(BaseSplitter):
    """字符分割器"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    def split_text(self, text: str) -> list[str]:
        return self.splitter.split_text(text)

    def split_documents(self, documents: list) -> list:
        return self.splitter.split_documents(documents)


class TokenSplitter(BaseSplitter):
    """令牌分割器"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_text(self, text: str) -> list[str]:
        return self.splitter.split_text(text)

    def split_documents(self, documents: list) -> list:
        return self.splitter.split_documents(documents)


from llama_index.core import Document


class LlamaSentenceSplitter(BaseSplitter):
    """LlamaIndex句子分割器"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_text(self, text: str) -> list[str]:
        # 创建LlamaIndex Document对象
        doc = Document(text=text, id_="test_doc")
        nodes = self.splitter.get_nodes_from_documents([doc])
        return [node.text for node in nodes]

    def split_documents(self, documents: list) -> list:
        # 将输入文档转换为LlamaIndex Document对象
        llama_docs = []
        for i, doc in enumerate(documents):
            if isinstance(doc, dict) and "text" in doc:
                llama_docs.append(Document(text=doc["text"], id_=f"doc_{i}"))
            else:
                llama_docs.append(doc)

        nodes = self.splitter.get_nodes_from_documents(llama_docs)
        return nodes


class LlamaSentenceWindowSplitter(BaseSplitter):
    """LlamaIndex句子窗口分割器"""

    def __init__(self, window_size: int = 3):
        self.splitter = SentenceWindowNodeParser(
            window_size=window_size
        )

    def split_text(self, text: str) -> list[str]:
        # 创建LlamaIndex Document对象
        doc = Document(text=text, id_="test_doc")
        nodes = self.splitter.get_nodes_from_documents([doc])
        return [node.text for node in nodes]

    def split_documents(self, documents: list) -> list:
        # 将输入文档转换为LlamaIndex Document对象
        llama_docs = []
        for i, doc in enumerate(documents):
            if isinstance(doc, dict) and "text" in doc:
                llama_docs.append(Document(text=doc["text"], id_=f"doc_{i}"))
            else:
                llama_docs.append(doc)

        nodes = self.splitter.get_nodes_from_documents(llama_docs)
        return nodes


class LlamaSemanticSplitter(BaseSplitter):
    """LlamaIndex语义分割器"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        # 加载本地模型
        model_path = load_model("infgrad/stella-large-zh-v2")

        # 使用本地模型创建嵌入模型
        embed_model = HuggingFaceEmbedding(
            model_name=model_path,
            embed_batch_size=8
        )

        self.splitter = SemanticSplitterNodeParser(
            embed_model=embed_model,
            breakpoint_percentile_threshold=95,  # 值越大分割越细
            sentence_splitter=self._chinese_sentence_tokenizer
        )

    def split_text(self, text: str) -> list[str]:
        # 创建LlamaIndex Document对象
        doc = Document(text=text, id_="test_doc")
        nodes = self.splitter.get_nodes_from_documents([doc])
        return [node.text for node in nodes]

    def split_documents(self, documents: list) -> list:
        # 将输入文档转换为LlamaIndex Document对象
        llama_docs = []
        for i, doc in enumerate(documents):
            if isinstance(doc, dict) and "text" in doc:
                llama_docs.append(Document(text=doc["text"], id_=f"doc_{i}"))
            else:
                llama_docs.append(doc)

        nodes = self.splitter.get_nodes_from_documents(llama_docs)
        return nodes

    # --- 3. 定义一个能处理中文的自定义分句函数 ---
    # 这个函数本身就是 SemanticSplitterNodeParser 所需要的"句子切分器"
    @staticmethod
    def _chinese_sentence_tokenizer(text: str) -> list[str]:
        sentences = re.findall(r'[^。！？…\n]+[。！？…\n]?', text)
        return [s.strip() for s in sentences if s.strip()]


class LlamaCombinedSplitter(BaseSplitter):
    """LlamaIndex组合分割器：使用语义切分作为主要方式，句子切分作为二次切分方式"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.max_chunk_size = chunk_size

        # 加载本地模型用于语义切分
        model_path = load_model("infgrad/stella-large-zh-v2")
        embed_model = HuggingFaceEmbedding(
            model_name=model_path,
            embed_batch_size=10
        )

        # 初始化语义分割器作为主要分割器
        self.primary_parser = SemanticSplitterNodeParser(
            embed_model=embed_model,
            breakpoint_percentile_threshold=95,  # 值越大分割越细
            sentence_splitter=self._chinese_sentence_tokenizer
        )

        # 初始化句子分割器作为次要分割器（用于处理过大的块）
        self.secondary_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator="\n\n",
        )

        # 初始化tokenizer用于计算大小
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def split_text(self, text: str) -> list[str]:

        # 打印输入文本
        logger.debug(f"\n{'=' * 50} 开始处理文本 {'=' * 50}")
        logger.debug(f"输入文本长度: {len(text)} 字符")
        logger.debug(f"输入文本内容:")
        logger.debug(f"{text[:500]}..." if len(text) > 500 else text)

        # 1. 首先使用语义分割器进行初步分割
        doc = Document(text=text, id_="test_doc")
        primary_nodes = self.primary_parser.get_nodes_from_documents([doc])

        logger.info(f"语义分割结果: {len(primary_nodes)} 个语义块")

        if not primary_nodes:
            logger.warning("语义分割未产生任何结果")
            return []

        # 2. 检查每个语义分割后的块大小，过大的块使用句子分割器进行二次分割
        final_chunks = []
        for i, node in enumerate(primary_nodes):
            node_text = node.text
            node_size = len(self.tokenizer.tokenize(node_text))

            logger.debug(f"\n处理语义块 {i + 1}:")
            logger.debug(f"  大小: {node_size} tokens")
            logger.debug(f"  内容: {node_text[:200]}..." if len(node_text) > 200 else node_text)

            if node_size <= self.max_chunk_size:
                # 大小合适，直接采纳
                logger.info(f"  结果: 大小合适 (<= {self.max_chunk_size} tokens)，直接采纳")
                final_chunks.append(node_text)
            else:
                # 过大，使用句子分割器进行二次分割
                logger.info(f"  结果: 过大 (> {self.max_chunk_size} tokens)，进行二次分割")
                sub_nodes = self.secondary_parser.get_nodes_from_documents([Document(text=node_text)])
                logger.info(f"  二次分割结果: {len(sub_nodes)} 个子块")
                final_chunks.extend([sub_node.text for sub_node in sub_nodes])

        logger.info(f"\n最终分割结果: {len(final_chunks)} 个块")
        logger.debug(f"{'=' * 50} 文本处理完成 {'=' * 50}")

        return final_chunks

    def split_documents(self, documents: list) -> list:
        # 将输入文档转换为LlamaIndex Document对象
        llama_docs = []
        for i, doc in enumerate(documents):
            if isinstance(doc, dict) and "text" in doc:
                llama_docs.append(Document(text=doc["text"], id_=f"doc_{i}"))
            else:
                llama_docs.append(doc)

        # 1. 首先使用语义分割器进行初步分割
        primary_nodes = self.primary_parser.get_nodes_from_documents(llama_docs)

        if not primary_nodes:
            return []

        # 2. 检查每个语义分割后的块大小，过大的块使用句子分割器进行二次分割
        final_nodes = []
        for i, node in enumerate(primary_nodes):
            node_text = node.text
            node_size = len(self.tokenizer.tokenize(node_text))

            if node_size <= self.max_chunk_size:
                # 大小合适，直接采纳
                final_nodes.append(node)
            else:
                # 过大，使用句子分割器进行二次分割
                sub_nodes = self.secondary_parser.get_nodes_from_documents([Document(text=node_text)])
                # 为每个子节点设置适当的ID
                for j, sub_node in enumerate(sub_nodes):
                    sub_node.id_ = f"{node.id_}_sub_{j}"
                    final_nodes.append(sub_node)

        return final_nodes

    # 使用与LlamaSemanticSplitter相同的中文分句函数
    @staticmethod
    def _chinese_sentence_tokenizer(text: str) -> list[str]:
        sentences = re.findall(r'[^。！？…\n]+[。！？…\n]?', text)
        return [s.strip() for s in sentences if s.strip()]
