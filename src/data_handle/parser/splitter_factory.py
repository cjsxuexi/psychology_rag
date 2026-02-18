import os
from src.data_handle.parser.splitters import (
    RecursiveSplitter, 
    CharacterSplitter, 
    TokenSplitter,
    LlamaSentenceSplitter,
    LlamaSentenceWindowSplitter,
    LlamaSemanticSplitter,
    LlamaCombinedSplitter
)


class SplitterFactory:
    """分割器工厂类，根据配置创建不同的分割器实例"""
    
    @staticmethod
    def create_splitter(
        splitter_type: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        创建分割器实例
        
        Args:
            splitter_type: 分割器类型，可选值：recursive, character, token, llama_sentence, llama_sentence_window, llama_semantic
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
            
        Returns:
            BaseSplitter: 分割器实例
        """
        # 从环境变量获取配置，优先使用传入的参数
        if splitter_type is None:
            splitter_type = os.getenv("TEXT_SPLITTER_TYPE", "recursive")
        
        if chunk_size is None:
            chunk_size = int(os.getenv("TEXT_CHUNK_SIZE", 1000))
        
        if chunk_overlap is None:
            chunk_overlap = int(os.getenv("TEXT_CHUNK_OVERLAP", 100))
        
        # 根据类型创建对应的分割器
        if splitter_type == "recursive":
            return RecursiveSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif splitter_type == "character":
            return CharacterSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif splitter_type == "token":
            return TokenSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif splitter_type == "llama_sentence":
            return LlamaSentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif splitter_type == "llama_sentence_window":
            return LlamaSentenceWindowSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif splitter_type == "llama_semantic":
            return LlamaSemanticSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif splitter_type == "llama_combined":
            return LlamaCombinedSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            raise ValueError(f"不支持的分割器类型: {splitter_type}")