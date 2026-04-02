from src.config import get_settings, SplitterType
from src.data_handle.parser.splitters import (
    RecursiveSplitter,
    CharacterSplitter,
    TokenSplitter,
    LlamaSentenceSplitter,
    LlamaSentenceWindowSplitter,
    LlamaSemanticSplitter,
    LlamaCombinedSplitter
)


# 分割器类型到类的映射表
SPLITTER_MAP = {
    SplitterType.RECURSIVE: RecursiveSplitter,
    SplitterType.CHARACTER: CharacterSplitter,
    SplitterType.TOKEN: TokenSplitter,
    SplitterType.LLAMA_SENTENCE: LlamaSentenceSplitter,
    SplitterType.LLAMA_SENTENCE_WINDOW: LlamaSentenceWindowSplitter,
    SplitterType.LLAMA_SEMANTIC: LlamaSemanticSplitter,
    SplitterType.LLAMA_COMBINED: LlamaCombinedSplitter,
}


def create_splitter(
    splitter_type: SplitterType = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
    window_size: int = None
):
    """
    创建分割器实例

    Args:
        splitter_type: 分割器类型，使用 SplitterType 枚举
        chunk_size: 文本块大小
        chunk_overlap: 文本块重叠大小
        window_size: 窗口大小（用于sentence_window类型）

    Returns:
        BaseSplitter: 分割器实例

    Raises:
        ValueError: 不支持的分割器类型
    """
    settings = get_settings()
    splitter_config = settings.splitter

    # 使用传入参数或配置默认值
    if splitter_type is None:
        splitter_type = splitter_config.type
    if chunk_size is None:
        chunk_size = splitter_config.chunk_size
    if chunk_overlap is None:
        chunk_overlap = splitter_config.chunk_overlap

    # 从映射表获取分割器类
    splitter_class = SPLITTER_MAP.get(splitter_type)
    if splitter_class is None:
        raise ValueError(
            f"不支持的分割器类型: {splitter_type}. "
            f"支持的类型: {list(SPLITTER_MAP.keys())}"
        )

    # 特殊处理需要window_size的类型
    if splitter_type == SplitterType.LLAMA_SENTENCE_WINDOW:
        return splitter_class(window_size=window_size)

    # 标准参数创建
    return splitter_class(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
