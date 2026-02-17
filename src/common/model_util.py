import os

from huggingface_hub import snapshot_download

from src.common.file_utils import get_project_base_directory

def load_model(model_name: str)-> str:
    """
    加载模型的公共方法

    Args:
        model_name (str): 模型名称或路径

    Returns:
        SentenceTransformer: 加载的模型实例

    Raises:
        RuntimeError: 加载模型失败时抛出异常
    """
    try:
        # 构建本地模型路径
        model_dir = os.path.join(get_project_base_directory(), "resource_package/models", model_name)

        # 检查本地模型是否存在
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            # 本地不存在，下载模型到本地
            model_dir = snapshot_download(
                repo_id=model_name,
                local_dir=model_dir,
                endpoint="https://hf-mirror.com"
            )

        return model_dir
    except Exception as e:
        raise RuntimeError(f"Failed to load or download SentenceTransformer model '{model_name}': {str(e)}")

def load_file_from_repo(repo_id: str, filename: str) -> str:
    """
    从Hugging Face仓库加载指定文件的公共方法

    Args:
        repo_id (str): Hugging Face仓库ID (格式: "username/repo_name")
        filename (str): 要下载的文件名

    Returns:
        str: 本地文件的完整路径

    Raises:
        RuntimeError: 下载文件失败时抛出异常
    """
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    try:
        # 构建本地文件路径
        base_dir = get_project_base_directory()
        file_dir = os.path.join(base_dir, "rag/res/deepdoc", repo_id.replace("/", "_"))
        file_path = os.path.join(file_dir, filename)

        # 检查本地文件是否存在
        if not os.path.exists(file_path):
            os.makedirs(file_dir, exist_ok=True)
            # 本地不存在，从仓库下载指定文件
            from huggingface_hub import hf_hub_download

            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=file_dir,
                local_dir_use_symlinks=False
            )

        return file_path
    except Exception as e:
        raise RuntimeError(f"Failed to download file '{filename}' from repo '{repo_id}': {str(e)}")

class ModelUtils:
    pass
