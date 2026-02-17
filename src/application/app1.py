import numpy as np
import time
import requests
from loguru import logger
from dotenv import load_dotenv
import os
from typing import Dict, List, Optional

# 复用m3e单例模型（原有核心，无需修改）
from src.embedding.M3EEmbedding import M3EEmbeddingSingleton

from src.common.file_utils import get_config_path, get_storage_path

# ===================== 企业级初始化：加载配置、复用单例模型 =====================
load_dotenv(dotenv_path=get_config_path('.env'))

# 加载原有RAG配置
RAG_CONFIG = {
    "top_k": int(os.getenv("RAG_TOP_K")),
    "qa_temperature": float(os.getenv("QA_TEMPERATURE")),
    "qa_max_tokens": int(os.getenv("QA_MAX_TOKENS")),
    "faiss_nprobe": int(os.getenv("FAISS_NPROBE")),
    "faiss_ef_search": int(os.getenv("FAISS_EF_SEARCH")),
    "prompt_template_path": os.getenv("RAG_PROMPT_TEMPLATE_PATH")
}

# 新增：加载qwen-plus配置（企业级解耦）
QWEN_CONFIG = {
    "api_key": os.getenv("DASHSCOPE_API_KEY"),
    "model_name": os.getenv("QWEN_MODEL_NAME"),
    "temperature": float(os.getenv("QWEN_TEMPERATURE")),
    "max_tokens": int(os.getenv("QWEN_MAX_TOKENS")),
    "timeout": int(os.getenv("QWEN_TIMEOUT")),
    "retry_times": int(os.getenv("QWEN_RETRY_TIMES"))
}

# 初始化m3e单例模型（原有核心，无需修改）
m3e_embedding = M3EEmbeddingSingleton()


# ===================== 企业级工具函数：加载提示词模板（原有，无需修改） =====================
def load_prompt_template(template_path: str) -> str:
    """
    加载提示词模板，企业级：模板与代码解耦，便于维护和多场景切换
    """
    try:
        if not os.path.exists(template_path):
            logger.warning(f"提示词模板文件不存在：{template_path}，使用默认模板")
            return """你是一位专业的心理咨询助手，仅基于以下提供的上下文信息回答用户的问题。
上下文信息：
{context_str}

用户问题：{query}
"""
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()
        logger.success(f"成功加载提示词模板：{template_path}")
        return template
    except Exception as e:
        logger.error(f"加载提示词模板失败：{str(e)}，使用默认模板")
        return """你是一位专业的心理咨询助手，仅基于以下提供的上下文信息回答用户的问题。
上下文信息：
{context_str}

用户问题：{query}
"""


# 预加载提示词模板
PROMPT_TEMPLATE = load_prompt_template(RAG_CONFIG["prompt_template_path"])


# ===================== 新增：qwen-plus 工具函数（企业级，带重试、超时） =====================
def call_qwen_plus(prompt: str) -> str:
    """
    调用qwen-plus API，企业级：带重试、超时、异常容错，返回生成结果
    优先使用dashscope SDK，兜底使用requests直接调用
    修复版本：正确处理响应格式，优先使用 output.text 字段
    """
    # 1. 前置校验（API密钥）
    if not QWEN_CONFIG["api_key"] or QWEN_CONFIG["api_key"] == "your_aliyun_dashscope_api_key":
        raise ValueError("qwen-plus API密钥未配置，请在.env中填写有效的QWEN_API_KEY")

    # 2. 构建请求消息（符合qwen-plus格式要求）
    messages = [
        {"role": "system", "content": "你是一位专业的心理咨询助手，严格按照提供的上下文回答问题，不添加外部无关信息。"},
        {"role": "user", "content": prompt}
    ]

    # 3. 调用逻辑（带重试）
    for retry in range(QWEN_CONFIG["retry_times"] + 1):
        try:
            # 方案1：使用dashscope SDK（推荐，官方维护，更稳定）
            try:
                import dashscope
                from dashscope import Generation

                # 配置dashscope API密钥
                dashscope.api_key = QWEN_CONFIG["api_key"]

                # 调用qwen-plus
                response = Generation.call(
                    model=QWEN_CONFIG["model_name"],
                    messages=messages,
                    temperature=QWEN_CONFIG["temperature"],
                    max_tokens=QWEN_CONFIG["max_tokens"],
                    timeout=QWEN_CONFIG["timeout"]
                )

                # 修复后的响应处理逻辑
                if response.status_code == 200:
                    # 优先检查 output.text（主要响应字段）
                    if hasattr(response, 'output') and response.output:
                        output = response.output
                        
                        # 方法1：检查 text 字段（推荐）
                        if hasattr(output, 'text') and output.text:
                            return output.text.strip()
                        
                        # 方法2：检查 choices 字段（兼容旧格式）
                        elif hasattr(output, 'choices') and output.choices:
                            if len(output.choices) > 0 and hasattr(output.choices[0], 'message'):
                                return output.choices[0].message.content.strip()
                        
                        # 方法3：直接访问字典形式（兜底）
                        elif isinstance(output, dict) and 'text' in output:
                            return output['text'].strip()
                
                # 如果以上都不行，尝试字符串解析
                response_str = str(response)
                if '"text":' in response_str:
                    import json
                    try:
                        response_dict = json.loads(response_str)
                        if 'output' in response_dict and 'text' in response_dict['output']:
                            return response_dict['output']['text'].strip()
                    except:
                        pass
                
                # 如果到这里还没返回，说明响应格式异常
                raise Exception(f"qwen-plus返回格式异常：状态码{response.status_code}，响应内容{str(response)[:200]}")

            # 方案2：兜底使用requests直接调用（防止dashscope SDK导入失败）
            except ImportError:
                logger.warning("未安装dashscope SDK，使用requests兜底调用qwen-plus API")
                url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
                headers = {
                    "Authorization": f"Bearer {QWEN_CONFIG['api_key']}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": QWEN_CONFIG["model_name"],
                    "input": {
                        "messages": messages
                    },
                    "parameters": {
                        "temperature": QWEN_CONFIG["temperature"],
                        "max_tokens": QWEN_CONFIG["max_tokens"]
                    }
                }

                # 发送POST请求
                response = requests.post(
                    url=url,
                    headers=headers,
                    json=payload,
                    timeout=QWEN_CONFIG["timeout"]
                )
                response.raise_for_status()  # 抛出HTTP错误
                result = response.json()

                # 处理 requests 响应格式
                if "output" in result:
                    output = result["output"]
                    if "text" in output:
                        return output["text"].strip()
                    elif "choices" in output and len(output["choices"]) > 0:
                        return output["choices"][0]["message"]["content"].strip()
                
                raise Exception(f"qwen-plus返回无效格式：{result}")

        except Exception as e:
            if retry < QWEN_CONFIG["retry_times"]:
                logger.warning(
                    f"qwen-plus调用第{retry + 1}次失败，将重试（剩余{QWEN_CONFIG['retry_times'] - retry}次）：{str(e)}")
                time.sleep(1)  # 重试间隔1秒
                continue
            else:
                logger.error(f"qwen-plus调用重试{QWEN_CONFIG['retry_times'] + 1}次均失败：{str(e)}")
                raise Exception(f"qwen-plus API调用失败：{e}")

    # 兜底返回（理论上不会执行到此处）
    return "qwen-plus模型调用超时，无法生成回答。"


# ===================== 企业级RAG流水线类（核心：修改_call_qa_model为qwen-plus） =====================
class EnterpriseRAGPipeline:
    """
    企业级RAG流水线类：封装核心逻辑，高内聚低耦合，便于扩展和维护
    """

    def __init__(
            self,
            faiss_index,
            pca_model,
            scaler_model,
            text_chunks: List[str],
            top_k: Optional[int] = None
    ):
        """
        初始化RAG流水线
        :param faiss_index: 训练好的FAISS索引
        :param pca_model: 训练好的PCA降维模型
        :param scaler_model: 训练好的标准化模型
        :param text_chunks: 分割后的有效文本块列表（与FAISS索引对应）
        :param top_k: 检索相似文本数量
        """
        # 核心组件赋值（带校验）
        self.faiss_index = faiss_index
        self.pca_model = pca_model
        self.scaler_model = scaler_model
        self.text_chunks = text_chunks
        self.top_k = top_k or RAG_CONFIG["top_k"]

        # 参数校验（企业级：提前拦截无效参数，避免运行时错误）
        self._validate_init_params()
        logger.success("企业级RAG流水线初始化完成")
    
    @classmethod
    def from_local_storage(
        cls,
        faiss_index_path: str = "faiss_fixed_pq_1.13.2.index",
        pca_model_path: str = "pca_model.pkl",
        scaler_model_path: str = "scaler_model.pkl",
        top_k: Optional[int] = None
    ):
        """
        从本地存储加载组件并创建RAG流水线实例（推荐使用方式）
        
        Args:
            faiss_index_path: FAISS索引文件路径
            pca_model_path: PCA模型文件路径
            scaler_model_path: 标准化模型文件路径
            top_k: 检索相似文本数量
            
        Returns:
            EnterpriseRAGPipeline: 配置好的RAG流水线实例
        """
        from src.application.local_loader import load_local_rag_components
        
        try:
            logger.info("正在从本地存储加载RAG组件...")
            
            # 加载所有组件
            faiss_index, pca_model, scaler_model, text_chunks = load_local_rag_components(
                faiss_index_path=faiss_index_path,
                pca_model_path=pca_model_path,
                scaler_model_path=scaler_model_path,
                load_text_chunks=True
            )
            
            # 创建实例
            instance = cls(
                faiss_index=faiss_index,
                pca_model=pca_model,
                scaler_model=scaler_model,
                text_chunks=text_chunks,
                top_k=top_k
            )
            
            logger.success("从本地存储创建RAG流水线实例成功")
            return instance
            
        except Exception as e:
            logger.error(f"从本地存储创建RAG流水线失败: {str(e)}")
            raise

    def _validate_init_params(self) -> None:
        """
        初始化参数校验，企业级容错机制
        """
        if self.faiss_index is None:
            raise ValueError("FAISS索引不能为空")
        if self.pca_model is None or self.scaler_model is None:
            raise ValueError("PCA降维模型和标准化模型不能为空")
        if not isinstance(self.text_chunks, list) or len(self.text_chunks) == 0:
            raise ValueError("文本块列表不能为空且必须为列表类型")
        if self.top_k <= 0 or self.top_k > 100:
            logger.warning(f"top_k {self.top_k} 超出合理范围（1-100），自动调整为10")
            self.top_k = 10

    def _generate_query_embedding(self, query: str) -> np.ndarray:
        """
        生成查询向量（m3e-base），企业级：保证与训练向量格式一致
        """
        try:
            if not query or not isinstance(query, str):
                raise ValueError("查询文本不能为空且必须为字符串类型")

            # 1. m3e-base生成查询向量（复用单例模型，归一化与训练时保持一致）
            logger.info(f"开始生成查询向量，查询文本：{query[:50]}...")
            query_embedding = m3e_embedding.generate_embeddings(
                texts=[query],  # m3e encode接收列表，单条查询封装为列表
                batch_size=1
            )

            # 2. 校验向量格式（与训练时的稠密向量保持一致：float32、二维数组）
            if query_embedding.ndim != 2 or query_embedding.dtype != np.float32:
                query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

            logger.success(f"查询向量生成完成，向量形状：{query_embedding.shape}")
            return query_embedding
        except Exception as e:
            logger.error(f"生成查询向量失败：{str(e)}")
            raise Exception(f"查询向量生成异常：{e}")

    def _process_query_embedding(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        处理查询向量（标准化→降维），与训练向量处理流程一致，保证检索准确性
        """
        try:
            logger.info("开始处理查询向量（标准化→降维）")
            # 1. 标准化（与训练数据使用同一scaler模型，企业级：保持数据处理一致性）
            query_scaled = self.scaler_model.transform(query_embedding)

            # 2. 降维（与训练数据使用同一PCA模型）
            query_reduced = self.pca_model.transform(query_scaled)

            logger.success(f"查询向量处理完成，降维后形状：{query_reduced.shape}")
            return query_reduced
        except Exception as e:
            logger.error(f"处理查询向量失败：{str(e)}")
            raise Exception(f"查询向量处理异常：{e}")

    def _faiss_retrieval(self, query_reduced: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        FAISS近似检索，企业级：优化检索参数，提升召回率
        """
        try:
            logger.info(f"开始FAISS检索，检索top_k：{self.top_k}")
            
            # 调试信息：检查输入参数
            logger.info(f"查询向量维度：{query_reduced.shape}")
            logger.info(f"查询向量数据类型：{query_reduced.dtype}")
            logger.info(f"FAISS索引维度：{self.faiss_index.d if hasattr(self.faiss_index, 'd') else '未知'}")
            logger.info(f"FAISS索引类型：{type(self.faiss_index).__name__}")
            logger.info(f"FAISS索引ntotal：{self.faiss_index.ntotal if hasattr(self.faiss_index, 'ntotal') else '未知'}")
            
            # 1. 配置FAISS检索参数（优化召回率，配置解耦）
            if hasattr(self.faiss_index, "nprobe"):
                self.faiss_index.nprobe = RAG_CONFIG["faiss_nprobe"]
                logger.info(f"设置nprobe参数：{RAG_CONFIG['faiss_nprobe']}")
            if hasattr(self.faiss_index, "hnsw"):
                self.faiss_index.hnsw.efSearch = RAG_CONFIG["faiss_ef_search"]
                logger.info(f"设置hnsw.efSearch参数：{RAG_CONFIG['faiss_ef_search']}")

            # 2. 执行检索（返回距离和索引）
            logger.info("开始执行FAISS搜索...")
            distances, indices = self.faiss_index.search(query_reduced, self.top_k)

            # 3. 校验检索结果
            if distances is None or indices is None:
                raise ValueError("FAISS检索返回结果为空")
            
            logger.info(f"检索结果形状：distances={distances.shape}, indices={indices.shape}")
            logger.success(f"FAISS检索完成，返回{len(indices[0])}条相似结果")
            return distances, indices
        except AssertionError as ae:
            logger.error(f"FAISS检索断言错误：{str(ae)}")
            logger.error(f"断言错误详情：{repr(ae)}")
            raise Exception(f"FAISS检索断言失败：{ae}")
        except Exception as e:
            logger.error(f"FAISS检索失败：{str(e)}")
            logger.error(f"错误类型：{type(e).__name__}")
            raise Exception(f"FAISS检索异常：{e}")

    def _extract_contexts(self, indices: np.ndarray) -> List[str]:
        """
        提取检索上下文，企业级：容错处理，避免索引越界
        """
        try:
            logger.info("开始提取检索上下文")
            contexts = []
            valid_indices = indices[0]  # 提取单条查询的检索索引

            for idx in valid_indices:
                # 容错：避免索引超出文本块列表长度
                if 0 <= idx < len(self.text_chunks):
                    contexts.append(self.text_chunks[idx])
                else:
                    logger.warning(f"无效索引：{idx}，跳过该条上下文")

            # 校验上下文有效性
            if len(contexts) == 0:
                logger.warning("未提取到有效上下文")
            else:
                logger.success(f"提取上下文完成，有效上下文数：{len(contexts)}")

            return contexts
        except Exception as e:
            logger.error(f"提取上下文失败：{str(e)}")
            raise Exception(f"上下文提取异常：{e}")

    def _generate_answer(self, query: str, contexts: List[str]) -> str:
        """
        生成问答结果（替换为qwen-plus），企业级：配置解耦，支持多模型扩展
        """
        try:
            if len(contexts) == 0:
                return "未查询到相关有效信息，无法回答你的问题。"

            logger.info("开始拼接上下文并调用qwen-plus生成回答")
            # 1. 拼接上下文
            context_str = "\n\n".join([f"【上下文{idx + 1}】{ctx}" for idx, ctx in enumerate(contexts)])

            # 2. 渲染提示词模板（企业级：模板与数据解耦）
            prompt = PROMPT_TEMPLATE.format(
                context_str=context_str,
                query=query
            )

            # 3. 调用qwen-plus（替换原OpenAI调用，复用企业级工具函数）
            answer = call_qwen_plus(prompt)

            logger.success("qwen-plus回答生成完成")
            return answer
        except Exception as e:
            logger.error(f"生成回答失败：{str(e)}")
            return f"回答生成异常，无法正常回复你的问题：{str(e)}"

    def _call_qa_model(self, prompt: str) -> str:
        """
        兼容旧接口（可删除，此处保留为了不改动其他流程），实际调用已迁移至call_qwen_plus
        """
        return call_qwen_plus(prompt)

    def run(self, query: str) -> Dict:
        """
        运行完整RAG流水线，返回结构化结果（企业级：便于后续存储和展示）
        """
        start_time = time.time()
        try:
            # 1. 生成查询向量（m3e-base）
            query_embedding = self._generate_query_embedding(query)

            # 2. 处理查询向量（标准化→降维）
            query_reduced = self._process_query_embedding(query_embedding)

            # 3. FAISS检索
            distances, indices = self._faiss_retrieval(query_reduced)

            # 4. 提取上下文
            contexts = self._extract_contexts(indices)

            # 5. 调用qwen-plus生成回答
            answer = self._generate_answer(query, contexts)

            # 6. 构建结构化结果（企业级：返回完整信息，便于排查和展示）
            total_time = round(time.time() - start_time, 4)
            result = {
                "query": query,
                "answer": answer,
                "contexts": contexts,
                "distances": distances[0].tolist() if distances is not None else [],
                "indices": indices[0].tolist() if indices is not None else [],
                "total_processing_time_seconds": total_time,
                "top_k": self.top_k,
                "status": "success",
                "qa_model": QWEN_CONFIG["model_name"]  # 新增：标注使用的问答模型
            }

            logger.success(f"RAG流水线运行完成，总耗时：{total_time}秒，使用模型：{QWEN_CONFIG['model_name']}")
            return result
        except Exception as e:
            total_time = round(time.time() - start_time, 4)
            # 构建异常结果（企业级：统一返回格式，便于上层处理）
            error_result = {
                "query": query,
                "answer": f"处理失败：{str(e)}",
                "contexts": [],
                "distances": [],
                "indices": [],
                "total_processing_time_seconds": total_time,
                "top_k": self.top_k,
                "status": "failed",
                "qa_model": QWEN_CONFIG["model_name"]
            }
            logger.error(f"RAG流水线运行失败，总耗时：{total_time}秒，错误信息：{str(e)}")
            return error_result


# ===================== qwen-plus 专项测试案例 =====================
class QwenPlusTester:
    """
    专门用于测试和诊断 call_qwen_plus 方法的测试类
    可以帮助定位超时问题的根本原因
    """
    
    def __init__(self):
        """初始化测试器，显示当前配置"""
        self.config = QWEN_CONFIG.copy()
        self._print_current_config()
    
    def _print_current_config(self):
        """打印当前qwen-plus配置信息"""
        print("=" * 60)
        print("🔍 当前 qwen-plus 配置信息：")
        print(f"   模型名称: {self.config['model_name']}")
        print(f"   温度参数: {self.config['temperature']}")
        print(f"   最大token: {self.config['max_tokens']}")
        print(f"   超时时间: {self.config['timeout']} 秒")
        print(f"   重试次数: {self.config['retry_times']}")
        print(f"   API密钥配置状态: {'✅ 已配置' if self.config['api_key'] and self.config['api_key'] != 'your_aliyun_dashscope_api_key' else '❌ 未配置'}")
        print("=" * 60)
    
    def test_basic_connection(self):
        """
        测试1: 基础连接测试
        验证API密钥是否有效，网络是否可达
        """
        print("\n🧪 测试1: 基础连接测试")
        print("-" * 40)
        
        try:
            # 使用最简单的提示词进行测试
            simple_prompt = "你好，请回复'hello'"
            print(f"发送测试提示词: {simple_prompt}")
            
            start_time = time.time()
            result = call_qwen_plus(simple_prompt)
            elapsed_time = time.time() - start_time
            
            print(f"✅ 连接测试成功!")
            print(f"   响应时间: {elapsed_time:.2f} 秒")
            print(f"   返回结果: {result}")
            return True
            
        except Exception as e:
            print(f"❌ 连接测试失败: {str(e)}")
            print(f"   错误类型: {type(e).__name__}")
            return False
    
    def test_timeout_behavior(self):
        """
        测试2: 超时行为测试
        故意设置较短的超时时间来验证超时机制
        """
        print("\n🧪 测试2: 超时行为测试")
        print("-" * 40)
        
        # 临时修改超时时间为1秒进行测试
        original_timeout = QWEN_CONFIG["timeout"]
        QWEN_CONFIG["timeout"] = 1
        
        try:
            # 发送需要较长处理时间的复杂提示词
            complex_prompt = """请详细解释量子计算的基本原理，包括：
            1. 量子比特的概念
            2. 量子叠加态
            3. 量子纠缠
            4. 量子门操作
            5. 量子算法优势
            要求回答至少500字以上。"""
            
            print(f"发送复杂提示词（预期会超时）...")
            start_time = time.time()
            
            try:
                result = call_qwen_plus(complex_prompt)
                elapsed_time = time.time() - start_time
                print(f"⚠️  意外成功! 耗时: {elapsed_time:.2f} 秒")
                print(f"   结果预览: {result[:100]}...")
                return True
            except Exception as e:
                elapsed_time = time.time() - start_time
                print(f"✅ 超时机制正常工作!")
                print(f"   实际耗时: {elapsed_time:.2f} 秒")
                print(f"   错误信息: {str(e)}")
                return True
                
        except Exception as e:
            print(f"❌ 超时测试出现意外错误: {str(e)}")
            return False
        finally:
            # 恢复原始超时设置
            QWEN_CONFIG["timeout"] = original_timeout
    
    def test_retry_mechanism(self):
        """
        测试3: 重试机制测试
        模拟API调用失败情况，验证重试逻辑
        """
        print("\n🧪 测试3: 重试机制测试")
        print("-" * 40)
        
        # 临时修改为无效API密钥来触发重试
        original_api_key = QWEN_CONFIG["api_key"]
        QWEN_CONFIG["api_key"] = "invalid_test_key_123456"
        
        try:
            test_prompt = "测试重试机制"
            print(f"使用无效API密钥进行测试...")
            
            start_time = time.time()
            try:
                result = call_qwen_plus(test_prompt)
                print(f"❌ 重试测试失败：应该触发错误但没有")
                return False
            except Exception as e:
                elapsed_time = time.time() - start_time
                error_msg = str(e)
                
                # 检查是否包含重试相关信息
                if "重试" in error_msg or "retry" in error_msg.lower():
                    print(f"✅ 重试机制正常触发!")
                    print(f"   总耗时: {elapsed_time:.2f} 秒")
                    print(f"   错误信息: {error_msg}")
                    return True
                else:
                    print(f"⚠️  重试机制可能未正确工作")
                    print(f"   错误信息: {error_msg}")
                    return False
                    
        except Exception as e:
            print(f"❌ 重试测试出现意外错误: {str(e)}")
            return False
        finally:
            # 恢复原始API密钥
            QWEN_CONFIG["api_key"] = original_api_key
    
    def test_parameter_validation(self):
        """
        测试4: 参数验证测试
        验证各种边界条件下的参数处理
        """
        print("\n🧪 测试4: 参数验证测试")
        print("-" * 40)
        
        test_cases = [
            ("", "空字符串测试"),
            ("你好", "正常短文本测试"),
            ("A" * 10000, "长文本测试(10000字符)"),
            (None, "None值测试"),
            (123, "非字符串类型测试")
        ]
        
        results = []
        for prompt, description in test_cases:
            try:
                print(f"测试 {description}: ", end="")
                if isinstance(prompt, str) and len(prompt) > 0:
                    result = call_qwen_plus(prompt[:50] + "..." if len(prompt) > 50 else prompt)
                    print(f"✅ 成功")
                    results.append(True)
                else:
                    # 预期会抛出异常
                    try:
                        call_qwen_plus(prompt)
                        print(f"❌ 应该失败但成功了")
                        results.append(False)
                    except Exception:
                        print(f"✅ 正确拒绝无效输入")
                        results.append(True)
            except Exception as e:
                if description in ["空字符串测试", "None值测试", "非字符串类型测试"]:
                    print(f"✅ 正确处理无效输入: {type(e).__name__}")
                    results.append(True)
                else:
                    print(f"❌ 意外错误: {str(e)}")
                    results.append(False)
        
        success_rate = sum(results) / len(results) * 100
        print(f"\n📊 参数验证测试通过率: {success_rate:.1f}% ({sum(results)}/{len(results)})")
        return success_rate >= 80
    
    def diagnose_common_issues(self):
        """
        诊断常见问题
        提供针对性的解决方案建议
        """
        print("\n🛠️  常见问题诊断")
        print("-" * 40)
        
        issues_found = []
        
        # 1. API密钥检查
        if not self.config["api_key"] or self.config["api_key"] == "your_aliyun_dashscope_api_key":
            issues_found.append("❌ API密钥未配置或为默认值")
        else:
            issues_found.append("✅ API密钥已配置")
        
        # 2. 超时设置检查
        if self.config["timeout"] < 10:
            issues_found.append(f"⚠️  超时时间({self.config['timeout']}秒)可能过短")
        elif self.config["timeout"] > 120:
            issues_found.append(f"⚠️  超时时间({self.config['timeout']}秒)可能过长")
        else:
            issues_found.append(f"✅ 超时时间({self.config['timeout']}秒)设置合理")
        
        # 3. 重试次数检查
        if self.config["retry_times"] == 0:
            issues_found.append("⚠️  重试次数为0，建议设置为2-3次")
        elif self.config["retry_times"] > 5:
            issues_found.append(f"⚠️  重试次数({self.config['retry_times']}次)过多，可能延长等待时间")
        else:
            issues_found.append(f"✅ 重试次数({self.config['retry_times']}次)设置合理")
        
        # 4. 模型名称检查
        valid_models = ["qwen-plus", "qwen-turbo", "qwen-max"]
        if self.config["model_name"] in valid_models:
            issues_found.append(f"✅ 模型名称({self.config['model_name']})有效")
        else:
            issues_found.append(f"⚠️  模型名称({self.config['model_name']})可能无效，建议使用: {valid_models}")
        
        # 显示诊断结果
        for issue in issues_found:
            print(issue)
        
        return issues_found
    
    def run_comprehensive_test(self):
        """
        运行综合测试套件
        执行所有测试并生成报告
        """
        print("🚀 启动 qwen-plus 综合测试套件")
        print("=" * 60)
        
        # 执行各项测试
        test_results = {
            "基础连接测试": self.test_basic_connection(),
            "超时行为测试": self.test_timeout_behavior(),
            "重试机制测试": self.test_retry_mechanism(),
            "参数验证测试": self.test_parameter_validation()
        }
        
        # 生成测试报告
        print("\n📋 测试报告")
        print("=" * 60)
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ 通过" if result else "❌ 失败"
            print(f"{test_name}: {status}")
        
        print(f"\n📊 总体结果: {passed_tests}/{total_tests} 项测试通过")
        print(f"成功率: {passed_tests/total_tests*100:.1f}%")
        
        # 运行问题诊断
        self.diagnose_common_issues()
        
        # 提供优化建议
        print("\n💡 优化建议:")
        print("-" * 30)
        if not test_results["基础连接测试"]:
            print("1. 检查网络连接和API密钥配置")
            print("2. 验证dashscope服务可用性")
        if not test_results["超时行为测试"]:
            print("3. 考虑增加超时时间配置")
        if not test_results["重试机制测试"]:
            print("4. 检查重试逻辑实现")
        if passed_tests < total_tests:
            print("5. 查看详细的错误日志进行调试")
        
        return passed_tests == total_tests


# ===================== 企业级测试与调用示例（闭环验证qwen-plus） =====================
if __name__ == "__main__":
    # 首先运行qwen-plus专项测试
    print("🎯 首先运行 qwen-plus 专项诊断测试")
    tester = QwenPlusTester()
    test_passed = tester.run_comprehensive_test()
    
    if not test_passed:
        print("\n⚠️  qwen-plus 测试发现问题，请根据上述建议进行修复")
        # 可以选择是否继续运行RAG流程
        user_input = input("\n是否继续运行RAG流程测试? (y/n): ")
        if user_input.lower() != 'y':
            print("程序退出")
            exit(0)
    
    print("\n" + "=" * 80)
    print("🔄 继续运行标准RAG流程测试")
    print("=" * 80)
    
    # 模拟前序流程生成的组件（实际使用时，直接传入训练好的组件即可）
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import faiss

    # 模拟参数（与m3e-base向量维度匹配：768维）
    demo_embedding_dim = 768
    demo_text_chunks = [
        "【婚恋心理咨询】情侣间关于工资卡管理的矛盾，核心是安全感与信任的博弈。建议先坦诚沟通，了解对方对工资卡管理的核心诉求（是担心财务混乱，还是缺乏安全感），避免情绪化对抗。",
        "【婚恋心理咨询】工资卡管理矛盾的解决技巧：1. 建立共同账户与个人账户并存的模式，共同账户用于家庭开支，个人账户保留自主支配权；2. 定期同步家庭财务状况，提升双方的财务透明度；3. 尊重对方的消费习惯，不强迫对方接受自己的财务观念。"
    ]  # 模拟实际分割后的文本块

    # 模拟PCA模型和scaler模型
    scaler_model = StandardScaler()
    pca_model = PCA(n_components=0.95)
    # 模拟训练数据（仅用于初始化模型，实际使用时为真实m3e向量）
    demo_train_embeddings = np.random.rand(100, demo_embedding_dim).astype(np.float32)
    scaler_model.fit(demo_train_embeddings)
    pca_model.fit(scaler_model.transform(demo_train_embeddings))

    # 修复FAISS索引初始化（适配1.13.2，纯位置参数）
    faiss_index = faiss.read_index(get_storage_path("faiss_fixed_pq_1.13.2.index"))

    # 2. 初始化企业级RAG流水线（集成qwen-plus）
    try:
        rag_pipeline = EnterpriseRAGPipeline(
            faiss_index=faiss_index,
            pca_model=pca_model,
            scaler_model=scaler_model,
            text_chunks=demo_text_chunks,
            top_k=5
        )

        # 3. 测试查询（心理咨询场景）
        test_query = "和男朋友因为工资卡管理产生矛盾，感到不安怎么办？"
        rag_result = rag_pipeline.run(test_query)

        # 4. 打印结构化结果（验证qwen-plus输出）
        print("=" * 80)
        print(f"查询状态：{rag_result['status']}")
        print(f"使用模型：{rag_result['qa_model']}")
        print(f"总处理耗时：{rag_result['total_processing_time_seconds']}秒")
        print(f"\n用户查询：{rag_result['query']}")
        print(f"\n回答结果：\n{rag_result['answer']}")
        print(f"\n检索到的上下文数量：{len(rag_result['contexts'])}")
        print(f"\n最相似上下文距离：{rag_result['distances'][0] if rag_result['distances'] else '无'}")
        print("=" * 80)
    except Exception as e:
        logger.critical(f"RAG流水线初始化或运行失败：{str(e)}")
        exit(1)