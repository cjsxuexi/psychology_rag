"""
企业级心理咨询师AI问答应用
基于Milvus+RetrievalEnhanced+QWEN实现智能心理咨询服务
"""

import os
import sys
from typing import List, Dict

from loguru import logger

from common.file_utils import get_config

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入核心组件
from data_process.milvus_stream_processor import MilvusStreamProcessor
from retrieval_augment.retrieval_enhanced import RetrievalEnhanced
from application.app1 import call_qwen_plus


class PsychologyChatBot:
    """
    企业级心理咨询师AI问答机器人
    
    核心功能流程：
    1. 用户输入问题
    2. Milvus检索获取前20条相似数据
    3. RetrievalEnhanced重排序获取前4条高质量数据
    4. QWEN大模型生成专业心理咨询服务回答
    """

    def __init__(self,
                 collection_name: str = "psychology_dialogues",
                 milvus_host: str = "localhost",
                 milvus_port: str = "19530",
                 reranker_model: str = "BAAI/bge-reranker-v2-m3"):
        """
        初始化心理咨询聊天机器人
        
        Args:
            collection_name: Milvus集合名称
            milvus_host: Milvus服务主机
            milvus_port: Milvus服务端口
            reranker_model: 重排序模型名称
        """
        self.collection_name = collection_name
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.reranker_model = reranker_model

        # 初始化核心组件
        self._initialize_components()

        logger.success("心理咨询师AI问答机器人初始化完成")

    def _initialize_components(self):
        """初始化核心组件"""
        try:
            # 1. 初始化Milvus检索器
            logger.info("正在初始化Milvus检索器...")
            self.milvus_processor = MilvusStreamProcessor(
                milvus_host=self.milvus_host,
                milvus_port=self.milvus_port,
                collection_name=self.collection_name
            )

            # 2. 初始化重排序器
            logger.info("正在初始化重排序器...")
            self.retrieval_enhancer = RetrievalEnhanced(
                model_name=self.reranker_model,
                use_fp16=True,
                batch_size=16
            )

            logger.success("所有核心组件初始化成功")

        except Exception as e:
            logger.error(f"组件初始化失败: {str(e)}")
            raise

    def _search_similar_chunks(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        从Milvus中检索相似的文本块
        
        Args:
            query: 用户查询问题
            top_k: 返回top-k结果
            
        Returns:
            List[Dict]: 检索结果列表
        """
        try:
            logger.info(f"开始Milvus检索，查询: {query[:50]}...")
            results = self.milvus_processor.search_similar_chunks(
                query_text=query,
                top_k=top_k
            )

            logger.success(f"Milvus检索完成，返回{len(results)}条结果")
            return results

        except Exception as e:
            logger.error(f"Milvus检索失败: {str(e)}")
            raise

    def _rerank_results(self, query: str, search_results: List[Dict], top_k: int = 4) -> List[Dict]:
        """
        使用RetrievalEnhanced对检索结果进行重排序
        
        Args:
            query: 用户查询问题
            search_results: Milvus检索结果
            top_k: 重排序后返回top-k结果
            
        Returns:
            List[Dict]: 重排序后的结果列表
        """
        try:
            logger.info(f"开始重排序，输入{len(search_results)}条结果...")

            reranked_results = self.retrieval_enhancer.rerank_results(
                query_text=query,
                search_results=search_results,
                top_k=top_k,
                return_scores=True
            )

            logger.success(f"重排序完成，返回{len(reranked_results)}条高质量结果")

            return reranked_results

        except Exception as e:
            logger.error(f"重排序失败: {str(e)}")
            raise

    def _generate_response(self, query: str, contexts: List[Dict]) -> str:
        """
        调用QWEN大模型生成回答
        
        Args:
            query: 用户查询问题
            contexts: 上下文信息列表
            
        Returns:
            str: 生成的回答
        """
        try:
            logger.info("开始调用QWEN大模型生成回答...")

            # 构建上下文字符串
            context_str = "\n\n".join([
                f"【参考案例{idx + 1}】\n标签: {ctx.get('tag', '无')}\n内容: {ctx.get('content', '')[:300]}..."
                for idx, ctx in enumerate(contexts)
            ])

            # 构建提示词 - 使用配置文件中的心理学提示词模板
            psychology_prompt_template = get_config("psychology_prompt.txt")

            # 渲染模板
            prompt = psychology_prompt_template.format(
                context_str=context_str,
                query=query
            )

            # 调用QWEN模型
            response = call_qwen_plus(prompt)

            logger.success("QWEN模型回答生成完成")
            return response

        except Exception as e:
            logger.error(f"QWEN模型调用失败: {str(e)}")
            return f"抱歉，暂时无法为您提供咨询服务。错误信息：{str(e)}"

    def chat(self, user_question: str) -> Dict:
        """
        核心聊天接口：处理用户问题并返回回答
        
        Args:
            user_question: 用户提问
            
        Returns:
            Dict: 包含完整处理过程和结果的字典
        """
        try:
            logger.info(f"收到用户问题: {user_question}")

            # 1. Milvus检索获取前20条数据
            milvus_results = self._search_similar_chunks(user_question, top_k=20)

            # 2. RetrievalEnhanced重排序获取前4条数据
            reranked_results = self._rerank_results(user_question, milvus_results, top_k=4)

            # 3. 调用QWEN大模型生成回答
            ai_response = self._generate_response(user_question, reranked_results)

            # 4. 构建完整响应
            response_data = {
                "user_question": user_question,
                "ai_response": ai_response,
                "milvus_raw_count": len(milvus_results),
                "final_context_count": len(reranked_results),
                "contexts_used": [
                    {
                        "rank": idx + 1,
                        "rerank_score": ctx.get("rerank_score", 0),
                        "original_distance": ctx.get("original_distance", 0),
                        "tag": ctx.get("tag", "无"),
                        "content_preview": ctx.get("content", "")[:100] + "..."
                    }
                    for idx, ctx in enumerate(reranked_results)
                ],
                "status": "success"
            }

            logger.success(f"问答流程完成，使用了{len(reranked_results)}条上下文")
            return response_data

        except Exception as e:
            logger.error(f"聊天处理失败: {str(e)}")
            return {
                "user_question": user_question,
                "ai_response": f"处理您的问题时出现错误：{str(e)}",
                "milvus_raw_count": 0,
                "final_context_count": 0,
                "contexts_used": [],
                "status": "error",
                "error_message": str(e)
            }

    def interactive_chat(self):
        """交互式聊天界面"""
        print("=" * 80)
        print("📌 输入 'quit' 或 'exit' 退出系统")
        print("=" * 80)

        while True:
            try:
                # 获取用户输入
                user_input = input("\n👤 您的问题: ").strip()

                # 退出条件
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("\n👋 感谢使用心理咨询师AI系统，祝您心情愉快！")
                    break

                # 空输入处理
                if not user_input:
                    print("⚠️  请输入有效的问题")
                    continue

                # 处理用户问题
                print("\n🤖 正在为您分析并生成专业建议...")
                result = self.chat(user_input)

                # 显示结果
                if result["status"] == "success":
                    print("\n" + "=" * 60)
                    print("🧠 AI心理咨询师回答：")
                    print("=" * 60)
                    print(result["ai_response"])

                    # 显示使用的上下文信息
                    print("\n" + "=" * 60)
                    print(f"📚 参考了{result['final_context_count']}个高质量案例：")
                    print("=" * 60)
                    for ctx in result["contexts_used"]:
                        print(f"\n📍 案例{ctx['rank']} (相关度: {ctx['rerank_score']:.4f})")
                        print(f"   标签: {ctx['tag']}")
                        print(f"   内容预览: {ctx['content_preview']}")
                else:
                    print(f"\n❌ 处理失败: {result['error_message']}")

            except KeyboardInterrupt:
                print("\n\n👋 系统已退出，再见！")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {str(e)}")
                continue

    def close(self):
        """关闭资源"""
        try:
            if hasattr(self, 'milvus_processor'):
                self.milvus_processor.close()
            logger.info("心理咨询师AI问答机器人资源已释放")
        except Exception as e:
            logger.error(f"关闭资源时出错: {str(e)}")


def main():
    """主函数 - 启动交互式心理咨询系统"""
    try:
        # 初始化聊天机器人
        chatbot = PsychologyChatBot(
            collection_name="psychology_dialogues",
            milvus_host="localhost",
            milvus_port="19530"
        )

        # 启动交互式聊天
        chatbot.interactive_chat()

    except Exception as e:
        logger.error(f"系统启动失败: {str(e)}")
        print(f"❌ 系统启动失败: {str(e)}")
        return 1

    finally:
        # 确保资源释放
        if 'chatbot' in locals():
            chatbot.close()

    return 0


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # 运行主程序
    exit_code = main()
    sys.exit(exit_code)
