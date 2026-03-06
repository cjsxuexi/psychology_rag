"""
企业级心理咨询师AI问答应用
基于Milvus+RetrievalEnhanced+QWEN实现智能心理咨询服务
"""

import os
import sys
from typing import List, Dict, Optional

from loguru import logger
from src.common.file_utils import get_config
from src.application.app1 import call_qwen_plus

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SimpleConversationMemory:
    """
    简单的对话记忆类，用于记录对话历史并生成摘要
    """
    
    def __init__(self):
        """初始化对话记忆"""
        self.conversation_history = []
        self.summary = ""
    
    def save_context(self, inputs: Dict[str, str], outputs: Dict[str, str]):
        """
        保存对话上下文
        
        Args:
            inputs: 输入内容，包含"input"键
            outputs: 输出内容，包含"output"键
        """
        user_input = inputs.get("input", "")
        ai_output = outputs.get("output", "")
        
        if user_input and ai_output:
            self.conversation_history.append({"user": user_input, "ai": ai_output})
            logger.info(f"对话历史已更新，当前记录数: {len(self.conversation_history)}")
            # 生成新的摘要
            self._update_summary()
    
    def load_memory_variables(self, inputs: Dict) -> Dict[str, str]:
        """
        加载记忆变量
        
        Args:
            inputs: 输入参数
            
        Returns:
            Dict: 包含历史摘要的字典
        """
        return {"history": self.summary}
    
    def _update_summary(self):
        """
        更新对话摘要
        """
        if not self.conversation_history:
            self.summary = ""
            return
        
        # 生成摘要的提示词
        summary_prompt = f"请为以下对话生成一个简洁的摘要，突出核心问题和解决方案：\n\n"
        
        # 构建对话历史字符串
        history_str = "\n".join([
            f"用户: {item['user']}\nAI: {item['ai']}"
            for item in self.conversation_history
        ])
        
        prompt = summary_prompt + history_str
        
        try:
            # 使用QWEN模型生成摘要
            self.summary = call_qwen_plus(prompt)
            logger.info(f"对话摘要已更新，摘要长度: {len(self.summary)}")
        except Exception as e:
            logger.warning(f"生成对话摘要失败: {str(e)}")
            # 降级方案：使用最后几条对话作为摘要
            if len(self.conversation_history) <= 3:
                self.summary = history_str
            else:
                recent_history = "\n".join([
                    f"用户: {item['user']}\nAI: {item['ai']}"
                    for item in self.conversation_history[-3:]
                ])
                self.summary = f"最近的对话：\n{recent_history}"
    
    def clear(self):
        """
        清空对话记忆
        """
        self.conversation_history = []
        self.summary = ""
        logger.info("对话记忆已清空")


# 导入核心组件
from src.data_process.milvus_stream_processor import MilvusStreamProcessor
from src.retrieval_augment.retrieval_enhanced import RetrievalEnhanced


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
            # 1. 初始化检索增强器（集成Milvus和重排序）
            logger.info("正在初始化检索增强器...")
            self.retrieval_enhancer = RetrievalEnhanced(
                model_name=self.reranker_model,
                use_fp16=True,
                batch_size=16,
                collection_name=self.collection_name,
                milvus_host=self.milvus_host,
                milvus_port=self.milvus_port
            )

            # 2. 初始化对话记忆组件
            logger.info("正在初始化对话记忆组件...")
            try:
                self.conversation_memory = SimpleConversationMemory()
                logger.info("对话记忆组件初始化成功")
            except Exception as e:
                logger.warning(f"初始化对话记忆组件失败: {str(e)}，对话记忆功能将不可用")
                self.conversation_memory = None

            logger.success("所有核心组件初始化成功")

        except Exception as e:
            logger.error(f"组件初始化失败: {str(e)}")
            raise



    def _generate_response(self, query: str, contexts: List[Dict], conversation_summary: str = "") -> str:
        """
        调用QWEN大模型生成回答
        
        Args:
            query: 用户查询问题
            contexts: 上下文信息列表
            conversation_summary: 对话历史摘要
            
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

            # 构建对话历史摘要
            history_str = "" if not conversation_summary else f"\n\n【对话历史摘要】\n{conversation_summary}"

            # 构建提示词 - 使用配置文件中的心理学提示词模板
            psychology_prompt_template = get_config("psychology_prompt.txt")

            # 渲染模板
            prompt = psychology_prompt_template.format(
                context_str=context_str,
                query=query,
                history_str=history_str
            )

            logger.info(f"调用QWEN模型的提示词: {prompt[:600]}...")
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
            
            # 获取对话历史摘要
            conversation_summary = ""
            if self.conversation_memory:
                conversation_summary = self.conversation_memory.load_memory_variables({}).get("history", "")
                logger.info(f"加载对话历史摘要: {conversation_summary[:100]}...")

            # 1. 使用检索增强器获取高质量结果
            reranked_results = self.retrieval_enhancer.retrieval(user_question, search_top_k=20, rerank_top_k=4)
            milvus_results = reranked_results  # 保持向后兼容

            # 3. 调用QWEN大模型生成回答
            ai_response = self._generate_response(user_question, reranked_results, conversation_summary)

            # 4. 更新对话记忆
            if self.conversation_memory:
                self.conversation_memory.save_context(
                    {"input": user_question},
                    {"output": ai_response}
                )
                logger.info("对话历史已更新到记忆组件")

            # 5. 构建完整响应
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
                "conversation_summary": conversation_summary[:200] + ("..." if len(conversation_summary) > 200 else ""),
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
                "conversation_summary": "",
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
            if hasattr(self, 'retrieval_enhancer'):
                self.retrieval_enhancer.close()
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