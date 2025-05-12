import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Union

from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app.core.function_manager import function_registry, function_adapter
from sqlalchemy.orm import Session

from app.ai.llm_manager import llm_manager
from app.ai.prompts import prompt_manager
from app.core.logger import app_logger as logger
from app.db.repositories import FileRepository
from app.processor.file_processor import FileProcessor
from app.schemas.chat import ChatResponse, Message, Conversation
from app.services.file_content_service import FileContentService
from app.services.memory_service import MemoryService
from app.services.message_processor import MessageProcessor
from app.services.model_strategies import ModelStrategyFactory
from app.services.stream_handler import StreamHandler
from app.services.web_search_service import WebSearchService

class ChatService:
    def __init__(self, db: Session):
        self.db = db
        # 初始化各种服务
        self.memory_service = MemoryService(db)
        self.file_processor = FileProcessor()
        
        self.message_processor = MessageProcessor(db)
        self.stream_handler = StreamHandler(db, self.memory_service)
        self.file_service = FileContentService(db)

    async def process_message(
            self,
            provider: str,
            model: str,
            message: str,
            conversation_id: Optional[str] = None,
            stream: bool = False,
            options: Optional[Dict[str, Any]] = None,
            file_ids: Optional[List[str]] = None,
    ) -> Union[StreamingResponse, ChatResponse]:
        """处理用户消息并获取AI响应"""
        # 初始化options
        if options is None:
            options = {}
        
        # 获取或创建会话
        conversation = self._get_or_create_conversation(conversation_id, provider, model, message)

        # 记录用户消息
        user_message = Message(role="user", content=message)
        conversation.messages.append(user_message)
        
        # 准备聊天历史
        chat_history = []
        for msg in conversation.messages:
            chat_history.append({"role": msg.role, "content": msg.content})

        # 从聊天历史中提取消息
        messages = self.message_processor.prepare_chat_messages(chat_history)

        # 保存会话（先保存用户消息）
        conversation.updated_at = datetime.now()
        self.memory_service.save_conversation(conversation)

        # 处理文件内容
        if file_ids and len(file_ids) > 0:
            # 检查文件状态
            status_response = self.file_service.check_files_status(file_ids, provider, model, conversation.id)
            if status_response:
                return status_response
                
            # 获取文件内容并增强消息
            file_contents = self.file_service.get_files_content(file_ids)
            if file_contents:
                messages = self.message_processor.enhance_with_file_content(messages, message, file_contents)

        # 根据是否为流式响应分别处理
        if stream:
            return await self._handle_stream_response(provider, model, messages, conversation.id, options)
        else:
            return await self._handle_normal_response(provider, model, messages, conversation.id, options)

    def _get_or_create_conversation(self, conversation_id, provider, model, message):
        """获取或创建会话"""
        conversation = None
        if conversation_id:
            conversation = self.memory_service.get_conversation(conversation_id)

        if not conversation:
            # 创建新对话
            conversation = Conversation(
                id=conversation_id or str(uuid.uuid4()),
                title=message[:30] + "..." if len(message) > 30 else message,
                provider=provider,
                model=model,
                messages=[]
            )
            
        return conversation

    async def _handle_stream_response(self, provider, model, messages, conversation_id, options=None):
        """处理流式响应"""
        # 默认options
        if options is None:
            options = {}
            
        # 判断是否使用推理模式
        use_reasoning = options.get("use_reasoning", False)
        
        # 判断是否使用函数调用
        use_function_call = options.get("use_function_call", False)
        print("use_function_call.....................", use_function_call)
        if use_function_call:
            return StreamingResponse(
                self.generate_function_call_stream(provider, model, messages, conversation_id, options),
                media_type="text/event-stream"
            )
        
        # 火山引擎特殊处理 - 直接使用OpenAI客户端访问API
        if provider == "volcengine":
            return StreamingResponse(
                self.stream_handler.direct_reasoning_stream(provider, model, messages, conversation_id),
                media_type="text/event-stream"
            )
            # 根据模型名称判断使用推理模式（向后兼容）
        elif provider in ("deepseek", "qwen") and use_reasoning:
            return StreamingResponse(
                self.stream_handler.generate_reasoning_stream(provider, model, messages, conversation_id),
                media_type="text/event-stream"
            )
        # 根据options判断使用推理模式
        elif use_reasoning:
            return StreamingResponse(
                self.stream_handler.generate_reasoning_stream(provider, model, messages, conversation_id),
                media_type="text/event-stream"
            )
        
        # 默认使用常规流式响应
        else:
            return StreamingResponse(
                self.stream_handler.generate_normal_stream(provider, model, messages, conversation_id),
                media_type="text/event-stream"
            )

    async def _handle_normal_response(self, provider, model, messages, conversation_id, options=None):
        """处理非流式响应"""
        # 默认options
        if options is None:
            options = {}
        
        # 获取适合的模型处理策略
        strategy = ModelStrategyFactory.get_strategy(provider, model, options)
        
        try:
            # 使用策略处理请求
            ai_message, reasoning_message = await strategy.process(provider, model, messages, conversation_id, self.memory_service, options)
            
            # 获取会话
            conversation = self.memory_service.get_conversation(conversation_id)
            
            # 如果有推理内容，添加到会话
            if reasoning_message:
                conversation.messages.append(reasoning_message)
            
            # 添加AI响应到会话
            conversation.messages.append(ai_message)
            
            # 更新并保存会话
            conversation.updated_at = datetime.now()
            self.memory_service.save_conversation(conversation)
            
            # 返回响应
            reasoning_content = reasoning_message.content if reasoning_message else ""
            return ChatResponse(
                id=str(uuid.uuid4()),
                provider=provider,
                model=model,
                message=ai_message,
                conversation_id=conversation.id,
                reasoning=reasoning_content
            )
        except Exception as e:
            logger.error(f"模型处理失败: {e}")
            raise

    async def generate_function_call_stream(self, provider, model, messages, conversation_id, options=None):
        """生成支持函数调用的流式响应"""
        
        print("generate_function_call_stream.....................")
        
        # 构造发送事件辅助函数
        async def send_event(event_type, content=None):
            data = {"type": event_type, "conversation_id": conversation_id}
            if content is not None:
                data["content"] = content
            return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        
        # 函数类型处理器映射
        function_handlers = {
            "web_search": self._handle_web_search_function,
            "hot_topics": self._handle_hot_topics_function,
            # 将来可以在这里添加更多函数处理器
        }
        
        # 创建上下文
        context = {
            "db": self.db,
            "conversation_id": conversation_id
        }
        
        # 获取模型
        llm = llm_manager.get_model(provider=provider, model=model)
        
        # 准备函数定义
        functions_kwargs = function_adapter.prepare_functions_for_model(provider, model)
        
        try:
            # 开始函数流
            yield await send_event("function_stream_start")
            
            # 第一次流式调用模型
            full_response = ""
            function_call_detected = False
            function_call_data = {}
            
            # 处理流式响应中的函数调用检测
            for chunk in llm.stream(messages, **functions_kwargs):
                # 检查流中是否有函数调用
                if not function_call_detected:
                    # 使用适配器检测函数调用
                    function_call_detected, function_call_data = function_adapter.detect_function_call_in_stream(chunk)
                    
                    if function_call_detected:
                        function_name = function_call_data['function'].get('name')
                        yield await send_event("function_call_detected", {
                            "function_type": function_name,
                            "description": f"需要调用函数: {function_name}"
                        })
                        break
                
                # 如果无函数调用，正常返回内容
                content = chunk.content if hasattr(chunk, 'content') else chunk
                if content:
                    full_response += content
                    yield await send_event("content", content)
            
            # 如果没有检测到函数调用，结束流
            if not function_call_detected:
                # 保存完整回应到会话历史
                await self.save_stream_response(conversation_id, full_response)
                yield await send_event("done")
                return
            
            # 执行函数调用
            function_name = function_call_data["function"].get("name", "")
            yield await send_event("executing_function", 
                                f"正在执行函数 {function_name}...")
            
            # 检查是否是web_search函数且没有query参数
            function_args = function_call_data["function"].get("arguments", "{}")
            
            # 尝试解析参数
            try:
                if isinstance(function_args, str):
                    args_dict = json.loads(function_args) if function_args.strip() else {}
                else:
                    args_dict = function_args
            except:
                args_dict = {}
                
            # 如果是web_search函数但没有query参数，使用LLM生成搜索查询
            if function_name == "web_search" and not args_dict.get("query"):
                yield await send_event("generating_query", "正在优化搜索查询...")
                
                # 从原始消息中提取用户的最后一条消息
                user_message = ""
                for msg in reversed(messages):
                    # 处理不同类型的消息对象
                    if hasattr(msg, "type") and msg.type == "human":
                        user_message = msg.content
                        break
                    elif hasattr(msg, "role") and msg.role == "user":
                        user_message = msg.content
                        break
                    elif isinstance(msg, dict) and msg.get("role") == "user":
                        user_message = msg.get("content", "")
                        break
                
                if user_message:
                    # 让LLM生成优化后的搜索查询
                    current_date = datetime.now().strftime("%Y年%m月%d日")
                    search_query_prompt = f"为以下用户问题生成一个简洁明确的搜索查询: '{user_message}'。如果问题中包含与当前时间相关的指代（例如'今天'、'目前'），请以 {current_date} 作为当前日期进行理解。请仅返回搜索查询文本本身，不要附加任何解释或说明。"
                    search_query_msgs = [{"role": "user", "content": search_query_prompt}]
                    
                    # 使用现有模型生成查询
                    search_query_response = await llm.ainvoke(search_query_msgs)
                    search_query = search_query_response.content if hasattr(search_query_response, 'content') else str(search_query_response)
                    
                    # 清理搜索查询（去除引号等）
                    search_query = search_query.strip().strip('"\'')
                    
                    # 更新函数参数
                    args_dict["query"] = search_query
                    function_call_data["function"]["arguments"] = json.dumps(args_dict)
                    
                    yield await send_event("query_generated", f"搜索查询: {search_query}")
            
            # 处理函数调用
            function_result = await function_adapter.process_function_call(
                provider, function_call_data["function"], context
            )
            
            # 准备工具消息
            tool_message_data = function_adapter.prepare_tool_message(
                provider, function_name, function_result, 
                function_call_data.get("tool_call_id")
            )
            
            # 通知函数执行完成
            yield await send_event("function_executed", {
                "function_type": function_name,
                "result": json.dumps(function_result, ensure_ascii=False)
            })
            
            # 检查是否有专门的处理器处理该函数类型
            handler = function_handlers.get(function_name)
            if handler:
                # 使用专门的处理器处理该函数类型
                async for event in handler(
                    send_event, function_call_data, function_result, conversation_id, llm, messages
                ):
                    yield event
                return
                
            # 如果没有专门的处理器，使用默认处理流程
                
            # 复制原始消息并添加函数调用结果
            full_messages = list(messages)
            # 添加LLM的函数调用响应
            if provider in ["deepseek", "openai", "anthropic", "qwen", "volcengine"]:
                # 使用tool_calls格式
                full_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "type": "function",
                        "function": function_call_data["function"],
                        "id": function_call_data.get("tool_call_id", "call_1")
                    }]
                })
            else:
                # 使用传统function_call格式
                full_messages.append({
                    "role": "assistant",
                    "content": "",
                    "function_call": function_call_data["function"]
                })
            # 添加函数执行结果
            full_messages.append(function_result)
            
            # 第二次调用模型生成最终回答
            yield await send_event("generating_response", "正在生成最终回答...")
            
            # 处理最终回答的流式响应
            final_response = ""
            for chunk in llm.stream(full_messages):
                content = chunk.content if hasattr(chunk, 'content') else chunk
                if content:
                    final_response += content
                    yield await send_event("content", content)
            
            # 保存完整对话历史
            await self.save_function_call_stream_response(
                conversation_id=conversation_id,
                function_name=function_name,
                function_args=function_call_data["function"].get("arguments", "{}"),
                function_result=function_result,
                final_response=final_response
            )
            
            # 完成标志
            yield await send_event("done")
            
        except Exception as e:
            # 错误处理
            logger.error(f"函数调用流处理出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield await send_event("error", f"处理出错: {str(e)}")

    async def _handle_web_search_function(self, send_event, function_call_data, function_result, 
                                          conversation_id, llm, messages):
        """处理web_search函数的专门处理器"""
        function_name = "web_search"
        
        # 对web_search函数的特殊处理
        if "results" in function_result:
            yield await send_event("content_direct", {
                "function_type": "web_search",
                "status": "processing"
            })
            
            # 格式化搜索结果
            results = function_result.get("results", [])
            query = function_result.get("query", "")
            
            if results:
                final_response = f"以下是关于{query}的搜索结果：\n\n"
                for i, result in enumerate(results, 1):
                    final_response += f"{i}. {result.get('title')}\n"
                    final_response += f"   {result.get('snippet')}\n"
                    final_response += f"   来源: {result.get('link')}\n\n"
            else:
                final_response = f"未找到关于{query}的搜索结果。"
            
            # 发送格式化的内容
            yield await send_event("content", final_response)
            
            # 保存完整对话历史
            await self.save_function_call_stream_response(
                conversation_id=conversation_id,
                function_name=function_name,
                function_args=function_call_data["function"].get("arguments", "{}"),
                function_result=function_result,
                final_response=final_response
            )
            
            # 完成标志
            yield await send_event("done")
        else:
            # 如果结果格式不符合预期，使用默认处理
            yield await send_event("error", "搜索结果格式不正确")

    async def _handle_hot_topics_function(self, send_event, function_call_data, function_result, 
                                          conversation_id, llm, messages):
        """处理hot_topics函数的专门处理器"""
        function_name = "hot_topics"
        
        # 对hot_topics函数的特殊处理
        if "topics" in function_result:
            yield await send_event("content_direct", {
                "function_type": "hot_topics",
                "status": "processing"
            })
            
            # 格式化热点话题结果
            topics = function_result.get("topics", [])
            if topics:
                final_response = "以下是最新热点话题：\n\n"
                for i, topic in enumerate(topics, 1):
                    final_response += f"{i}. {topic.get('title')}\n"
                    final_response += f"   {topic.get('description')}\n"
                    final_response += f"   来源: {topic.get('source')}, 类别: {topic.get('category')}\n\n"
            else:
                final_response = "当前没有热点话题信息。"
            
            # 发送格式化的内容
            yield await send_event("content", final_response)
            
            # 保存完整对话历史
            await self.save_function_call_stream_response(
                conversation_id=conversation_id,
                function_name=function_name,
                function_args=function_call_data["function"].get("arguments", "{}"),
                function_result=function_result,
                final_response=final_response
            )
            
            # 完成标志
            yield await send_event("done")
        else:
            # 如果结果格式不符合预期，使用默认处理
            yield await send_event("error", "热点话题结果格式不正确")

    async def save_function_call_stream_response(self, conversation_id, function_name, 
                                           function_args, function_result, final_response):
        """保存函数调用流式响应到对话历史"""
        try:
            conversation = self.memory_service.get_conversation(conversation_id)
            if conversation:
                # 创建函数调用请求消息，根据函数类型提供不同描述
                function_descriptions = {
                    "web_search": "我需要搜索网络获取更多信息...",
                    "hot_topics": "我将查询最新的热点话题...",
                    # 未来可以在这里添加更多函数类型描述
                }
                
                function_desc = function_descriptions.get(function_name, f"我需要调用 {function_name} 函数获取信息...")
                
                # 创建函数调用消息
                function_call_message = Message(
                    role="assistant",
                    content=function_desc
                )
                
                # 创建函数结果消息
                function_result_message = Message(
                    role="function",
                    content=json.dumps(function_result, ensure_ascii=False)
                )
                
                # 创建最终AI响应消息
                ai_message = Message(
                    role="assistant",
                    content=final_response
                )
                
                # 添加所有消息到会话
                conversation.messages.append(function_call_message)
                conversation.messages.append(function_result_message)
                conversation.messages.append(ai_message)
                
                # 更新会话时间
                conversation.updated_at = datetime.now()
                
                # 保存到数据库
                self.memory_service.save_conversation(conversation)
        except Exception as e:
            logger.error(f"保存函数调用流式响应失败: {e}")

    def get_all_conversations(self) -> List[Conversation]:
        """获取所有对话"""
        return self.memory_service.get_all_conversations()

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """获取特定对话"""
        return self.memory_service.get_conversation(conversation_id)

    def delete_conversation(self, conversation_id: str) -> bool:
        """删除特定对话"""
        try:
            # 然后删除数据库记录
            return self.memory_service.delete_conversation(conversation_id)
        except Exception as e:
            logging.error(f"删除对话失败: {e}")
            return False

    async def generate_title(
            self,
            message: Optional[str] = None,
            conversation_id: Optional[str] = None,
            options: Optional[Dict[str, Any]] = None
    ) -> str:
        """生成与消息或会话相关的标题"""
        # 如果提供了会话ID，获取会话
        conversation = None
        if conversation_id:
            conversation = self.memory_service.get_conversation(conversation_id)
            if not conversation:
                raise ValueError(f"找不到会话ID: {conversation_id}")

            # 使用会话的最后一次对话（用户和助手的消息）作为输入
            if not message and conversation.messages:
                # 获取会话中最后的用户和助手消息
                user_message = None
                assistant_message = None
                
                # 从后向前查找最近的一组对话
                for i in range(len(conversation.messages) - 1, -1, -1):
                    msg = conversation.messages[i]
                    if not assistant_message and msg.role == "assistant":
                        assistant_message = msg.content
                    if not user_message and msg.role == "user":
                        user_message = msg.content
                    if user_message and assistant_message:
                        break
                
                # 组合用户和助手的消息
                dialog_messages = []
                if user_message:
                    dialog_messages.append(f"用户: {user_message}")
                if assistant_message:
                    dialog_messages.append(f"助手: {assistant_message}")
                
                if dialog_messages:
                    message = "\n".join(dialog_messages)
                else:
                    # 如果没有找到对话，回退到之前的逻辑
                    user_messages = []
                    for msg in conversation.messages:
                        if msg.role == "user":
                            user_messages.append(msg.content)
                            if len(user_messages) >= 3:
                                break
                    
                    if user_messages:
                        message = "\n".join(user_messages)

        if not message:
            raise ValueError("必须提供消息内容或有效的会话ID")

        # 使用提示词管理器获取并格式化提示词
        prompt = prompt_manager.format_prompt("generate_title", content=message)

        try:
            # 获取AI模型并生成标题
            llm = llm_manager.get_default_model()
            response = llm.invoke([HumanMessage(content=prompt)])

            if hasattr(response, 'content'):  # ChatModel返回的响应
                title = response.content
            else:  # 普通LLM返回的响应
                title = response

            # 清理标题（去除多余的引号、空白和解释性文字）
            title = title.strip().strip('"\'')

            # 如果标题中包含"标题："等前缀，去除
            prefixes = ["标题：", "标题:", "主题：", "主题:"]
            for prefix in prefixes:
                if title.startswith(prefix):
                    title = title[len(prefix):].strip()

            # 限制标题长度
            if len(title) > 30:
                title = title[:30] + "..."

            # 如果提供了会话ID，更新会话标题
            if conversation_id and conversation:
                conversation.title = title
                conversation.updated_at = datetime.now()
                self.memory_service.save_conversation(conversation)

            return title
        except Exception as e:
            logging.error(f"生成标题时发生错误: {str(e)}")
            # 如果生成失败，返回一个默认标题
            if conversation_id:
                return f"对话 {conversation_id[:8]}..."
            else:
                return "新对话"

    async def generate_suggested_questions(
        self,
        conversation_id: str,
        latest_only: bool = True,
        options: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """生成与当前对话轮次相关的推荐问题"""
        # 获取会话
        conversation = self.memory_service.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"找不到会话ID: {conversation_id}")

        # 准备对话内容 - 只取最近一轮对话(最新的用户问题和AI回答)
        latest_user_msg = None
        latest_ai_msg = None
        
        # 从后向前查找最近的用户消息和AI回答
        for i in range(len(conversation.messages) - 1, -1, -1):
            msg = conversation.messages[i]
            if not latest_ai_msg and msg.role == "assistant":
                latest_ai_msg = msg.content
            elif not latest_user_msg and msg.role == "user":
                latest_user_msg = msg.content
            if latest_user_msg and latest_ai_msg:
                break
        
        # 组合最近一轮对话
        dialog_content = ""
        if latest_user_msg:
            dialog_content += f"用户: {latest_user_msg}\n"
        if latest_ai_msg:
            dialog_content += f"助手: {latest_ai_msg}"
        
        if not dialog_content:
            # 如果没有对话内容，返回默认问题
            return [
                "有什么我可以帮您解答的问题吗？",
                "您想了解更多哪方面的信息？",
                "还有其他我能帮助您的事情吗？"
            ]

        # 使用提示词管理器获取并格式化提示词
        prompt = prompt_manager.format_prompt("generate_suggested_questions", content=dialog_content)

        try:
            # 获取AI模型并生成问题
            llm = llm_manager.get_default_model()
            response = llm.invoke([HumanMessage(content=prompt)])

            if hasattr(response, 'content'):  # ChatModel返回的响应
                response_text = response.content
            else:  # 普通LLM返回的响应
                response_text = response

            # 解析响应文本，提取问题
            questions = self._parse_questions(response_text)
            
            return questions[:3]  # 确保只返回3个问题
        except Exception as e:
            logger.error(f"生成推荐问题时发生错误: {str(e)}")
            # 如果生成失败，返回默认问题
            return [
                "您对这个主题还有其他问题吗？",
                "您想了解更多相关信息吗？",
                "您想要探讨这个话题的哪些方面？"
            ]

    def _parse_questions(self, response_text: str) -> List[str]:
        """从响应文本中解析出问题列表"""
        questions = []
        
        # 尝试不同的解析方法
        # 1. 尝试按数字列表解析
        import re
        numbered_questions = re.findall(r'\d+[\.\)]\s*(.*?)(?=\n\d+[\.\)]|\n*$)', response_text, re.DOTALL)
        if numbered_questions and len(numbered_questions) >= 3:
            return [q.strip() for q in numbered_questions]
        
        # 2. 按行分割
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        for line in lines:
            # 移除行首的数字、点、括号等
            cleaned_line = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
            if cleaned_line:
                questions.append(cleaned_line)
        
        # 如果没有找到足够的问题，返回原始文本分成的前三行
        if len(questions) < 3:
            questions = lines[:3] if len(lines) >= 3 else lines
        
        return questions
        
    async def handle_function_calls(self, provider, model, messages, conversation_id, options=None):
        """
        处理函数调用流程
        
        参数:
            provider: 模型提供商
            model: 模型名称
            messages: 聊天消息列表
            conversation_id: 会话ID
            options: 其他选项
            
        返回:
            Chat响应对象
        """
        # 默认options
        if options is None:
            options = {}
            
        # 准备上下文
        context = {
            "db": self.db,
            "conversation_id": conversation_id
        }
        
        # 获取AI模型
        llm = llm_manager.get_model(provider=provider, model=model)
        
        # 准备函数定义
        functions_kwargs = function_adapter.prepare_functions_for_model(provider, model)
        
        try:
            # 调用模型，让其决定是否需要调用函数
            response = await llm.ainvoke(messages, **functions_kwargs)
            
            # 提取函数调用信息
            function_call, tool_call_id = function_adapter.extract_function_call(provider, response)
            
            # 如果没有函数调用，直接创建回复消息
            if not function_call:
                # 创建AI消息
                ai_message = Message(
                    role="assistant",
                    content=response.content if hasattr(response, 'content') else str(response)
                )
                
                # 获取会话
                conversation = self.memory_service.get_conversation(conversation_id)
                
                # 添加AI响应到会话
                conversation.messages.append(ai_message)
                
                # 更新并保存会话
                conversation.updated_at = datetime.now()
                self.memory_service.save_conversation(conversation)
                
                # 返回响应
                return ChatResponse(
                    id=str(uuid.uuid4()),
                    provider=provider,
                    model=model,
                    message=ai_message,
                    conversation_id=conversation.id
                )
            
            # 记录函数调用
            logger.info(f"模型选择调用函数: {function_call.get('name')}")
            
            # 获取会话
            conversation = self.memory_service.get_conversation(conversation_id)
            
            # 添加AI选择调用函数的消息
            ai_function_message = Message(
                role="assistant",
                content=response.content if hasattr(response, 'content') and response.content else f"我需要调用 {function_call.get('name')} 函数获取更多信息..."
            )
            conversation.messages.append(ai_function_message)
            
            # 处理函数调用
            function_result = await function_adapter.process_function_call(provider, function_call, context)
            
            # 准备工具消息
            function_name = function_call.get("name", "")
            tool_message_data = function_adapter.prepare_tool_message(
                provider, function_name, function_result, tool_call_id
            )
            
            # 创建工具消息
            tool_message = Message(
                role=tool_message_data["role"],
                content=tool_message_data["content"]
            )
            conversation.messages.append(tool_message)
            
            # 复制原始消息并添加函数调用结果
            full_messages = list(messages)
            full_messages.append(response)
            full_messages.append(tool_message_data)
            
            # 再次调用模型生成最终回答
            final_response = await llm.ainvoke(full_messages)
            
            # 创建最终AI消息
            final_ai_message = Message(
                role="assistant",
                content=final_response.content if hasattr(final_response, 'content') else str(final_response)
            )
            
            # 添加最终消息到会话
            conversation.messages.append(final_ai_message)
            
            # 更新并保存会话
            conversation.updated_at = datetime.now()
            self.memory_service.save_conversation(conversation)
            
            # 返回响应
            return ChatResponse(
                id=str(uuid.uuid4()),
                provider=provider,
                model=model,
                message=final_ai_message,
                conversation_id=conversation.id
            )
        except Exception as e:
            logger.error(f"函数调用处理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 创建错误消息
            error_message = Message(
                role="assistant",
                content=f"在处理函数调用时出现错误: {str(e)}"
            )
            
            # 返回错误响应
            return ChatResponse(
                id=str(uuid.uuid4()),
                provider=provider,
                model=model,
                message=error_message,
                conversation_id=conversation_id
            )

    async def save_stream_response(self, conversation_id, response_content):
        """保存普通流式响应到对话历史"""
        try:
            conversation = self.memory_service.get_conversation(conversation_id)
            if conversation:
                # 创建AI响应消息
                ai_message = Message(
                    role="assistant",
                    content=response_content
                )
                
                # 添加AI响应到会话
                conversation.messages.append(ai_message)
                
                # 更新会话时间
                conversation.updated_at = datetime.now()
                
                # 保存到数据库
                self.memory_service.save_conversation(conversation)
        except Exception as e:
            logger.error(f"保存流式响应失败: {e}")