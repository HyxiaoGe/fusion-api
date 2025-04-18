import asyncio
import base64
import io
import os
from typing import List, Optional, Any, Dict

from app.core.logger import app_logger as logger


class FileProcessor:
    """文件处理器，统一使用千问视觉大模型来处理文件"""

    def __init__(self):
        self.model = "qwen-omni-turbo"

        try:
            pass
            # from openai import OpenAI
            # self.client = OpenAI(
            #     api_key=os.getenv("DASHSCOPE_API_KEY"),
            #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            # )
            # logger.info(f"通义千问视觉模型初始化成功: {self.model}")
        except Exception as e:
            logger.error(f"初始化通义千问视觉模型失败: {e}")
            raise e

    async def process_files(
            self,
            file_paths: List[str],
            query: str = None,
            mime_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        处理文件并返回处理结果

        参数:
            file_paths: 文件路径列表
            query: 用户关于文件的问题或指示
            mime_types: 文件MIME类型列表，与file_paths顺序对应

        返回:
            包含处理结果的字典
        """
        try:
            # 生成默认查询（如果未提供）
            if not query:
                query = "请分析这些文件的内容并提供详细描述。"

            files_data = []
            for i, path in enumerate(file_paths):
                mime_type = mime_types[i] if mime_types and i < len(mime_types) else self._guess_mime_type(path)
                file_data = self._prepare_file_data(path, mime_type)
                files_data.append(file_data)
            # 构造提示信息
            prompt = self._build_prompt(query, files_data)
            # 调用模型
            response = await self._call_model(prompt, files_data)
            logger.info(f"处理文件成功: {response}")

            return {
                "content": response,
                "model": self.model
            }

        except Exception as e:
            logger.error(f"处理文件失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "content": f"处理文件时发生错误: {str(e)}",
                "error": str(e)
            }

    def _prepare_file_data(self, file_path: str, mime_type: str) -> Dict[str, Any]:
        """准备文件数据"""
        try:
            with open(file_path, "rb") as f:
                file_content = f.read()
                # 通义千问支持base64编码的文件
                b64_content = base64.b64encode(file_content).decode("utf-8")
                # 提取文件内容（对于文本类文件）
                extracted_text = self._extract_text_content(file_path, file_content, mime_type)
                return {
                    "file_name": os.path.basename(file_path),
                    "mime_type": mime_type,
                    "content": b64_content
                }
        except Exception as e:
            logger.error(f"准备文件数据失败 {file_path}: {e}")
            raise

    def _extract_text_content(self, file_path: str, file_content: bytes, mime_type: str) -> Optional[str]:
        """从文件中提取文本内容"""
        try:
            # 根据文件类型提取文本
            if mime_type.startswith("text/") or file_path.endswith((".txt", ".md", ".text")):
                # 文本文件
                return file_content.decode('utf-8', errors='replace')

            elif mime_type == "application/pdf" or file_path.endswith(".pdf"):
                # PDF文件
                try:
                    import PyPDF2
                    with io.BytesIO(file_content) as pdf_file:
                        reader = PyPDF2.PdfReader(pdf_file)
                        text = ""
                        for page_num in range(len(reader.pages)):
                            text += reader.pages[page_num].extract_text() + "\n"
                        return text
                except ImportError:
                    logger.warning("PyPDF2库未安装，无法解析PDF文件内容")
                    return None
                except Exception as e:
                    logger.error(f"解析PDF文件失败: {e}")
                    return None

            elif mime_type in ["application/vnd.ms-excel",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"] or file_path.endswith(
                (".xls", ".xlsx")):
                # Excel文件
                try:
                    import pandas as pd
                    with io.BytesIO(file_content) as excel_file:
                        df = pd.read_excel(excel_file)
                        return df.to_string(index=False)
                except ImportError:
                    logger.warning("pandas库未安装，无法解析Excel文件内容")
                    return None
                except Exception as e:
                    logger.error(f"解析Excel文件失败: {e}")
                    return None

            elif mime_type == "text/csv" or file_path.endswith(".csv"):
                # CSV文件
                try:
                    import pandas as pd
                    with io.BytesIO(file_content) as csv_file:
                        df = pd.read_csv(csv_file)
                        return df.to_string(index=False)
                except ImportError:
                    logger.warning("pandas库未安装，无法解析CSV文件内容")
                    return None
                except Exception as e:
                    logger.error(f"解析CSV文件失败: {e}")
                    return None

            elif mime_type in ["application/msword",
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document"] or file_path.endswith(
                (".doc", ".docx", ".dot")):
                # Word文档
                try:
                    import docx
                    with io.BytesIO(file_content) as doc_file:
                        doc = docx.Document(doc_file)
                        return "\n".join([para.text for para in doc.paragraphs])
                except ImportError:
                    logger.warning("python-docx库未安装，无法解析Word文件内容")
                    return None
                except Exception as e:
                    logger.error(f"解析Word文件失败: {e}")
                    return None

            # 其他文件类型不提取文本
            return None

        except Exception as e:
            logger.error(f"提取文件文本失败: {e}")
            return None

    def _guess_mime_type(self, file_path: str) -> str:
        """根据文件扩展名猜测MIME类型"""
        ext = os.path.splitext(file_path)[1].lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
            ".pdf": "application/pdf",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".dot": "application/msword",
            ".txt": "text/plain",
            ".text": "text/plain",
            ".md": "text/markdown",
            ".csv": "text/csv",
            ".xls": "application/vnd.ms-excel",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        }
        return mime_map.get(ext, "application/octet-stream")

    def _build_prompt(self, query: str, files_data: List[Dict[str, Any]]) -> str:
        """构建模型提示"""
        # 基础提示
        prompt = f"""请分析以下文件并回答问题。

        问题: {query}
        
        """
        # 添加文件信息
        for i, file in enumerate(files_data):
            prompt += f"文件 {i + 1}: {file['file_name']} (类型: {file['mime_type']})\n"

            # 如果有提取的文本内容，添加到提示中
            if file.get('extracted_text'):
                # 限制文本长度，避免提示过长
                text = file['extracted_text']
                if len(text) > 3000:  # 限制每个文件提取的文本长度
                    text = text[:3000] + "...(内容过长已截断)"

                prompt += f"文件内容:\n{text}\n\n"

        return prompt

    async def _call_model(self, prompt: str, files_data: List[Dict[str, Any]]) -> str:
        """使用通义千问视觉模型处理文件"""
        try:
            # 构建消息内容
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "你是一个专业的图像分析助手，请详细分析图片内容。"}]
                }
            ]

            # 构建用户消息
            user_content = []

            # 添加图片
            for file in files_data:
                if file["mime_type"].startswith("image/"):
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{file['mime_type']};base64,{file['content']}"
                        }
                    })

            # 添加文本提示
            user_content.append({"type": "text", "text": prompt})

            # 添加完整的用户消息
            messages.append({
                "role": "user",
                "content": user_content
            })

            logger.info(f"正在调用千问视觉模型处理文件，提示长度: {len(prompt)}")

            # 调用API
            stream_response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                modalities=["text"]  # 只需要文本输出
            )

            # 收集流式响应
            full_response = ""

            async def process_stream():
                nonlocal full_response
                try:
                    # 迭代处理每个响应块
                    for chunk in stream_response:
                        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, 'content') and delta.content:
                                full_response += delta.content
                except Exception as e:
                    logger.error(f"处理流式响应时出错: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

            # 处理流式响应
            await process_stream()

            # 返回完整响应
            if full_response:
                logger.info(f"full_response: {full_response}")
                return full_response
            else:
                logger.warning("流式API响应中没有找到有效内容")
                return "无法处理图片，API返回为空"

        except Exception as e:
            logger.error(f"调用千问视觉模型失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
