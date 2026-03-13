import json
from typing import Dict, Any

from app.core.logger import app_logger as logger
from app.db.repositories import FileRepository
from app.processor.file_processor import FileProcessor


def _normalize_content(content: Any) -> str:
    """将文件分析结果归一化为文本内容。"""
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


async def _run_file_analysis(file_processor: FileProcessor, file, query: str) -> str:
    """执行文件分析并在失败时抛出明确错误。"""
    result = await file_processor.process_files(
        [file.path],
        query=query,
        mime_types=[file.mimetype],
    )

    error_message = result.get("error")
    if error_message:
        raise RuntimeError(error_message)

    content = _normalize_content(result.get("content", ""))
    if not content:
        raise ValueError("无法解析文件内容")
    return content


async def analyze_file_handler(args: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    文件分析函数处理器
    
    参数:
        args: 函数参数，包含:
            - file_id: 要分析的文件ID
            - analysis_type: 分析类型 (summary, extract_data, answer_questions)
        context: 上下文信息，包含数据库连接
        
    返回:
        文件分析结果
    """
    try:
        # 提取参数
        file_id = args.get("file_id")
        if not file_id:
            return {"error": "文件ID不能为空"}
            
        analysis_type = args.get("analysis_type", "summary")
        query = args.get("query", "")
        
        # 获取数据库连接
        db = context.get("db")
        if not db:
            return {"error": "数据库连接未提供"}
            
        # 获取文件信息
        file_repo = FileRepository(db)
        file = file_repo.get_file_by_id(file_id)
        
        if not file:
            return {"error": f"文件不存在: {file_id}"}
            
        # 根据分析类型处理
        if analysis_type == "summary":
            if file.parsed_content:
                return {
                    "file_id": file_id,
                    "file_name": file.original_filename,
                    "mimetype": file.mimetype,
                    "summary": file.parsed_content,
                    "analysis_type": "summary"
                }

            content = await _run_file_analysis(
                FileProcessor(),
                file,
                "请提供这个文件的简短摘要",
            )
            file_repo.update_file(
                file_id=file_id,
                updates={
                    "parsed_content": content,
                    "status": "processed",
                },
            )

            return {
                "file_id": file_id,
                "file_name": file.original_filename,
                "mimetype": file.mimetype,
                "summary": content,
                "analysis_type": "summary"
            }
                
        elif analysis_type == "extract_data":
            # 提取文件中的数据
            content = await _run_file_analysis(
                FileProcessor(),
                file,
                "请提取这个文件中的关键数据和信息，使用结构化格式" + (f": {query}" if query else ""),
            )
            
            return {
                "file_id": file_id,
                "file_name": file.original_filename,
                "mimetype": file.mimetype,
                "extracted_data": content,
                "analysis_type": "extract_data"
            }
            
        elif analysis_type == "answer_questions":
            if not query:
                return {"error": "需要提供问题 (query 参数)"}
                
            # 回答关于文件的问题
            content = await _run_file_analysis(FileProcessor(), file, query)
            
            return {
                "file_id": file_id,
                "file_name": file.original_filename,
                "mimetype": file.mimetype,
                "question": query,
                "answer": content,
                "analysis_type": "answer_questions"
            }
            
        else:
            return {"error": f"不支持的分析类型: {analysis_type}"}
            
    except (RuntimeError, ValueError) as e:
        logger.warning(f"文件分析失败: {e}")
        return {"error": f"文件分析失败: {str(e)}"}
    except Exception as e:
        logger.exception("文件分析处理器异常详情")
        return {"error": f"文件分析失败: {str(e)}"}
