import asyncio
from typing import Dict, Any

from app.core.logger import app_logger as logger
from app.db.repositories import FileRepository
from app.processor.file_processor import FileProcessor

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
            
        logger.info(f"分析文件: {file_id}, 类型: {analysis_type}, 原始文件名: {file.original_filename}")
        
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
            else:
                # 文件未解析，使用处理器进行解析
                file_processor = FileProcessor()
                result = await file_processor.process_files(
                    [file.path], 
                    query="请提供这个文件的简短摘要", 
                    mime_types=[file.mimetype]
                )
                
                # 更新文件解析结果
                content = result.get("content", "无法解析文件内容")
                file_repo.update_file(
                    file_id=file_id, 
                    updates={
                        "parsed_content": content,
                        "status": "processed"
                    }
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
            file_processor = FileProcessor()
            result = await file_processor.process_files(
                [file.path], 
                query="请提取这个文件中的关键数据和信息，使用结构化格式" + (f": {query}" if query else ""), 
                mime_types=[file.mimetype]
            )
            
            return {
                "file_id": file_id,
                "file_name": file.original_filename,
                "mimetype": file.mimetype,
                "extracted_data": result.get("content", "无法提取数据"),
                "analysis_type": "extract_data"
            }
            
        elif analysis_type == "answer_questions":
            if not query:
                return {"error": "需要提供问题 (query 参数)"}
                
            # 回答关于文件的问题
            file_processor = FileProcessor()
            result = await file_processor.process_files(
                [file.path], 
                query=query, 
                mime_types=[file.mimetype]
            )
            
            return {
                "file_id": file_id,
                "file_name": file.original_filename,
                "mimetype": file.mimetype,
                "question": query,
                "answer": result.get("content", "无法回答问题"),
                "analysis_type": "answer_questions"
            }
            
        else:
            return {"error": f"不支持的分析类型: {analysis_type}"}
            
    except Exception as e:
        logger.error(f"文件分析处理器出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": f"文件分析失败: {str(e)}"}