import logging
from typing import List, Dict, Optional

from app.db.repositories import FileRepository
from app.schemas.chat import ChatResponse, Message


class FileContentService:
    """处理聊天文件内容的服务类"""
    
    def __init__(self, db):
        self.db = db
        self.file_repo = FileRepository(db)
    
    def get_files_content(self, file_ids: List[str]) -> Dict[str, str]:
        """获取文件解析后的内容"""
        if not file_ids or len(file_ids) == 0:
            return {}
            
        # 获取文件解析结果
        file_contents = self.file_repo.get_parsed_file_content(file_ids)
        return file_contents
    
    def check_files_status(self, file_ids: List[str], provider: str, model: str, conversation_id: str) -> Optional[ChatResponse]:
        """检查文件状态，如果有文件未完成解析则返回相应响应"""
        if not file_ids or len(file_ids) == 0:
            return None
            
        # 获取文件解析结果
        file_contents = self.file_repo.get_parsed_file_content(file_ids)
        
        # 检查是否所有文件都已解析
        for file_id in file_ids:
            if file_id not in file_contents:
                file = self.file_repo.get_file_by_id(file_id)
                if file and file.status == "parsing":
                    # 文件仍在解析中
                    return ChatResponse(
                        id="file_parsing",
                        provider=provider,
                        model=model,
                        message=Message(
                            role="assistant",
                            content="文件正在解析中，请稍后再试..."
                        ),
                        conversation_id=conversation_id
                    )
                elif file and file.status == "error":
                    # 文件解析出错
                    return ChatResponse(
                        id="file_error",
                        provider=provider,
                        model=model,
                        message=Message(
                            role="assistant",
                            content=f"文件处理出错: {file.processing_result.get('message', '未知错误')}"
                        ),
                        conversation_id=conversation_id
                    )
        
        # 所有文件都已准备就绪
        return None 