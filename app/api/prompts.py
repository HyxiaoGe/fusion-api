import os
from typing import List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.repositories import PromptTemplateRepository
from app.schemas.prompts import PromptTemplate

router = APIRouter()
PROMPTS_DIR = "./prompts"
os.makedirs(PROMPTS_DIR, exist_ok=True)


@router.get("/", response_model=List[PromptTemplate])
def get_prompts(db: Session = Depends(get_db)):
    """获取所有提示词模板"""
    repo = PromptTemplateRepository(db=db)
    return repo.get_all()


@router.post("/", response_model=PromptTemplate)
def create_prompt(prompt: PromptTemplate, db: Session = Depends(get_db)):
    """创建新的提示词模板"""
    repo = PromptTemplateRepository(db=db)
    return repo.create(prompt)


@router.get("/{prompt_id}", response_model=PromptTemplate)
def get_prompt(prompt_id: str, db: Session = Depends(get_db)):
    """获取特定提示词模板"""
    repo = PromptTemplateRepository(db=db)
    prompt = repo.get_by_id(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="提示词模板未找到")
    return prompt


@router.put("/{prompt_id}", response_model=PromptTemplate)
def update_prompt(prompt_id: str, prompt_data: Dict[str, Any], db: Session = Depends(get_db)):
    """更新提示词模板"""
    repo = PromptTemplateRepository(db=db)
    prompt = repo.update(prompt_id, prompt_data)
    if not prompt:
        raise HTTPException(status_code=404, detail="提示词模板未找到")
    return prompt


@router.delete("/{prompt_id}")
def delete_prompt(prompt_id: str, db: Session = Depends(get_db)):
    """删除提示词模板"""
    repo = PromptTemplateRepository(db=db)
    success = repo.delete(prompt_id)
    if not success:
        raise HTTPException(status_code=404, detail="提示词模板未找到")
    return {"status": "success", "message": "提示词模板已删除"}


@router.post("/load")
def load_prompts_to_manager(db: Session = Depends(get_db)):
    """将所有提示词模板加载到提示词管理器中"""
    repo = PromptTemplateRepository(db=db)
    repo.load_to_prompt_manager()
    return {"status": "success", "message": "提示词模板已加载到管理器"}


@router.post("/sync-default-templates")
def sync_default_templates(db: Session = Depends(get_db)):
    """同步默认提示词模板到数据库"""
    from app.ai.prompts.templates import (
        GENERATE_TITLE_PROMPT,
        GENERATE_SUGGESTED_QUESTIONS_PROMPT,
        FILE_ANALYSIS_PROMPT,
        FILE_CONTENT_ENHANCEMENT_PROMPT,
        HOT_TOPIC_ANALYSIS_PROMPT,
        WEB_SEARCH_RESULTS_PROMPT
    )
    
    # 默认模板集合
    default_templates = [
        {"title": "生成标题", "content": GENERATE_TITLE_PROMPT, "tags": ["system", "title"]},
        {"title": "生成推荐问题", "content": GENERATE_SUGGESTED_QUESTIONS_PROMPT, "tags": ["system", "questions"]},
        {"title": "文件分析", "content": FILE_ANALYSIS_PROMPT, "tags": ["system", "file"]},
        {"title": "文件内容增强", "content": FILE_CONTENT_ENHANCEMENT_PROMPT, "tags": ["system", "file"]},
        {"title": "热点话题分析", "content": HOT_TOPIC_ANALYSIS_PROMPT, "tags": ["system", "topic"]},
        {"title": "提炼网页搜索结果", "content": WEB_SEARCH_RESULTS_PROMPT, "tags": ["system", "web"]}
    ]
    
    repo = PromptTemplateRepository(db=db)
    created_count = 0
    
    # 获取所有已有模板
    existing_templates = {template.title: template for template in repo.get_all()}
    
    for template_data in default_templates:
        # 如果已存在同名模板，则跳过
        if template_data["title"] in existing_templates:
            continue
            
        # 创建新模板
        template = PromptTemplate(
            title=template_data["title"],
            content=template_data["content"],
            tags=template_data["tags"]
        )
        repo.create(template)
        created_count += 1
    
    return {
        "status": "success", 
        "message": f"已同步 {created_count} 个默认提示词模板到数据库"
    }
