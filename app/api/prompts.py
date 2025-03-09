import os
from typing import List

from fastapi import APIRouter, Depends
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
