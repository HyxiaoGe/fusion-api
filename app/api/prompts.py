from fastapi import APIRouter, Depends, HTTPException
from app.schemas.prompts import PromptTemplate
from typing import List
import json
import os

router = APIRouter()
PROMPTS_DIR = "./prompts"
os.makedirs(PROMPTS_DIR, exist_ok=True)

@router.get("/", response_model=List[PromptTemplate])
def get_prompts():
    """获取所有提示词模板"""
    prompts = []
    for filename in os.listdir(PROMPTS_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(PROMPTS_DIR, filename), "r", encoding="utf-8") as f:
                prompt_data = json.load(f)
                prompts.append(PromptTemplate(**prompt_data))
    return prompts

@router.post("/", response_model=PromptTemplate)
def create_prompt(prompt: PromptTemplate):
    """创建新的提示词模板"""
    with open(os.path.join(PROMPTS_DIR, f"{prompt.id}.json"), "w", encoding="utf-8") as f:
        json.dump(prompt.model_dump(), f, ensure_ascii=False, indent=2)
    return prompt