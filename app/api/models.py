from fastapi import APIRouter, HTTPException
from app.ai.llm_manager import llm_manager
from typing import List, Dict

router = APIRouter()


@router.get("/available", response_model=List[str])
def get_available_models():
    """获取所有可用的AI模型列表"""
    try:
        models = llm_manager.list_available_models()
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info/{model_name}")
def get_model_info(model_name: str):
    """获取特定模型的详细信息"""
    try:
        # 这里可以根据不同模型返回不同的信息
        model_info = {
            "wenxin": {
                "name": "文心一言",
                "version": "ERNIE-Bot-4",
                "capabilities": ["文本生成", "多轮对话", "创意写作", "代码生成"],
                "max_tokens": 8000
            },
            "qianwen": {
                "name": "通义千问",
                "version": "Qianfan-Chinese-Llama-2-7B",
                "capabilities": ["文本生成", "多轮对话", "知识问答"],
                "max_tokens": 4000
            },
            # "claude": {
            #     "name": "Claude",
            #     "version": "claude-3-sonnet-20240229",
            #     "capabilities": ["文本生成", "多轮对话", "创意写作", "代码生成", "推理"],
            #     "max_tokens": 32000
            # },
            "deepseek": {
                "name": "Deepseek",
                "version": "deepseek-chat",
                "capabilities": ["文本生成", "多轮对话", "代码生成"],
                "max_tokens": 16000
            }
        }

        if model_name not in model_info:
            raise HTTPException(status_code=404, detail=f"找不到模型 {model_name} 的信息")

        return model_info[model_name]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))