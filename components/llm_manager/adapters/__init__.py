from typing import Any
from components.llm_manager.adapters.base import BaseLLMAdapter
from components.llm_manager.adapters.default import DefaultModelAdapter
from components.llm_manager.adapters.gpt_oss import GptOssAdapter

def get_model_adapter(model_name: str, client: Any) -> BaseLLMAdapter:
    """Factory selecting the appropriate adapter for the given model."""
    if model_name and "gpt-oss" in model_name.lower():
        return GptOssAdapter(model_name=model_name, client=client)
    return DefaultModelAdapter(model_name=model_name, client=client)

__all__ = ["BaseLLMAdapter", "DefaultModelAdapter", "GptOssAdapter", "get_model_adapter"]
