import logging
from typing import Any, Dict, List, Optional, Union
from fastapi import HTTPException
from components.llm_manager.adapters.base import BaseLLMAdapter

logger = logging.getLogger(__name__)

class DefaultModelAdapter(BaseLLMAdapter):
    """Standard adapter using pure native OCI GenAI capabilities (Gemini, Llama, Cohere)."""

    def generate(
        self,
        prompt: Union[str, List[Any]],
        system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        max_tokens: int = 2000,
        temperature: float = 1.0,
        top_p: float = 1.0,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        instructions = system_prompt
        effective_input = prompt
        if isinstance(prompt, list):
            system_msgs = [
                m.get("content", "")
                for m in prompt
                if isinstance(m, dict) and m.get("role") == "system"
            ]
            if system_msgs and not instructions:
                instructions = system_msgs[0]
            effective_input = [
                m for m in prompt
                if not (isinstance(m, dict) and m.get("role") == "system")
            ]

        call_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "input": effective_input,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "name": name or "llm_model",
        }
        if instructions:
            call_kwargs["instructions"] = instructions
        if session_id:
            call_kwargs["conversation"] = session_id
        if tools:
            call_kwargs["tools"] = tools

        response = self.client.responses.create(**call_kwargs)

        text_result = getattr(response, "output_text", None)
        if not text_result and hasattr(response, "output") and response.output:
            extracted = []
            for item in response.output:
                for c in getattr(item, "content", []):
                    t = getattr(c, "text", "")
                    if t:
                        extracted.append(t)
            if extracted:
                text_result = "".join(extracted)

        if text_result and text_result.strip():
            return text_result.strip()

        raise HTTPException(status_code=500, detail="No response content from AI model")
