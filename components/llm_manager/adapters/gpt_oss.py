import logging
from typing import Any, Dict, List, Optional, Union
from fastapi import HTTPException
from components.llm_manager.adapters.base import BaseLLMAdapter

logger = logging.getLogger(__name__)

class GptOssAdapter(BaseLLMAdapter):
    """Adapter for openai.gpt-oss-120b with conversation bug mitigations."""

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
        if session_id:
            history = self._get_clean_history(session_id)
            if history:
                full_messages: List[Dict[str, Any]] = []
                full_messages.extend(history)
                if isinstance(prompt, list):
                    full_messages.extend(
                        [m for m in prompt if isinstance(m, dict) and m.get("role") != "system"]
                    )
                else:
                    full_messages.append({"role": "user", "content": str(prompt)})
                effective_input = full_messages
        elif isinstance(prompt, list):
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

        min_tokens = max(max_tokens, 150)

        call_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "input": effective_input,
            "max_output_tokens": min_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "name": name or "llm_model",
        }
        if instructions:
            call_kwargs["instructions"] = instructions
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
            clean_text = text_result.strip()
            if session_id:
                self._save_clean_turn(session_id, prompt, clean_text)
            return clean_text

        raise HTTPException(status_code=500, detail="No response content from AI model")

    def _get_clean_history(self, session_id: str) -> List[Dict[str, str]]:
        try:
            items_page = self.client.conversations.items.list(conversation_id=session_id)
            items_list = list(items_page.data) if hasattr(items_page, "data") else []
            history: List[Dict[str, str]] = []
            for item in reversed(items_list):
                role = getattr(item, "role", None)
                if role in ("user", "assistant"):
                    cnt = getattr(item, "content", "")
                    if isinstance(cnt, list):
                        text = "".join(getattr(x, "text", str(x)) for x in cnt)
                    elif hasattr(cnt, "text"):
                        text = getattr(cnt, "text", str(cnt))
                    else:
                        text = str(cnt)
                    if text and text.strip():
                        history.append({"role": role, "content": text.strip()})
            return history
        except Exception as e:
            logger.warning(f"[GptOssAdapter] Could not retrieve history for '{session_id}': {e}")
            return []

    def _save_clean_turn(self, session_id: str, prompt: Any, assistant_reply: str) -> None:
        try:
            user_text = prompt if isinstance(prompt, str) else str(prompt)
            self.client.conversations.items.create(
                conversation_id=session_id,
                items=[
                    {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
                    {"role": "assistant", "content": [{"type": "input_text", "text": assistant_reply}]},
                ],
            )
        except Exception as e:
            logger.warning(f"[GptOssAdapter] Failed to save turn to '{session_id}': {e}")
