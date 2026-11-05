from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class BaseLLMAdapter(ABC):
    """Abstract base adapter for model-specific request preparation and execution."""

    def __init__(self, model_name: str, client: Any):
        self.model_name = model_name
        self.client = client

    @abstractmethod
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
        """Executes generation against the model and returns clean output text."""
        pass
