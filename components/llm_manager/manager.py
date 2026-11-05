"""
Unified LLM & Vector Embedding Manager for OCI Generative AI and Langfuse Observability.
- Employs Model Adapters to isolate model-specific behaviors (GPT-OSS vs Gemini/Llama).
- Zero-boilerplate automatic request, token, and latency tracing via langfuse.openai.
- OCI GenAI Inference client for dense vector embeddings (Cohere Embed v4).
"""
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union
from fastapi import HTTPException
import oci
from openai import OpenAI

try:
    from langfuse.openai import OpenAI as LangfuseOpenAI
    from langfuse import observe, propagate_attributes
    LANGFUSE_AVAILABLE = True
except ImportError:
    LangfuseOpenAI = OpenAI
    observe = lambda *args, **kwargs: (lambda f: f)
    propagate_attributes = None
    LANGFUSE_AVAILABLE = False

from core.config import (
    COMPARTMENT_ID,
    EMBEDDING_MODEL_ID,
    ENDPOINT,
    LANGFUSE_BASE_URL,
    LANGFUSE_PROJECT_FLAG,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_PROJECT_ID,
    MODEL_NAME,
    OCI_CONFIG_FILE,
    OCI_CONFIG_PROFILE,
    OCI_REGION,
)
from core.logger import get_request_context, get_user_context
from components.llm_manager.adapters import get_model_adapter

logger = logging.getLogger(__name__)

# Populate Langfuse environment variables for auto-tracing
if LANGFUSE_BASE_URL and not os.environ.get("LANGFUSE_HOST"):
    os.environ["LANGFUSE_HOST"] = LANGFUSE_BASE_URL
if LANGFUSE_PUBLIC_KEY and not os.environ.get("LANGFUSE_PUBLIC_KEY"):
    os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
if LANGFUSE_SECRET_KEY and not os.environ.get("LANGFUSE_SECRET_KEY"):
    os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY


class LLMManager:
    """Unified LLM Manager delegating model execution to specialized adapters."""

    def __init__(self):
        self.project_id = LLM_PROJECT_ID
        self.base_url = LLM_BASE_URL
        self.api_key = LLM_API_KEY
        self.model_name = MODEL_NAME
        self.compartment_id = COMPARTMENT_ID
        self.endpoint = ENDPOINT
        self.project_flag = LANGFUSE_PROJECT_FLAG

        # 1. Single unified OpenAI client wrapped with Langfuse auto-tracing
        if LANGFUSE_AVAILABLE and LANGFUSE_PUBLIC_KEY:
            self._client = LangfuseOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                project=self.project_id,
            )
        else:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                project=self.project_id,
            )
        self.client = self._client

        # 2. Select model-specific adapter
        self.adapter = get_model_adapter(self.model_name, self._client)
        logger.info(
            f"[llm_manager] Initialized model '{self.model_name}' using adapter: {self.adapter.__class__.__name__}"
        )

        # 3. Native OCI GenAI Inference client for dense Cohere vector embeddings
        self._init_oci_genai_client()

    def _init_oci_genai_client(self) -> None:
        try:
            if os.path.exists(OCI_CONFIG_FILE):
                self._oci_config = oci.config.from_file(OCI_CONFIG_FILE, OCI_CONFIG_PROFILE)
            else:
                self._oci_config = {}

            if OCI_REGION:
                self._oci_config["region"] = OCI_REGION

            client_kwargs: Dict[str, Any] = {"config": self._oci_config}
            if self.endpoint:
                client_kwargs["service_endpoint"] = self.endpoint

            self._genai_client = oci.generative_ai_inference.GenerativeAiInferenceClient(**client_kwargs)
            logger.info(f"[llm_manager] OCI Generative AI client initialized for dense vector embeddings.")
        except Exception as e:
            logger.warning(f"[llm_manager] Could not initialize native OCI GenAI client: {e}")
            self._genai_client = None

    def llm_model(
        self,
        prompt: Union[str, List[Any]],
        system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        max_tokens: int = 2000,
        temperature: float = 1.0,
        top_p: float = 1.0,
        user_id: Optional[str] = None,
        project_flag: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Executes LLM inference via the assigned model adapter with Langfuse auto-tracing."""
        t0 = time.time()
        effective_user_id = str(user_id) if user_id else get_user_context()
        effective_project_flag = project_flag or self.project_flag
        trace_name = name or "llm_model"

        try:
            if propagate_attributes:
                tags = [effective_project_flag] if effective_project_flag else []
                metadata = {"model": self.model_name}
                if effective_project_flag:
                    metadata["project"] = effective_project_flag

                with propagate_attributes(
                    user_id=effective_user_id,
                    session_id=session_id,
                    tags=tags,
                    metadata=metadata,
                    trace_name=trace_name,
                ):
                    result = self.adapter.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        session_id=session_id,
                        tools=tools,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        name=trace_name,
                        **kwargs,
                    )
            else:
                result = self.adapter.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    session_id=session_id,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    name=trace_name,
                    **kwargs,
                )

            duration_ms = (time.time() - t0) * 1000
            logger.info(f"[llm_manager] Inference completed in {duration_ms:.1f}ms | output_chars={len(result)}")
            return result

        except Exception as e:
            duration_ms = (time.time() - t0) * 1000
            logger.error(f"[llm_manager] Inference failed after {duration_ms:.1f}ms: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

    def embed_text(
        self,
        inputs: Union[str, List[str]],
        model_id: Optional[str] = None,
        truncate: str = "NONE",
        user_id: Optional[str] = None,
        project_flag: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Generates dense vector embeddings using OCI Cohere Embed with Langfuse observation."""
        if not self._genai_client:
            raise HTTPException(
                status_code=500,
                detail="OCI Generative AI client is not initialized (check OCI configuration).",
            )

        t0 = time.time()
        effective_user_id = str(user_id) if user_id else get_user_context()
        effective_project_flag = project_flag or self.project_flag
        target_model = model_id or EMBEDDING_MODEL_ID

        normalized_inputs = [inputs] if isinstance(inputs, str) else list(inputs)
        input_count = len(normalized_inputs)

        genai_models = oci.generative_ai_inference.models
        detail = genai_models.EmbedTextDetails(
            serving_mode=genai_models.OnDemandServingMode(model_id=target_model),
            inputs=normalized_inputs,
            truncate=truncate,
            compartment_id=self.compartment_id,
        )

        try:
            if propagate_attributes:
                with propagate_attributes(
                    user_id=effective_user_id,
                    tags=["embedding", effective_project_flag] if effective_project_flag else ["embedding"],
                    metadata={"count": input_count, "model": target_model},
                    trace_name="embed_text",
                ):
                    response = self._genai_client.embed_text(detail)
            else:
                response = self._genai_client.embed_text(detail)

            duration_ms = (time.time() - t0) * 1000
            logger.info(f"[llm_manager] Generated {input_count} embeddings via {target_model} in {duration_ms:.1f}ms")
            return response.data
        except Exception as e:
            duration_ms = (time.time() - t0) * 1000
            logger.error(f"[llm_manager] Embedding generation failed after {duration_ms:.1f}ms: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Embedding Error: {str(e)}")

llm_manager = LLMManager()
