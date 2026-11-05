import array
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from core.config import HOST, PORT, DEBUG, SCHEMA, MODEL_NAME
from core.logger import setup_logger
from components.llm_manager import llm_manager
from components.ai_manager import ai_manager
from components.db_manager import db_manager
from schemas.ai_schemas import (
    ChatRequest, ChatResponse,
    EmbedRequest, EmbedResponse,
    VectorSearchRequest, VectorSearchResponse, VectorMatchItem,
    TranscribeRequest, TranscribeResponse,
    SynthesizeRequest, SynthesizeResponse
)

logger = setup_logger("EnterpriseAI-recsys")

app = FastAPI(
    title="EnterpriseAI-recsys",
    description="Enterprise OCI Generative AI, OCI AI Speech, and Oracle Database 23ai AI Vector Search Engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["Health & Diagnostics"])
def health_check():
    """System health verification across OCI GenAI, Speech, and Oracle 23ai."""
    return {
        "status": "healthy",
        "engine": "EnterpriseAI-recsys",
        "model": MODEL_NAME,
        "database_schema": SCHEMA or "DEFAULT",
        "oci_genai_active": bool(llm_manager.client),
        "oci_speech_active": bool(ai_manager.speech_client)
    }

@app.post("/api/ai/chat", response_model=ChatResponse, tags=["OCI Generative AI"])
def generate_chat_response(req: ChatRequest):
    """
    Executes conversational inference through OCI Generative AI using specialized Model Adapters
    and automated Langfuse observability.
    """
    try:
        reply = llm_manager.llm_model(
            prompt=req.prompt,
            system_prompt=req.system_prompt,
            session_id=req.session_id,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            name="enterprise_chat"
        )
        return ChatResponse(
            success=True,
            response=reply,
            model=llm_manager.model_name,
            session_id=req.session_id
        )
    except Exception as e:
        logger.error(f"Chat generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/embed", response_model=EmbedResponse, tags=["OCI Generative AI"])
def generate_embeddings(req: EmbedRequest):
    """
    Generates dense vector embeddings (1024 dimensions) using OCI Generative AI
    hosting Cohere Embed v4.
    """
    try:
        res = llm_manager.embed_text(inputs=req.texts, truncate=req.truncate)
        embeddings = res.embeddings
        dim = len(embeddings[0]) if embeddings else 0
        return EmbedResponse(
            success=True,
            count=len(embeddings),
            dimensions=dim,
            embeddings=embeddings
        )
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/vector-match", response_model=VectorSearchResponse, tags=["Oracle 23ai AI Vector Search"])
def vector_search(req: VectorSearchRequest):
    """
    Performs in-database semantic similarity search against Oracle Autonomous Database 23ai
    using native VECTOR columns and VECTOR_DISTANCE(..., COSINE).
    """
    try:
        # 1. Compute query vector using OCI GenAI
        embed_res = llm_manager.embed_text(req.query)
        query_embedding = embed_res.embeddings[0]

        # 2. Sanitize to native float array for Oracle 23ai binding
        query_vec_array = array.array('f', query_embedding)

        # 3. In-database Cosine similarity search
        target_schema = SCHEMA or "RECSYS_NEW"
        sql = f"""
        SELECT DOC_ID, TITLE, CONTENT,
               VECTOR_DISTANCE(DOC_EMBEDDING, :1, COSINE) AS distance
        FROM {target_schema}.ENTERPRISE_DOCUMENTS_VECTOR
        ORDER BY distance ASC
        FETCH FIRST :2 ROWS ONLY
        """
        res = db_manager.run_query_with_columns(sql, [query_vec_array, req.top_k])
        
        matches = []
        if res.get("is_data_found") and res.get("data"):
            for row in res["data"]:
                dist = float(row.get("DISTANCE", 1.0))
                if req.threshold is None or dist <= req.threshold:
                    matches.append(VectorMatchItem(
                        doc_id=str(row.get("DOC_ID")),
                        title=str(row.get("TITLE")),
                        content=str(row.get("CONTENT")),
                        distance=dist
                    ))

        return VectorSearchResponse(
            success=True,
            query=req.query,
            matches_count=len(matches),
            matches=matches
        )
    except Exception as e:
        logger.error(f"Vector search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/speech/transcribe", response_model=TranscribeResponse, tags=["OCI AI Speech"])
def trigger_speech_transcription(req: TranscribeRequest):
    """Submits an asynchronous batch audio transcription job to OCI AI Speech."""
    try:
        job = ai_manager.create_transcription_job(
            file_name=req.audio_object_name,
            display_name=req.display_name
        )
        return TranscribeResponse(
            success=True,
            job_id=job.id,
            status=job.lifecycle_state
        )
    except Exception as e:
        logger.error(f"Speech transcription failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/speech/synthesize", response_model=SynthesizeResponse, tags=["OCI AI Speech"])
def synthesize_audio(req: SynthesizeRequest):
    """Synthesizes human-like speech from text using OCI AI Speech TTS directly to Object Storage."""
    try:
        result = ai_manager.synthesize_speech(
            text=req.text,
            voice_id=req.voice_id,
            output_format=req.output_format
        )
        return SynthesizeResponse(
            success=True,
            status=result["status"],
            object_name=result["object_name"],
            bucket_name=result["bucket_name"]
        )
    except Exception as e:
        logger.error(f"Speech synthesis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=DEBUG)
