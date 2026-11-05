from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class ChatRequest(BaseModel):
    prompt: str = Field(..., description="User prompt or input query")
    system_prompt: Optional[str] = Field(None, description="System instructions for the AI model")
    session_id: Optional[str] = Field(None, description="Unique conversation session ID")
    max_tokens: int = Field(2000, description="Maximum tokens for generation")
    temperature: float = Field(0.7, description="Sampling temperature")

class ChatResponse(BaseModel):
    success: bool
    response: str
    model: str
    session_id: Optional[str] = None

class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., description="List of text strings to embed")
    truncate: str = Field("NONE", description="Truncation strategy: NONE, START, END")

class EmbedResponse(BaseModel):
    success: bool
    count: int
    dimensions: int
    embeddings: List[List[float]]

class VectorSearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(5, description="Number of nearest neighbors to retrieve")
    threshold: Optional[float] = Field(0.5, description="Maximum cosine distance threshold")

class VectorMatchItem(BaseModel):
    doc_id: str
    title: str
    content: str
    distance: float

class VectorSearchResponse(BaseModel):
    success: bool
    query: str
    matches_count: int
    matches: List[VectorMatchItem]

class TranscribeRequest(BaseModel):
    audio_object_name: str = Field(..., description="Path/object name inside OCI Object Storage bucket")
    display_name: Optional[str] = "BatchTranscriptionJob"

class TranscribeResponse(BaseModel):
    success: bool
    job_id: str
    status: str

class SynthesizeRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize into speech")
    voice_id: str = Field("Bob", description="Voice ID (e.g. Bob, Jenny)")
    output_format: str = Field("MP3", description="Audio format: MP3 or WAV")

class SynthesizeResponse(BaseModel):
    success: bool
    status: str
    object_name: str
    bucket_name: str
