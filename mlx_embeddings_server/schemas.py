from typing import List, Optional, Union

from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(..., description="The input text(s) or image URL(s) to embed.")
    model: Optional[str] = Field("qnguyen3/colqwen2.5-v0.2-mlx", description="The model ID.")
    encoding_format: Optional[str] = Field("float", description="Format of the embeddings (float or base64).")
    user: Optional[str] = None


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    index: int
    # Allow List[List[float]] for multi-vector embeddings (ColBERT/ColQwen)
    embedding: Union[List[float], List[List[float]]]


class Usage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingObject]
    model: str
    usage: Usage
