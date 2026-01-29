import logging
from contextlib import asynccontextmanager
from time import monotonic

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from mlx_embeddings_server.backend import ModelManager, get_embeddings
from mlx_embeddings_server.schemas import (
    EmbeddingObject,
    EmbeddingRequest,
    EmbeddingResponse,
    Usage,
)

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ModelManager.get_instance()
    yield


app = FastAPI(title="MLX Embeddings Server", lifespan=lifespan)


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    inputs = request.input
    if isinstance(inputs, str):
        inputs = [inputs]

    try:
        t1 = monotonic()
        logger.info(f"Processing {len(inputs)} inputs for embeddings")
        embeddings = await run_in_threadpool(get_embeddings, inputs)
        t2 = monotonic()
        logger.info(f"Processing {len(inputs)} inputs for embeddings [COMPLETED in {(t2 - t1) * 1000:.2f} ms]")
    except Exception as e:
        logger.exception("Error processing embeddings")
        raise HTTPException(status_code=500, detail=str(e))

    data = []
    for i, embed in enumerate(embeddings):
        data.append(EmbeddingObject(index=i, embedding=embed))

    return EmbeddingResponse(
        data=data, model=request.model or "qnguyen3/colqwen2.5-v0.2-mlx", usage=Usage(prompt_tokens=0, total_tokens=0)
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("mlx_embeddings_server.main:app", host="0.0.0.0", port=8888, reload=True)
