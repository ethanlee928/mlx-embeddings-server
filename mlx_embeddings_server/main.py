from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException

from mlx_embeddings_server.backend import ModelManager, get_embeddings
from mlx_embeddings_server.schemas import (
    EmbeddingObject,
    EmbeddingRequest,
    EmbeddingResponse,
    Usage,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    ModelManager.get_instance()
    yield


app = FastAPI(title="MLX Embeddings Server", lifespan=lifespan)


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    inputs = request.input
    if isinstance(inputs, str):
        inputs = [inputs]

    try:
        # We invoke the blocking MLX code.
        # In a production async server, heavy MLX/Torch compute should be offloaded to a threadpool
        # or separate process to avoid blocking the event loop.
        # For simplicity in this `uv` setup, we run it directly,
        # but wrapping in `fastapi.concurrency.run_in_threadpool` is better practice.
        from fastapi.concurrency import run_in_threadpool

        embeddings = await run_in_threadpool(get_embeddings, inputs)
    except Exception as e:
        import traceback

        traceback.print_exc()
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
    uvicorn.run("mlx_embeddings_server.main:app", host="0.0.0.0", port=8000, reload=True)
