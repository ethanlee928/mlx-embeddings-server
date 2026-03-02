import argparse
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from time import monotonic

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from mlx_embeddings_server.backend import ModelManager
from mlx_embeddings_server.schemas import (
    EmbeddingObject,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelListResponse,
    ModelObject,
    Usage,
)

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model in the thread pool so the event loop stays responsive
    await run_in_threadpool(ModelManager.get_instance)
    # Start the batching drain loop as a background task
    task = asyncio.create_task(ModelManager.get_instance().batching_engine.start())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="MLX Embeddings Server", lifespan=lifespan)


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    # Get the currently loaded model ID from the backend singleton
    loaded_model_id = ModelManager.get_instance().model_id

    return ModelListResponse(
        data=[
            ModelObject(
                id=loaded_model_id,
                owned_by="mlx-embeddings-server",
            )
        ]
    )


@app.get("/v1/models/{model_id}", response_model=ModelObject)
async def retrieve_model(model_id: str):
    loaded_model_id = ModelManager.get_instance().model_id

    if model_id != loaded_model_id:
        raise HTTPException(
            status_code=404, detail=f"Model '{model_id}' not found. Loaded model is '{loaded_model_id}'."
        )

    return ModelObject(
        id=loaded_model_id,
        owned_by="mlx-embeddings-server",
    )


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    loaded_model_id = ModelManager.get_instance().model_id
    if request.model and request.model != loaded_model_id:
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model}' not found. Loaded model is '{loaded_model_id}'."
        )

    inputs = request.input
    if isinstance(inputs, str):
        inputs = [inputs]

    try:
        t1 = monotonic()
        logger.info(f"Processing {len(inputs)} inputs for embeddings")
        embeddings = await ModelManager.get_instance().batching_engine.embed(inputs)
        t2 = monotonic()
        logger.info(f"Processing {len(inputs)} inputs for embeddings [COMPLETED in {(t2 - t1) * 1000:.2f} ms]")
    except Exception as e:
        logger.exception("Error processing embeddings")
        raise HTTPException(status_code=500, detail=str(e))

    data = []
    for i, embed in enumerate(embeddings):
        data.append(EmbeddingObject(index=i, embedding=embed))

    return EmbeddingResponse(data=data, model=loaded_model_id, usage=Usage(prompt_tokens=0, total_tokens=0))


@app.get("/health")
async def health():
    return {"status": "ok"}


def start():
    parser = argparse.ArgumentParser(description="MLX Embeddings Server")
    parser.add_argument("--model", type=str, help="Model ID to load", required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8888, help="Port to bind to")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of Uvicorn worker processes (default: 1; >1 duplicates the model in memory for each worker)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Uvicorn log level (default: info)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=64,
        help="Maximum number of inputs per batched inference call (default: 64)",
    )
    parser.add_argument(
        "--max-wait-ms",
        type=float,
        default=20.0,
        help="Maximum time in ms to wait for additional requests before dispatching a batch (default: 20)",
    )
    args = parser.parse_args()

    os.environ["MODEL_ID"] = args.model
    os.environ["BATCH_MAX_SIZE"] = str(args.max_batch_size)
    os.environ["BATCH_MAX_WAIT_MS"] = str(args.max_wait_ms)

    uvicorn.run(
        "mlx_embeddings_server.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    start()
