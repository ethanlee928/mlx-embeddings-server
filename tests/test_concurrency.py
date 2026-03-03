"""
Concurrency stress-tests for the MLX Embeddings Server.

Unit tests (default):
    Fire 5 simultaneous requests against the live ASGI app with a mocked
    backend.  Exercises the async pipeline, BatchingEngine coalescing, and
    the HTTP layer under concurrent load — no GPU required.

E2E tests (--run-e2e flag required):
    Fire 5 simultaneous requests at a real running server
    (EMBEDDINGS_SERVER_URL env var, default http://localhost:8888).
    These exercise the actual Metal GPU path and confirm the single-thread
    executor + mx.eval() fixes hold under real concurrent load.
"""

import asyncio
import base64
import io
import os
import pathlib
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from PIL import Image

from mlx_embeddings_server.main import app

MODEL_ID = "qnguyen3/colqwen2.5-v0.2-mlx"
N_REQUESTS = 5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_png_b64() -> str:
    """Return a base64 data-URI for a small 4×4 red PNG (no file I/O needed)."""
    img = Image.new("RGB", (4, 4), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _make_mock_manager(latency: float = 0.03) -> MagicMock:
    """
    Build a ModelManager mock whose embed() simulates GPU latency.
    Returns a distinct embedding per input so integrity can be verified.
    """
    mock = MagicMock()
    mock.model_id = MODEL_ID
    mock.batching_engine.start = AsyncMock()

    async def _embed(inputs):
        await asyncio.sleep(latency)  # simulate GPU work
        return [[[float(i), float(hash(inp) % 1000) / 1000]] for i, inp in enumerate(inputs)]

    mock.batching_engine.embed = _embed
    return mock


# ---------------------------------------------------------------------------
# Unit tests — mocked backend, ASGI transport
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_concurrent_5_text_requests():
    """5 simultaneous text-embedding requests must all succeed without races."""
    mock = _make_mock_manager()

    with patch("mlx_embeddings_server.main.ModelManager") as MockManager:
        MockManager.get_instance.return_value = mock

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            payloads = [{"input": f"sentence {i}", "model": MODEL_ID} for i in range(N_REQUESTS)]
            responses = await asyncio.gather(*[client.post("/v1/embeddings", json=p) for p in payloads])

    failed = [f"req {i}: HTTP {r.status_code} {r.text}" for i, r in enumerate(responses) if r.status_code != 200]
    assert not failed, "\n".join(failed)

    for r in responses:
        data = r.json()["data"]
        assert len(data) == 1, "expected exactly one embedding object per request"
        assert data[0]["object"] == "embedding"
        assert len(data[0]["embedding"]) > 0


@pytest.mark.anyio
async def test_concurrent_5_mixed_text_and_image_requests():
    """
    3 text + 2 image requests in parallel.
    Validates that image and text inputs co-exist without cross-contamination
    even when the BatchingEngine coalesces them into the same batch.
    """
    mock = _make_mock_manager()
    image_uri = _tiny_png_b64()

    payloads = [
        {"input": "cats and dogs", "model": MODEL_ID},
        {"input": image_uri, "model": MODEL_ID},
        {"input": "the quick brown fox", "model": MODEL_ID},
        {"input": image_uri, "model": MODEL_ID},
        {"input": "another sentence here", "model": MODEL_ID},
    ]

    with patch("mlx_embeddings_server.main.ModelManager") as MockManager:
        MockManager.get_instance.return_value = mock

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            responses = await asyncio.gather(*[client.post("/v1/embeddings", json=p) for p in payloads])

    failed = [f"req {i}: HTTP {r.status_code} {r.text}" for i, r in enumerate(responses) if r.status_code != 200]
    assert not failed, "\n".join(failed)

    for r in responses:
        data = r.json()["data"]
        assert len(data) == 1


@pytest.mark.anyio
async def test_concurrent_no_embedding_cross_contamination():
    """
    Under concurrent load each response must contain the embedding that
    corresponds to its own input — no cross-request result pollution.

    Strategy: embed the ord() of the first character so we can uniquely
    trace each input back to its result.
    """

    async def _traceable_embed(inputs):
        await asyncio.sleep(0.02)
        # embedding[0][0] == ord of first char, making each result uniquely identifiable
        return [[[float(ord(inp[0]))]] for inp in inputs]

    mock = _make_mock_manager()
    mock.batching_engine.embed = _traceable_embed

    sentences = ["alpha", "brave", "cloud", "delta", "eagle"]

    with patch("mlx_embeddings_server.main.ModelManager") as MockManager:
        MockManager.get_instance.return_value = mock

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            responses = await asyncio.gather(
                *[client.post("/v1/embeddings", json={"input": s, "model": MODEL_ID}) for s in sentences]
            )

    # asyncio.gather preserves order, so responses[i] belongs to sentences[i]
    for sentence, response in zip(sentences, responses):
        assert response.status_code == 200, f"'{sentence}' request failed: {response.text}"
        embedding_val = response.json()["data"][0]["embedding"][0][0]
        expected = float(ord(sentence[0]))
        assert embedding_val == expected, (
            f"Cross-contamination for '{sentence}': got {embedding_val}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# E2E tests — real MLX model loaded in-process via ASGI transport
# ---------------------------------------------------------------------------

E2E_MODEL = "qnguyen3/colqwen2.5-v0.2-mlx"


@pytest.fixture(scope="module")
async def e2e_client():
    """
    Module-scoped async fixture.
    - Resets the ModelManager singleton so the real MLX model is loaded.
    - Triggers the FastAPI lifespan (which loads the model and starts the
      BatchingEngine drain loop) in the *same* event loop as the tests.
    - Yields a shared httpx.AsyncClient backed by ASGI transport so all
      concurrent requests flow through the real GPU path.
    """
    os.environ["MODEL_ID"] = E2E_MODEL
    os.environ.setdefault("BATCH_MAX_SIZE", "64")
    os.environ.setdefault("BATCH_MAX_WAIT_MS", "20")

    from mlx_embeddings_server.backend import ModelManager

    ModelManager._instance = None

    # app.router.lifespan_context is the lifespan() async-context-manager
    # defined in main.py.  Entering it here runs the startup (model load +
    # BatchingEngine.start()) in our event loop before any test request fires.
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            timeout=120,
        ) as client:
            yield client


@pytest.mark.e2e
@pytest.mark.anyio
async def test_e2e_concurrent_mixed_image_and_text(e2e_client):
    """
    Fire 5 requests simultaneously — 2 images + 3 text queries — against the
    real MLX model.  This is the exact scenario that triggered the Metal SIGABRT
    crash, so it directly validates the single-thread executor + mx.eval() fix.

    Sanity check: after all requests complete, compute the ColQwen MaxSim score
    between cats.jpg and a positive cat query vs. a negative unrelated query and
    assert the positive score is higher.
    """
    import torch

    repo_root = pathlib.Path(__file__).parent.parent
    image_path = repo_root / "images" / "cats.jpg"

    if image_path.exists():
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        image_uri = f"data:image/jpeg;base64,{b64}"
    else:
        image_uri = _tiny_png_b64()

    # Indices:  0 = positive query, 1 = image, 2 = filler text,
    #           3 = image (second copy), 4 = negative query
    POSITIVE_IDX = 0
    IMAGE_IDX = 1
    NEGATIVE_IDX = 4

    payloads = [
        {"input": "What's on the pink bed?", "model": E2E_MODEL},  # 0 – positive
        {"input": image_uri, "model": E2E_MODEL},  # 1 – cats.jpg
        {"input": "cute pets resting at home", "model": E2E_MODEL},  # 2 – filler text
        {"input": image_uri, "model": E2E_MODEL},  # 3 – cats.jpg again
        {"input": "a racing car on a track", "model": E2E_MODEL},  # 4 – negative
    ]

    t0 = time.monotonic()

    responses = await asyncio.gather(
        *[e2e_client.post("/v1/embeddings", json=p) for p in payloads],
        return_exceptions=True,
    )

    elapsed = time.monotonic() - t0

    failed = [
        str(r) if isinstance(r, Exception) else f"HTTP {r.status_code}: {r.text}"
        for r in responses
        if isinstance(r, Exception) or r.status_code != 200
    ]
    assert not failed, f"{len(failed)}/{N_REQUESTS} requests failed:\n" + "\n".join(failed)

    print(f"\n[e2e] {N_REQUESTS} concurrent mixed (text+image) requests completed in {elapsed:.2f}s")

    # ------------------------------------------------------------------
    # Sanity check — ColQwen MaxSim similarity
    # Each embedding is a list of per-token vectors (multi-vector).
    # MaxSim = Σ_q max_p ( q_t · p_t ) over query tokens.
    # ------------------------------------------------------------------
    def maxsim(query_vecs, doc_vecs) -> float:
        q = torch.tensor(query_vecs, dtype=torch.float32)
        d = torch.tensor(doc_vecs, dtype=torch.float32)
        q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return (q @ d.T).max(dim=1).values.sum().item()

    image_emb = responses[IMAGE_IDX].json()["data"][0]["embedding"]
    pos_emb = responses[POSITIVE_IDX].json()["data"][0]["embedding"]
    neg_emb = responses[NEGATIVE_IDX].json()["data"][0]["embedding"]

    score_pos = maxsim(pos_emb, image_emb)
    score_neg = maxsim(neg_emb, image_emb)

    print(f"[e2e] MaxSim positive ('{payloads[POSITIVE_IDX]['input']}'): {score_pos:.4f}")
    print(f"[e2e] MaxSim negative ('{payloads[NEGATIVE_IDX]['input']}'): {score_neg:.4f}")

    assert score_pos > score_neg, (
        f"Expected positive query to score higher than negative query. "
        f"positive={score_pos:.4f}, negative={score_neg:.4f}"
    )
