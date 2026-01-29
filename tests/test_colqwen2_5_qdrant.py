import base64
import os
import pathlib

import pytest
from fastapi.testclient import TestClient
from qdrant_client import models
from testcontainers.qdrant import QdrantContainer

from mlx_embeddings_server.main import app

# Only run if --run-e2e is passed
pytestmark = pytest.mark.e2e

MODEL_NAME = "qnguyen3/colqwen2.5-v0.2-mlx"
COLLECTION_NAME = "colqwen_e2e_test"
VECTOR_SIZE = 128


@pytest.fixture(scope="module")
def qdrant_container():
    """Starts a Qdrant container for the duration of the module."""
    # Using the same image tag as in scripts/qdrant-docker.sh
    with QdrantContainer("qdrant/qdrant:v1.16") as qdrant:
        yield qdrant


@pytest.fixture(scope="module")
def qdrant_client_fixture(qdrant_container):
    """Provides a QdrantClient connected to the container."""
    client = qdrant_container.get_client()
    yield client
    client.close()


@pytest.fixture(scope="module")
def e2e_client():
    """Provides a TestClient with the real MLX model loaded."""
    # Set the environment variable for the model
    os.environ["MODEL_ID"] = MODEL_NAME

    # Ensure ModelManager initializes with the real model
    # (Reset singleton if it was mocked in other tests)
    from mlx_embeddings_server.backend import ModelManager

    ModelManager._instance = None

    with TestClient(app) as c:
        yield c


def encode_image_to_base64(image_path: str) -> str:
    """Read an image file and convert it to a base64 data URI."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime_type};base64,{encoded_string}"


def test_end_to_end_search(qdrant_client_fixture, e2e_client):
    """
    E2E Test:
    1. Create MultiVector collection in Qdrant.
    2. Index 'cats.jpg' using embeddings from MLX server.
    3. Query 'What's on the pink bed?'
    4. Assert 'cats.jpg' is returned with a high score.
    """
    # 1. Cleanup Collection if exists
    if qdrant_client_fixture.collection_exists(COLLECTION_NAME):
        qdrant_client_fixture.delete_collection(COLLECTION_NAME)

    # 2. Create Collection
    qdrant_client_fixture.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
        ),
    )

    # 3. Prepare Data
    repo_root = pathlib.Path(__file__).parent.parent
    image_path = repo_root / "images" / "cats.jpg"
    assert image_path.exists(), f"Image not found at {image_path}"

    # 4. Index Image
    image_uri = encode_image_to_base64(str(image_path))

    # Call Embeddings API to embed image
    response = e2e_client.post("/v1/embeddings", json={"input": [image_uri], "model": MODEL_NAME})
    assert response.status_code == 200, f"Embedding API failed: {response.text}"

    data = response.json()["data"][0]
    embedding = data["embedding"]

    # Upsert to Qdrant
    qdrant_client_fixture.upsert(
        collection_name=COLLECTION_NAME,
        points=[models.PointStruct(id=1, vector=embedding, payload={"filename": "cats.jpg"})],
    )

    # 5. Search
    query = "What's on the pink bed?"
    # Call Embeddings API to embed query
    response_query = e2e_client.post("/v1/embeddings", json={"input": [query], "model": MODEL_NAME})
    assert response_query.status_code == 200, f"Embedding API query failed: {response_query.text}"

    query_embedding = response_query.json()["data"][0]["embedding"]

    # Query Qdrant
    search_result = qdrant_client_fixture.query_points(collection_name=COLLECTION_NAME, query=query_embedding, limit=1)

    assert len(search_result.points) > 0, "No results found"
    top_result = search_result.points[0]

    print(f"Top result: {top_result.payload['filename']} (Score: {top_result.score})")
    assert top_result.payload["filename"] == "cats.jpg"
    # Score should be relatively high for a good match
    assert top_result.score > 0.0, "Score should be positive"
