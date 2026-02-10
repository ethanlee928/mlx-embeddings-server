import base64
import os
import pathlib

import pytest
import torch
from fastapi.testclient import TestClient

from mlx_embeddings_server.main import app

# Only run if --run-e2e is passed
pytestmark = pytest.mark.e2e

MODEL_NAME = "mlx-community/siglip-so400m-patch14-384"


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


def test_siglip_end_to_end_search(e2e_client):
    """
    E2E Test for SigLIP:
    1. Embed 'cats.jpg' using MLX server.
    2. Embed query 'A photo of cats'.
    3. Embed negative query 'A photo of a car'.
    4. Assert dot product(image, positive_query) > dot product(image, negative_query).
    """
    # 1. Prepare Data
    repo_root = pathlib.Path(__file__).parent.parent
    image_path = repo_root / "images" / "cats.jpg"
    assert image_path.exists(), f"Image not found at {image_path}"

    image_uri = encode_image_to_base64(str(image_path))

    # 2. Index Image
    response = e2e_client.post("/v1/embeddings", json={"input": [image_uri], "model": MODEL_NAME})
    assert response.status_code == 200, f"Embedding API failed: {response.text}"
    image_embedding = response.json()["data"][0]["embedding"]

    # 3. Search
    positive_query = "A photo of cats"
    negative_query = "A photo of a car"

    response_query = e2e_client.post(
        "/v1/embeddings",
        json={"input": [positive_query, negative_query], "model": MODEL_NAME},
    )
    assert response_query.status_code == 200, f"Embedding API query failed: {response_query.text}"

    data = response_query.json()["data"]
    pos_embedding = data[0]["embedding"]
    neg_embedding = data[1]["embedding"]

    # 4. Compare
    image_tensor = torch.tensor(image_embedding)
    pos_tensor = torch.tensor(pos_embedding)
    neg_tensor = torch.tensor(neg_embedding)

    score_pos = (image_tensor @ pos_tensor).item()
    score_neg = (image_tensor @ neg_tensor).item()

    print(f"Positive Score ('A photo of cats'): {score_pos}")
    print(f"Negative Score ('A photo of a car'): {score_neg}")

    assert score_pos > score_neg, f"Positive match should be higher than negative. Pos: {score_pos}, Neg: {score_neg}"
    assert score_pos > 0.1, f"Score should be positive and reasonably high, got {score_pos}"
