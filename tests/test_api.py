from unittest.mock import patch

from fastapi.testclient import TestClient


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_embedding_single_text(client: TestClient):
    """Test embedding generation for a single text input string."""
    with patch("mlx_embeddings_server.main.get_embeddings") as mock_get_embeddings:
        # Mock return: One input, which is a list of vectors (simulating ColBERT/ColQwen output)
        # Structure: List[Input] -> List[Vectors] -> List[Float]
        mock_get_embeddings.return_value = [[[0.1, 0.2], [0.3, 0.4]]]

        payload = {"input": "This is a test", "model": "colqwen2.5"}
        response = client.post("/v1/embeddings", json=payload)

        assert response.status_code == 200
        data = response.json()

        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["model"] == "colqwen2.5"

        # Check first embedding object
        embed_obj = data["data"][0]
        assert embed_obj["index"] == 0
        assert embed_obj["object"] == "embedding"
        assert embed_obj["embedding"] == [[0.1, 0.2], [0.3, 0.4]]

        # Verify backend call inputs
        mock_get_embeddings.assert_called_once_with(["This is a test"])


def test_create_embedding_batch_text(client: TestClient):
    """Test embedding generation for a list of inputs."""
    with patch("mlx_embeddings_server.main.get_embeddings") as mock_get_embeddings:
        # Mock return for 2 inputs
        mock_get_embeddings.return_value = [
            [[0.1, 0.1]],  # Input 1
            [[0.2, 0.2]],  # Input 2
        ]

        payload = {"input": ["Hello", "World"]}
        response = client.post("/v1/embeddings", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2

        assert data["data"][0]["index"] == 0
        assert data["data"][0]["embedding"] == [[0.1, 0.1]]

        assert data["data"][1]["index"] == 1
        assert data["data"][1]["embedding"] == [[0.2, 0.2]]

        mock_get_embeddings.assert_called_once_with(["Hello", "World"])


def test_create_embedding_validation_error(client: TestClient):
    """Test validation error when input is missing."""
    payload = {"model": "colqwen2.5"}
    response = client.post("/v1/embeddings", json=payload)
    assert response.status_code == 422


def test_backend_exception_handling(client: TestClient):
    """Test that backend exceptions result in 500 errors."""
    with patch("mlx_embeddings_server.main.get_embeddings", side_effect=Exception("Model failure")):
        payload = {"input": "crash"}
        response = client.post("/v1/embeddings", json=payload)
        assert response.status_code == 500
        assert "Model failure" in response.json()["detail"]


def test_create_embedding_model_not_found(client: TestClient):
    payload = {"input": "test", "model": "wrong-model"}
    response = client.post("/v1/embeddings", json=payload)
    assert response.status_code == 404
    assert "Model 'wrong-model' not found" in response.json()["detail"]


def test_list_models(client: TestClient):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()

    assert data["object"] == "list"
    assert len(data["data"]) == 1

    model = data["data"][0]
    assert model["id"] == "colqwen2.5"
    assert model["object"] == "model"
    assert model["owned_by"] == "mlx-embeddings-server"
    assert isinstance(model["created"], int)


def test_retrieve_model(client: TestClient):
    # Test valid model
    response = client.get("/v1/models/colqwen2.5")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "colqwen2.5"
    assert data["object"] == "model"
    assert data["owned_by"] == "mlx-embeddings-server"

    # Test invalid model
    response = client.get("/v1/models/non-existent-model")
    assert response.status_code == 404
