from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from mlx_embeddings_server.main import app


@pytest.fixture
def client():
    # Patch ModelManager used in main.py lifespan to avoid loading the model during tests
    with patch("mlx_embeddings_server.main.ModelManager") as MockManager:
        # Configure the mock to return a dummy instance if needed,
        # though lifespan only calls get_instance()
        MockManager.get_instance.return_value = MagicMock()

        with TestClient(app) as c:
            yield c
