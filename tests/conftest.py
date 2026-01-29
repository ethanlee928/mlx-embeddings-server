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
        mock_instance = MagicMock()
        mock_instance.model_id = "colqwen2.5"
        MockManager.get_instance.return_value = mock_instance

        with TestClient(app) as c:
            yield c


def pytest_addoption(parser):
    parser.addoption("--run-e2e", action="store_true", default=False, help="run end-to-end integration tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: mark test as end-to-end integration test")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-e2e"):
        # --run-e2e given in cli: do not skip e2e tests
        return
    skip_e2e = pytest.mark.skip(reason="need --run-e2e option to run")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip_e2e)
