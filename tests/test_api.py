from fastapi.testclient import TestClient

from highdimensionsmap import HDMScanner, MotionNoiseTracker, app, hdm
from highdimensionsmap.api import app as api_app

client = TestClient(api_app)


def test_healthcheck():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_transform_endpoint():
    payload = {"input": [0.1, 0.2, 0.3, 0.4]}
    response = client.post("/transform", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "signature" in data
    assert isinstance(data["signature"], list)


def test_query_endpoint():
    payload = {
        "dataset": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        "query": [0.1, 0.2, 0.3, 0.4],
        "k": 1,
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "indices" in data
    assert "distances" in data


def test_generate_high_dimensional_dataset():
    scanner = HDMScanner(input_dim=12, latent_modes=6, steps=18, seed=7)
    dataset = scanner.generate_dataset(5)
    assert dataset.shape == (5, 12)
    signature = scanner.transform(dataset)
    assert signature.shape[0] == 5


def test_tracker_and_public_exports():
    scanner = HDMScanner(input_dim=10, latent_modes=4, steps=12, seed=11)
    tracker = MotionNoiseTracker(scanner)
    trajectory = scanner.generate_dataset(8)
    result = tracker.track(trajectory)
    assert "signatures" in result
    assert "velocity" in result
    assert "noise" in result
    assert isinstance(hdm, type)
