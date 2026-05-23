"""Integration tests for IndustrialSentinel API endpoints."""
import pytest
import requests

API_URL = "http://localhost:8000"


def api_available():
    """Check if API is running."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


pytestmark = pytest.mark.skipif(
    not api_available(), reason="API not running at localhost:8000"
)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self):
        r = requests.get(f"{API_URL}/health")
        assert r.status_code == 200

    def test_health_models_loaded(self):
        r = requests.get(f"{API_URL}/health")
        data = r.json()["data"]
        assert data["models_loaded"] is True

    def test_health_latency_reasonable(self):
        r = requests.get(f"{API_URL}/health")
        data = r.json()["data"]
        assert data["test_inference_latency_ms"] < 200

    def test_health_has_model_version(self):
        r = requests.get(f"{API_URL}/health")
        data = r.json()["data"]
        assert "model_version" in data
        assert data["model_version"] != "unknown"


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_valid_input(self):
        payload = {
            "unit": 1,
            "readings": [[518.67, 641.82, 1589.7, 1400.6, 14.62, 21.61,
                          554.36, 2388.02, 9046.19, 1.3, 47.47, 521.66,
                          2388.02, 8138.62, 8.4195, 0.03, 392, 2388,
                          100.0, 39.06, 23.419]]
        }
        r = requests.post(f"{API_URL}/predict", json=payload)
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["predicted_rul"] > 0
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert 0 <= data["anomaly_score"] <= 1.0
        assert isinstance(data["alert"], bool)
        assert data["recommendation"] != ""

    def test_predict_malformed_input_returns_422(self):
        payload = {"unit": 1, "readings": [[1, 2, 3]]}
        r = requests.post(f"{API_URL}/predict", json=payload)
        assert r.status_code == 422

    def test_predict_missing_fields_returns_422(self):
        r = requests.post(f"{API_URL}/predict", json={"unit": 1})
        assert r.status_code == 422

    def test_predict_response_envelope(self):
        payload = {
            "unit": 1,
            "readings": [[518.67, 641.82, 1589.7, 1400.6, 14.62, 21.61,
                          554.36, 2388.02, 9046.19, 1.3, 47.47, 521.66,
                          2388.02, 8138.62, 8.4195, 0.03, 392, 2388,
                          100.0, 39.06, 23.419]]
        }
        r = requests.post(f"{API_URL}/predict", json=payload)
        body = r.json()
        assert body["status"] == "ok"
        assert "data" in body
        assert "timestamp" in body


class TestDemoEndpoint:
    """Tests for /demo/{engine_id} endpoint."""

    def test_demo_valid_engine(self):
        r = requests.get(f"{API_URL}/demo/1")
        assert r.status_code == 200
        data = r.json()["data"]
        assert "predicted_rul" in data
        assert "true_rul" in data
        assert data["engine_id"] == 1

    def test_demo_invalid_engine_returns_422(self):
        r = requests.get(f"{API_URL}/demo/999")
        assert r.status_code == 422

    def test_demo_zero_engine_returns_422(self):
        r = requests.get(f"{API_URL}/demo/0")
        assert r.status_code == 422

    def test_demo_high_rul_engine_is_low_risk(self):
        """Engine 1 has true RUL=112, should be LOW risk."""
        r = requests.get(f"{API_URL}/demo/1")
        data = r.json()["data"]
        assert data["risk_level"] == "LOW"

    def test_demo_low_rul_engine_is_high_risk(self):
        """Engine 100 has true RUL=20, should be HIGH or CRITICAL."""
        r = requests.get(f"{API_URL}/demo/100")
        data = r.json()["data"]
        assert data["risk_level"] in ["HIGH", "CRITICAL"]


class TestResultsEndpoint:
    """Tests for /results endpoint."""

    def test_results_returns_metrics(self):
        r = requests.get(f"{API_URL}/results")
        assert r.status_code == 200
        data = r.json()["data"]
        assert "rmse" in data
        assert "mae" in data
        assert "nasa_score" in data
        assert data["rmse"] < 16.0
        assert data["nasa_score"] < 500


class TestDriftEndpoint:
    """Tests for /drift endpoint."""

    def test_drift_returns_report(self):
        payload = {
            "unit": 1,
            "readings": [[518.67, 641.82, 1589.7, 1400.6, 14.62, 21.61,
                          554.36, 2388.02, 9046.19, 1.3, 47.47, 521.66,
                          2388.02, 8138.62, 8.4195, 0.03, 392, 2388,
                          100.0, 39.06, 23.419]] * 30
        }
        r = requests.post(f"{API_URL}/drift", json=payload)
        assert r.status_code == 200
        data = r.json()["data"]
        assert "sensors_checked" in data
        assert "sensors_drifted" in data
        assert data["sensors_checked"] == 14
