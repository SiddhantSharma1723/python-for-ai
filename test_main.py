from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_predict():
    response = client.post(
        "/predict",
        json = {   
            "hours_studied": 6,
            "attendance_pct": 85,
            "prev_score": 75,
            "sleep_hours": 7
        }
    )
        
    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "probability" in data

    assert data["prediction"] in [0,1]
    assert 0 <= data["probability"] <=1


def test_invalid_input():
    response = client.post(
        "/predict",
        json = {   
            "hours_studied": "six",
            "attendance_pct": 85,
            "prev_score": 75,
            "sleep_hours": 7
        }
    )
        
    assert response.status_code == 422

def test_invalid_range():
    response = client.post(
        "/predict",
        json = {
            "hours_studied": "6",
            "attendance_pct": 110,
             "prev_score": 75,
            "sleep_hours": 7
            }
        )

def test_valid_predict():
    response = client.post(
        "/predict",
        json = {   
            "hours_studied": 6,
            "attendance_pct": 85,
            "prev_score": 75,
            "sleep_hours": 7
        }
    )
        
    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [0,1]
    assert 0 <= result["probability"] <= 1

def test_missing_predict():
    response = client.post(
        "/predict",
        json = {
            "hours_studied": 6,
            "attendance_pct": 80,
            "prev_score": 70
        }
    )

    assert response.status_code == 422