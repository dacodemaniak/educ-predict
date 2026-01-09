import pytest
from fastapi.testclient import TestClient
from backend.student_api import app

client = TestClient(app)

# Mock valid datas
VALID_STUDENT = {
    "school": "GP", "sex": "F", "age": 18, "address": "U", "famsize": "GT3", "Pstatus": "T",
    "Medu": 4, "Fedu": 4, "Mjob": "teacher", "Fjob": "services", "reason": "course", "guardian": "mother",
    "traveltime": 1, "studytime": 2, "failures": 0, "schoolsup": "yes", "famsup": "no", "paid": "no",
    "activities": "no", "nursery": "yes", "higher": "yes", "internet": "yes",
    "famrel": 4, "freetime": 3, "goout": 4, "absences": 6, "G1": 12, "G2": 11
}

def test_predict_success():
    """Check enpoint returns correct response with a valid strategy"""
    # Note: accuracy model must be present in model folder
    with TestClient(app) as client:
        response = client.post("/predict/accuracy", json=VALID_STUDENT)
        assert response.status_code == 200
        data = response.json()
        assert "is_failure" in data
        assert "probability" in data

def test_predict_invalid_age():
    """Check endpoint reject over age"""
    invalid_data = VALID_STUDENT.copy()
    invalid_data["age"] = 99
    with TestClient(app) as client:
        response = client.post("/predict/accuracy", json=invalid_data)
        assert response.status_code == 422

def test_predict_forbidden_data():
    """Check sensitive data sent raised an error"""
    invalid_data = VALID_STUDENT.copy()
    invalid_data["romantic"] = "yes" # No more present in model
    with TestClient(app) as client:
        response = client.post("/predict/accuracy", json=invalid_data)
        
        assert response.status_code == 422

def test_health_check():
    """Health check test"""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200