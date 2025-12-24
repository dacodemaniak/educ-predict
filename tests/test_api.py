import pytest
from fastapi.testclient import TestClient
import os
import joblib
import pandas as pd
from student_api import app # API sets in student_api.py

client = TestClient(app)

def test_healthcheck():
    """Check API is running"""
    response = client.get("/")
    assert response.status_code in [200, 404]

def test_predict_success():
    """Valid prediction testing"""
    payload = {
        "school": "GP",
        "sex": "F",
        "age": 17,
        "address": "U",
        "studytime": 3,
        "failures": 0,
        "absences": 2,
        "G1": 15,
        "G2": 16
    }
    response = client.post("/predict", json=payload)
    
    # Successful response with prediction results
    if response.status_code == 200:
        data = response.json()
        assert "is_failure" in data
        assert "failure_probability" in data
        assert isinstance(data["is_failure"], bool)
    else:
        # If the model is not found (pure test environment)
        assert response.status_code == 503 

def test_predict_invalid_data():
    """Check that the API rejects incorrect data types"""
    payload = {
        "school": "GP",
        "age": "pas_un_nombre", # Should trigger a Pydantic validation error
        "G1": 12
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Unprocessable Entity

def test_predict_missing_fields():
    """Check that the API rejects incomplete payloads"""
    payload = {"school": "MS"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422