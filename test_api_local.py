from fastapi.testclient import TestClient
from src.api.main import app
import json

client = TestClient(app)

def test_predict():
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }
    
    response = client.post("/predict", json=payload)
    print(response.status_code)
    print(response.json())
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()

if __name__ == "__main__":
    test_predict()
