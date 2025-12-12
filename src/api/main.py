from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from fastapi.middleware.cors import CORSMiddleware

# Define the input schema matching the training features
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float # The frontend might send this as float directly

app = FastAPI(title="Customer Churn Prediction API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = os.path.join("src", "model", "model.pkl")
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}

@app.post("/predict")
def predict(data: CustomerData):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert dictionary to DataFrame
        input_df = pd.DataFrame([data.dict()])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        result = "Yes" if prediction == 1 else "No"
        
        # Simple reasoning logic (generic, based on common drivers)
        reasoning = []
        if data.Contract == "Month-to-month":
            reasoning.append("Month-to-month contracts have higher churn risk.")
        if data.tenure < 12:
            reasoning.append("New customers (low tenure) are more likely to churn.")
        if data.InternetService == "Fiber optic":
            reasoning.append("Fiber optic users often have higher churn due to competitive offers or price.")
        if probability > 0.7:
             reasoning.append("High probability score indicates strong churn signals.")
             
        return {
            "prediction": result,
            "probability": float(probability),
            "reasoning": reasoning
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
