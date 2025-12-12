# Customer Churn Prediction System

This project is a complete machine learning system designed to predict churn risk. It uses a Random Forest classifier trained on the Telco Customer Churn dataset and exposes predictions via a FastAPI backend and a responsive HTML/JS frontend.

## Project Structure
- `src/api`: FastAPI backend application.
- `src/model`: Scripts for data preprocessing and model training.
- `src/web`: Frontend HTML dashboard.
- `data`: Directory for storing the dataset.
- `Dockerfile` & `docker-compose.yml`: Containerization configuration.

## How to Run Locally

### 1. Prerequisites
Ensure you have Python installed (version 3.9 or higher recommended).

### 2. Install Dependencies
Open your terminal in the project root and run:
```bash
pip install -r requirements.txt
```

### 3. Train the Model
Before running the API, you need to train the model and generate the artifacts (`model.pkl`):
```bash
python src/model/train.py
```
*Note: This script will verify the dataset exists in `data/` and train the model.*

### 4. Start the API Server
Run the FastAPI server using Uvicorn:
```bash
uvicorn src.api.main:app --reload
```
- The API will start at: `http://localhost:8000`
- Interactive API Docs: `http://localhost:8000/docs`

### 5. Access the Frontend
Open the `src/web/index.html` file directly in your web browser. 
- You do not need a separate web server for the frontend in local mode.
- Simply double-click the file or drag it into your browser window.

## Running with Docker
If you prefer using Docker, you can run the entire stack with a single command:
```bash
docker-compose up --build
```
- **Frontend**: http://localhost:8080
- **API**: http://localhost:8000
