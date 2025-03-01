from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
from typing import List

app = FastAPI()
app.title = 'ML Yearly Sales Prection API'
app.version = '1.0.0'

# Pydantic schema to receive the features
class Features(BaseModel):
    features: List[float]

# Pydantic schema to return the prediction result
class PredictionResult(BaseModel):
    prediction: float

@app.post("/v1/predict/", response_model=PredictionResult, tags = ['Prediction'])
async def get_prediction(features: Features):
    # Open the pickle file in binary read mode
    try:
        with open('./models/model.pkl', 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model file not found")
    
    X = [features.features]  # Model expects a 2D array for prediction (list of lists)

    # Make prediction
    try:
        value = model.predict(X)  # Prediction is a numpy array or similar
        return PredictionResult(prediction=value[0])  # Assuming single output
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


