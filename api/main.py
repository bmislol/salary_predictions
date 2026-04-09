from fastapi import FastAPI, Query, Depends, HTTPException
import joblib
import pandas as pd
from api.schemas import PredictionInput

app = FastAPI(title="Salary Prediction API")

# load encoder and trained model
model = joblib.load("models/salary_model.joblib")
encoders = joblib.load("models/encoders.joblib")

@app.get("/predict")
def predict_salary(params: PredictionInput = Depends()):
    """
    Takes job details via GET query parameters and returns a prediction.
    Requirement 3: Build a GET API for Predictions.
    """

    # 1. create a dictonary from the inputs
    input_dict = params.dict()

    
    # 2. TRANSFORM STRINGS TO NUMBERS (using the encoders we saved during cleaning)
    try:
        for col, le in encoders.items():
            input_dict[col] = le.transform([input_dict[col]])[0]
    except ValueError as e:
        # if the user sends a job title that wasnt in the original dataset (not limited to job titles)
        raise HTTPException(status_code=400, detail=f"Unknown value provided: {str(e)}")
    
    
    # 3. Make prediction (after everything is encoded)
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    
    return {
        "status": "success",
        "predicted_salary_usd": round(float(prediction), 2)
    }

@app.get("/")
def health_check():
    return {"message": "Salary API is up and running!"}