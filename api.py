"""
FastAPI Backend for ASD Classification Model
Provides REST API endpoint for predictions using LightGBM model
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Optional
import os

# Initialize FastAPI app
app = FastAPI(
    title="ASD Classification API",
    description="API for predicting ASD (Autism Spectrum Disorder) vs TD (Typical Development) using eye-tracking features",
    version="1.0.0"
)

# CORS middleware - Allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",  # Vite default
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        # Add your production frontend URL here
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and feature structure on startup
MODEL_PATH = Path("model_results/lgbm_model.pkl")
TRAINING_DATA_PATH = Path("final_training_dataset.csv")

model = None
feature_columns = None
training_medians = {}

@app.on_event("startup")
async def load_model():
    """Load the trained model and feature structure on startup"""
    global model, feature_columns, training_medians
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    print("Loading model...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("✓ Model loaded")
    
    # Load training data to get feature structure
    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {TRAINING_DATA_PATH}")
    
    df_train = pd.read_csv(TRAINING_DATA_PATH)
    feature_cols = [col for col in df_train.columns if col != 'Class']
    
    # Remove zero variance features (same as training)
    zero_var_features = df_train[feature_cols].columns[df_train[feature_cols].nunique() == 1].tolist()
    feature_columns = [col for col in feature_cols if col not in zero_var_features]
    
    # Calculate medians for missing feature imputation
    for feat in feature_columns:
        training_medians[feat] = df_train[feat].median()
    
    print(f"✓ Loaded {len(feature_columns)} features")
    print("API ready!")

# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    """Request model for prediction - all 58 features required"""
    Tracking_F_1: float
    Tracking_F_2: float
    Tracking_F_3: float
    Tracking_F_4: float
    GazePoint_of_I_1: float
    GazePoint_of_I_2: float
    GazePoint_of_I_3: float
    GazePoint_of_I_4: float
    GazePoint_of_I_5: float
    GazePoint_of_I_6: float
    GazePoint_of_I_7: float
    GazePoint_of_I_8: float
    GazePoint_of_I_9: float
    GazePoint_of_I_10: float
    GazePoint_of_I_11: float
    GazePoint_of_I_12: float
    Recording_1: float
    Recording_2: float
    Recording_3: float
    gaze_hori_1: float
    gaze_hori_2: float
    gaze_hori_3: float
    gaze_hori_4: float
    gaze_vert_1: float
    gaze_vert_2: float
    gaze_vert_3: float
    gaze_vert_4: float
    gaze_velo_1: float
    gaze_velo_2: float
    gaze_velo_3: float
    gaze_velo_4: float
    blink_count_1: float
    blink_count_2: float
    blink_count_3: float
    blink_count_4: float
    fix_count_1: float
    fix_count_2: float
    fix_count_3: float
    fix_count_4: float
    sac_count_1: float
    sac_count_2: float
    sac_count_3: float
    sac_count_4: float
    trial_dur_1: float
    trial_dur_2: float
    sampling_rate_1: float
    blink_rate_1: float
    fixation_rate_1: float
    saccade_rate_1: float
    fix_dur_avg_1: float
    right_eye_c_1: float
    right_eye_c_2: float
    right_eye_c_3: float
    left_eye_c_1: float
    left_eye_c_2: float
    left_eye_c_3: float
    Age: float = Field(..., description="Age in years")
    Gender_encoded: int = Field(..., description="Gender: 0=Female, 1=Male")

    class Config:
        json_schema_extra = {
            "example": {
                "Tracking_F_1": 41.42,
                "Tracking_F_2": 7.11,
                "Tracking_F_3": 41.42,
                "Tracking_F_4": 41.42,
                "GazePoint_of_I_1": 427.30,
                "GazePoint_of_I_2": 351.02,
                "GazePoint_of_I_3": -103.01,
                "GazePoint_of_I_4": 1394.18,
                "GazePoint_of_I_5": 238.39,
                "GazePoint_of_I_6": 199.46,
                "GazePoint_of_I_7": -21.51,
                "GazePoint_of_I_8": 923.54,
                "GazePoint_of_I_9": 427.30,
                "GazePoint_of_I_10": 351.02,
                "GazePoint_of_I_11": -103.01,
                "GazePoint_of_I_12": 1394.18,
                "Recording_1": 1549.0,
                "Recording_2": 3105620.26,
                "Recording_3": 3157660.21,
                "gaze_hori_1": 427.30,
                "gaze_hori_2": 351.02,
                "gaze_hori_3": 427.30,
                "gaze_hori_4": 351.02,
                "gaze_vert_1": 238.39,
                "gaze_vert_2": 199.46,
                "gaze_vert_3": 238.39,
                "gaze_vert_4": 199.46,
                "gaze_velo_1": 102.12,
                "gaze_velo_2": 1391.21,
                "gaze_velo_3": 102.12,
                "gaze_velo_4": 1391.21,
                "blink_count_1": 229.0,
                "blink_count_2": 229.0,
                "blink_count_3": 229.0,
                "blink_count_4": 229.0,
                "fix_count_1": 332.0,
                "fix_count_2": 332.0,
                "fix_count_3": 332.0,
                "fix_count_4": 332.0,
                "sac_count_1": 183.0,
                "sac_count_2": 183.0,
                "sac_count_3": 183.0,
                "sac_count_4": 183.0,
                "trial_dur_1": 52039.94,
                "trial_dur_2": 52.04,
                "sampling_rate_1": 29.77,
                "blink_rate_1": 17.66,
                "fixation_rate_1": 25.56,
                "saccade_rate_1": 14.10,
                "fix_dur_avg_1": 0.0018,
                "right_eye_c_1": 427.30,
                "right_eye_c_2": 351.02,
                "right_eye_c_3": 1497.20,
                "left_eye_c_1": 427.30,
                "left_eye_c_2": 351.02,
                "left_eye_c_3": 1497.20,
                "Age": 9.4,
                "Gender_encoded": 1
            }
        }

class PredictionResponse(BaseModel):
    """Response model for prediction"""
    prediction: str = Field(..., description="Predicted class: 'ASD' or 'TD'")
    probability_asd: float = Field(..., description="Probability of ASD (0-1)")
    probability_td: float = Field(..., description="Probability of TD (0-1)")
    confidence: str = Field(..., description="Confidence level: 'High', 'Moderate', or 'Low'")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    features_count: int

class FeaturesResponse(BaseModel):
    """List of required features"""
    features: list[str]
    count: int

# API Endpoints

@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "ASD Classification API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "features": "/features",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features_count": len(feature_columns) if feature_columns else 0
    }

@app.get("/features", response_model=FeaturesResponse, tags=["General"])
async def get_features():
    """Get list of required features for prediction"""
    if feature_columns is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "features": feature_columns,
        "count": len(feature_columns)
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict ASD vs TD classification
    
    Takes eye-tracking features and returns prediction with probabilities
    """
    if model is None or feature_columns is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please wait for startup to complete.")
    
    try:
        # Convert request to DataFrame
        input_data = request.dict()
        
        # Create DataFrame with single row
        df_input = pd.DataFrame([input_data])
        
        # Ensure correct feature order and handle missing features
        X_input = pd.DataFrame(index=[0])
        for feat in feature_columns:
            if feat in df_input.columns:
                X_input[feat] = df_input[feat].iloc[0]
            else:
                # Fill missing features with training median
                X_input[feat] = training_medians.get(feat, 0.0)
        
        # Reorder to match training feature order
        X_input = X_input[feature_columns]
        
        # Make prediction
        prediction_proba = model.predict_proba(X_input)[0]
        prediction = model.predict(X_input)[0]
        
        # Convert to class labels
        pred_class = "ASD" if prediction == 1 else "TD"
        prob_asd = float(prediction_proba[1])
        prob_td = float(prediction_proba[0])
        
        # Determine confidence
        if prob_asd > 0.75 or prob_td > 0.75:
            confidence = "High"
        elif prob_asd > 0.65 or prob_td > 0.65:
            confidence = "Moderate"
        else:
            confidence = "Low"
        
        return {
            "prediction": pred_class,
            "probability_asd": prob_asd,
            "probability_td": prob_td,
            "confidence": confidence
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

