"""
REST API for Dementia Prediction Models

This FastAPI application provides endpoints for model predictions,
allowing external access to trained models.

Usage:
    python api/app.py
    
Access:
    - Server: http://localhost:8000
    - API Docs: http://localhost:8000/docs
    - Health: http://localhost:8000/health
"""

import os
import sys
import pickle
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    import pandas as pd
    import numpy as np
except ImportError:
    print("FastAPI not installed. Install with: pip install fastapi uvicorn pydantic")
    print("This is optional for API functionality.")
    sys.exit(1)


# Initialize FastAPI app
app = FastAPI(
    title="Dementia Prediction API",
    description="REST API for early dementia detection using ML models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# Request/Response models
class PatientData(BaseModel):
    """Patient clinical data for prediction."""
    Age: float = Field(..., ge=0, le=120, description="Patient age in years")
    EDUC: float = Field(..., ge=0, le=30, description="Years of education")
    MMSE: float = Field(..., ge=0, le=30, description="Mini-Mental State Examination score")
    eTIV: float = Field(..., ge=500, le=2500, description="Estimated total intracranial volume")
    nWBV: float = Field(..., ge=0, le=1, description="Normalized whole brain volume")
    ASF: float = Field(..., ge=0.5, le=2, description="Atlas scaling factor")
    gender: str = Field(..., pattern="^(M|F)$", description="Gender (M or F)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "Age": 75,
                "EDUC": 12,
                "MMSE": 28,
                "eTIV": 1500,
                "nWBV": 0.7,
                "ASF": 1.2,
                "gender": "F"
            }
        }


class PredictionResponse(BaseModel):
    """Prediction result."""
    prediction: int = Field(..., description="Predicted class (0=non-demented, 1=demented)")
    probability: float = Field(..., ge=0, le=1, description="Probability of dementia")
    model_used: str = Field(..., description="Model name")
    confidence: str = Field(..., description="Confidence level")


# Global model storage
models = {}
preprocessor = None


def load_models():
    """Load trained models at startup."""
    global models, preprocessor
    
    models_dir = Path(__file__).parent.parent / 'models'
    
    try:
        # Load preprocessor
        preprocessor_path = models_dir / 'preprocessor.pkl'
        if preprocessor_path.exists():
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            print("✓ Loaded preprocessor")
        
        # Load models
        model_files = {
            'random_forest': 'random_forest.pkl',
            'gradient_boosting': 'gradient_boosting.pkl',
            'stacking_ensemble': 'stacking_ensemble.pkl',
            'voting_ensemble': 'voting_ensemble.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = models_dir / filename
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    models[model_name] = pickle.load(f)
                print(f"✓ Loaded {model_name}")
        
        if not models:
            print("⚠ No models loaded. Train models first with: python main.py")
        else:
            print(f"\n✓ Loaded {len(models)} models")
        
    except Exception as e:
        print(f"✗ Error loading models: {e}")


def prepare_input(patient_data: PatientData) -> pd.DataFrame:
    """Prepare patient data for prediction."""
    data = {
        'Age': [patient_data.Age],
        'EDUC': [patient_data.EDUC],
        'MMSE': [patient_data.MMSE],
        'eTIV': [patient_data.eTIV],
        'nWBV': [patient_data.nWBV],
        'ASF': [patient_data.ASF],
        'M/F': [patient_data.gender]
    }
    
    df = pd.DataFrame(data)
    
    # Preprocess if preprocessor is available
    if preprocessor:
        X_processed = preprocessor.transform(df)
        feature_names = preprocessor.get_feature_names_out()
        return pd.DataFrame(X_processed, columns=feature_names)
    else:
        return df


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    print("\n" + "="*60)
    print("Dementia Prediction API - Starting Up")
    print("="*60 + "\n")
    load_models()
    print("\n" + "="*60)
    print("API Ready!")
    print("="*60 + "\n")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Dementia Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "models": "/models",
            "predict": "/predict/{model_name}"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "preprocessor_loaded": preprocessor is not None
    }


@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "available_models": list(models.keys()),
        "count": len(models),
        "recommended": "stacking_ensemble" if "stacking_ensemble" in models else None
    }


@app.post("/predict/{model_name}", response_model=PredictionResponse)
async def predict(model_name: str, patient_data: PatientData):
    """
    Make prediction using specified model.
    
    Args:
        model_name: Name of the model to use
        patient_data: Patient clinical data
    
    Returns:
        Prediction result with probability
    """
    # Check if model exists
    if model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {list(models.keys())}"
        )
    
    try:
        # Prepare input
        X = prepare_input(patient_data)
        
        # Get model
        model = models[model_name]
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]  # Probability of class 1 (demented)
        
        # Determine confidence
        if probability < 0.3 or probability > 0.7:
            confidence = "high"
        elif probability < 0.4 or probability > 0.6:
            confidence = "medium"
        else:
            confidence = "low"
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            model_used=model_name,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/ensemble", response_model=PredictionResponse)
async def predict_ensemble(patient_data: PatientData):
    """
    Make prediction using best available ensemble model.
    
    Automatically selects stacking_ensemble if available, otherwise voting_ensemble.
    """
    # Select best ensemble
    if "stacking_ensemble" in models:
        model_name = "stacking_ensemble"
    elif "voting_ensemble" in models:
        model_name = "voting_ensemble"
    else:
        raise HTTPException(
            status_code=503,
            detail="No ensemble models available"
        )
    
    return await predict(model_name, patient_data)


@app.post("/predict/all")
async def predict_all_models(patient_data: PatientData):
    """
    Get predictions from all available models.
    
    Useful for comparison and consensus.
    """
    if not models:
        raise HTTPException(
            status_code=503,
            detail="No models available"
        )
    
    results = {}
    
    for model_name in models.keys():
        try:
            result = await predict(model_name, patient_data)
            results[model_name] = result.dict()
        except Exception as e:
            results[model_name] = {"error": str(e)}
    
    # Calculate consensus
    predictions = [r['prediction'] for r in results.values() if 'prediction' in r]
    if predictions:
        consensus = 1 if sum(predictions) > len(predictions)/2 else 0
        results['consensus'] = {
            'prediction': consensus,
            'agreement': sum(p == consensus for p in predictions) / len(predictions)
        }
    
    return results


def main():
    """Run the API server."""
    print("\n" + "="*60)
    print("Starting Dementia Prediction API Server")
    print("="*60 + "\n")
    print("Server will be available at:")
    print("  - http://localhost:8000")
    print("  - API Docs: http://localhost:8000/docs")
    print("  - Health Check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server\n")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
