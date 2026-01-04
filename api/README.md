# API Directory

REST API for serving dementia prediction models.

## Overview

The API provides HTTP endpoints for making predictions using trained models, enabling:
- Remote access to models
- Web application integration
- External system integration
- Real-time predictions

## Quick Start

### Start the API Server

```bash
python api/app.py
```

Server will be available at:
- **Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **API Documentation**: http://localhost:8000/redoc

## Prerequisites

### Install API Dependencies

```bash
pip install fastapi uvicorn pydantic
```

Or use the conda environment:
```bash
conda env create -f environment.yml
conda activate ad-ensemble
```

### Train Models First

The API requires trained models:

```bash
# Option 1: Use main.py
python main.py --mode tabular

# Option 2: Use training scripts
python scripts/train_tabular.py
python scripts/train_ensemble.py
```

## API Endpoints

### Health Check

```bash
GET /health
```

Returns server status and loaded models.

### List Models

```bash
GET /models
```

Returns list of available models.

### Make Prediction

```bash
POST /predict/{model_name}
```

Make prediction using a specific model.

**Example**:
```bash
curl -X POST "http://localhost:8000/predict/stacking_ensemble" \
     -H "Content-Type: application/json" \
     -d '{
       "Age": 75,
       "EDUC": 12,
       "MMSE": 28,
       "eTIV": 1500,
       "nWBV": 0.7,
       "ASF": 1.2,
       "gender": "F"
     }'
```

**Response**:
```json
{
  "prediction": 1,
  "probability": 0.78,
  "model_used": "stacking_ensemble",
  "confidence": "high"
}
```

### Ensemble Prediction

```bash
POST /predict/ensemble
```

Automatically uses the best available ensemble model.

### All Models Prediction

```bash
POST /predict/all
```

Get predictions from all available models with consensus.

## Input Parameters

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| Age | float | 0-120 | Patient age in years |
| EDUC | float | 0-30 | Years of education |
| MMSE | float | 0-30 | Mini-Mental State Examination score |
| eTIV | float | 500-2500 | Estimated total intracranial volume |
| nWBV | float | 0-1 | Normalized whole brain volume |
| ASF | float | 0.5-2 | Atlas scaling factor |
| gender | string | M/F | Gender |

## Response Fields

| Field | Type | Description |
|-------|------|-------------|
| prediction | int | 0 = non-demented, 1 = demented |
| probability | float | Probability of dementia (0-1) |
| model_used | string | Name of model used |
| confidence | string | high/medium/low |

## Using the Interactive Docs

1. Start the server: `python api/app.py`
2. Open browser: http://localhost:8000/docs
3. Try the endpoints using the "Try it out" button
4. See example requests and responses

## Python Client Example

```python
import requests

# Server URL
url = "http://localhost:8000/predict/stacking_ensemble"

# Patient data
patient_data = {
    "Age": 75,
    "EDUC": 12,
    "MMSE": 28,
    "eTIV": 1500,
    "nWBV": 0.7,
    "ASF": 1.2,
    "gender": "F"
}

# Make request
response = requests.post(url, json=patient_data)

# Get prediction
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.2%}")
print(f"Confidence: {result['confidence']}")
```

## JavaScript Client Example

```javascript
const patientData = {
  Age: 75,
  EDUC: 12,
  MMSE: 28,
  eTIV: 1500,
  nWBV: 0.7,
  ASF: 1.2,
  gender: "F"
};

fetch('http://localhost:8000/predict/stacking_ensemble', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(patientData)
})
.then(response => response.json())
.then(data => {
  console.log('Prediction:', data.prediction);
  console.log('Probability:', data.probability);
  console.log('Confidence:', data.confidence);
});
```

## Deployment

### Development
```bash
python api/app.py
```

### Production (with uvicorn)
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker
```bash
# Build image
docker build -t dementia-ml-api .

# Run container
docker run -p 8000:8000 dementia-ml-api
```

## Security Considerations

**Important**: This API is for research/demonstration purposes.

For production deployment:
- [ ] Add authentication (API keys, OAuth)
- [ ] Enable HTTPS/TLS
- [ ] Add rate limiting
- [ ] Implement request validation
- [ ] Add logging and monitoring
- [ ] Follow HIPAA/GDPR compliance if handling real patient data

## Troubleshooting

### "No models available"
- Train models first: `python main.py --mode tabular`
- Check `models/` directory exists and contains `.pkl` files

### "FastAPI not installed"
- Install: `pip install fastapi uvicorn pydantic`
- Or use conda environment

### "Port already in use"
- Change port: `uvicorn api.app:app --port 8001`
- Or stop the process using port 8000

## API Documentation

- **OpenAPI spec**: http://localhost:8000/openapi.json
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Further Reading

- FastAPI Documentation: https://fastapi.tiangolo.com/
- Main Project README: [../README.md](../README.md)
- Reproducibility Guide: [../REPRODUCIBILITY.md](../REPRODUCIBILITY.md)

---

**Status**: Optional feature for web integration  
**Dependencies**: fastapi, uvicorn, pydantic (installed via environment.yml)
