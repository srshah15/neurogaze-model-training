# Quick Start Guide

## Installation

The core packages (pandas, numpy, lightgbm) are already installed. Just install the API dependencies:

```bash
pip install fastapi uvicorn pydantic python-multipart
```

Or install everything from requirements.txt:
```bash
pip install -r requirements.txt
```

## Run the API

```bash
uvicorn api:app --reload --port 8000
```

The API will be available at:
- **API**: `http://localhost:8000`
- **Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Test the API

### Using curl:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Tracking_F_1": 41.42,
    "Tracking_F_2": 7.11,
    ...
    "Age": 9.4,
    "Gender_encoded": 1
  }'
```

### Health Check:
```bash
curl http://localhost:8000/health
```

## Frontend Integration

See `FRONTEND_INTEGRATION.md` for TypeScript examples and full integration guide.

