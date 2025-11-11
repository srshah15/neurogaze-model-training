# ASD Classification API

FastAPI backend for predicting ASD (Autism Spectrum Disorder) vs TD (Typical Development) using eye-tracking features.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Ensure Model File Exists

Make sure `model_results/lgbm_model.pkl` exists. If not, train the model first:

```bash
python3 train_models.py
```

### 3. Run the API Server

```bash
uvicorn api:app --reload --port 8000
```

Or:

```bash
python3 api.py
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Endpoints

### `GET /`
Root endpoint with API information

### `GET /health`
Health check - returns API status and model loading status

### `GET /features`
Returns list of all 58 required features for prediction

### `POST /predict`
Main prediction endpoint

**Request Body:**
```json
{
  "Tracking_F_1": 41.42,
  "Tracking_F_2": 7.11,
  // ... all 58 features
  "Age": 9.4,
  "Gender_encoded": 1
}
```

**Response:**
```json
{
  "prediction": "ASD",
  "probability_asd": 0.5019,
  "probability_td": 0.4981,
  "confidence": "Low"
}
```

## Frontend Integration

### TypeScript Example

```typescript
interface PredictionRequest {
  Tracking_F_1: number;
  Tracking_F_2: number;
  // ... all 58 features
  Age: number;
  Gender_encoded: number;
}

interface PredictionResponse {
  prediction: "ASD" | "TD";
  probability_asd: number;
  probability_td: number;
  confidence: "High" | "Moderate" | "Low";
}

async function predictASD(data: PredictionRequest): Promise<PredictionResponse> {
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data)
  });
  
  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }
  
  return await response.json();
}
```

## CORS

The API is configured to allow requests from:
- `http://localhost:3000` (React default)
- `http://localhost:3001`
- `http://localhost:5173` (Vite default)
- `http://localhost:8080`

To add your production frontend URL, edit the `allow_origins` list in `api.py`.

## Production Deployment

For production, use a production ASGI server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

Or use Gunicorn with Uvicorn workers:

```bash
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Notes

- Model is loaded once on startup for fast predictions
- Missing features are automatically filled with training data medians
- All 58 features are required (excluding 2 zero-variance features)
- Model trained on pediatric data (ages 2.7-12.9 years)

