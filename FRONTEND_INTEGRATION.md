# Frontend Integration Guide

## API Endpoint

```
POST http://localhost:8000/predict
```

## Request Format

Send a JSON object with all 58 features:

```typescript
{
  "Tracking_F_1": number,
  "Tracking_F_2": number,
  "Tracking_F_3": number,
  "Tracking_F_4": number,
  "GazePoint_of_I_1": number,
  "GazePoint_of_I_2": number,
  "GazePoint_of_I_3": number,
  "GazePoint_of_I_4": number,
  "GazePoint_of_I_5": number,
  "GazePoint_of_I_6": number,
  "GazePoint_of_I_7": number,
  "GazePoint_of_I_8": number,
  "GazePoint_of_I_9": number,
  "GazePoint_of_I_10": number,
  "GazePoint_of_I_11": number,
  "GazePoint_of_I_12": number,
  "Recording_1": number,
  "Recording_2": number,
  "Recording_3": number,
  "gaze_hori_1": number,
  "gaze_hori_2": number,
  "gaze_hori_3": number,
  "gaze_hori_4": number,
  "gaze_vert_1": number,
  "gaze_vert_2": number,
  "gaze_vert_3": number,
  "gaze_vert_4": number,
  "gaze_velo_1": number,
  "gaze_velo_2": number,
  "gaze_velo_3": number,
  "gaze_velo_4": number,
  "blink_count_1": number,
  "blink_count_2": number,
  "blink_count_3": number,
  "blink_count_4": number,
  "fix_count_1": number,
  "fix_count_2": number,
  "fix_count_3": number,
  "fix_count_4": number,
  "sac_count_1": number,
  "sac_count_2": number,
  "sac_count_3": number,
  "sac_count_4": number,
  "trial_dur_1": number,
  "trial_dur_2": number,
  "sampling_rate_1": number,
  "blink_rate_1": number,
  "fixation_rate_1": number,
  "saccade_rate_1": number,
  "fix_dur_avg_1": number,
  "right_eye_c_1": number,
  "right_eye_c_2": number,
  "right_eye_c_3": number,
  "left_eye_c_1": number,
  "left_eye_c_2": number,
  "left_eye_c_3": number,
  "Age": number,
  "Gender_encoded": number  // 0 = Female, 1 = Male
}
```

## Response Format

```typescript
{
  "prediction": "ASD" | "TD",
  "probability_asd": number,  // 0.0 to 1.0
  "probability_td": number,   // 0.0 to 1.0
  "confidence": "High" | "Moderate" | "Low"
}
```

## TypeScript Interfaces

```typescript
interface PredictionRequest {
  Tracking_F_1: number;
  Tracking_F_2: number;
  Tracking_F_3: number;
  Tracking_F_4: number;
  GazePoint_of_I_1: number;
  GazePoint_of_I_2: number;
  GazePoint_of_I_3: number;
  GazePoint_of_I_4: number;
  GazePoint_of_I_5: number;
  GazePoint_of_I_6: number;
  GazePoint_of_I_7: number;
  GazePoint_of_I_8: number;
  GazePoint_of_I_9: number;
  GazePoint_of_I_10: number;
  GazePoint_of_I_11: number;
  GazePoint_of_I_12: number;
  Recording_1: number;
  Recording_2: number;
  Recording_3: number;
  gaze_hori_1: number;
  gaze_hori_2: number;
  gaze_hori_3: number;
  gaze_hori_4: number;
  gaze_vert_1: number;
  gaze_vert_2: number;
  gaze_vert_3: number;
  gaze_vert_4: number;
  gaze_velo_1: number;
  gaze_velo_2: number;
  gaze_velo_3: number;
  gaze_velo_4: number;
  blink_count_1: number;
  blink_count_2: number;
  blink_count_3: number;
  blink_count_4: number;
  fix_count_1: number;
  fix_count_2: number;
  fix_count_3: number;
  fix_count_4: number;
  sac_count_1: number;
  sac_count_2: number;
  sac_count_3: number;
  sac_count_4: number;
  trial_dur_1: number;
  trial_dur_2: number;
  sampling_rate_1: number;
  blink_rate_1: number;
  fixation_rate_1: number;
  saccade_rate_1: number;
  fix_dur_avg_1: number;
  right_eye_c_1: number;
  right_eye_c_2: number;
  right_eye_c_3: number;
  left_eye_c_1: number;
  left_eye_c_2: number;
  left_eye_c_3: number;
  Age: number;
  Gender_encoded: number;
}

interface PredictionResponse {
  prediction: "ASD" | "TD";
  probability_asd: number;
  probability_td: number;
  confidence: "High" | "Moderate" | "Low";
}
```

## Example Usage

### Fetch API

```typescript
async function predictASD(featureData: PredictionRequest): Promise<PredictionResponse> {
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(featureData)
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `API error: ${response.statusText}`);
  }
  
  return await response.json();
}

// Usage
const result = await predictASD({
  Tracking_F_1: 41.42,
  Tracking_F_2: 7.11,
  // ... all features
  Age: 9.4,
  Gender_encoded: 1
});

console.log(`Prediction: ${result.prediction}`);
console.log(`ASD Probability: ${(result.probability_asd * 100).toFixed(2)}%`);
```

### Axios

```typescript
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
});

async function predictASD(featureData: PredictionRequest): Promise<PredictionResponse> {
  const response = await api.post('/predict', featureData);
  return response.data;
}
```

## Error Handling

The API returns standard HTTP status codes:

- `200` - Success
- `400` - Bad Request (validation error)
- `500` - Server Error (prediction error)
- `503` - Service Unavailable (model not loaded)

Error response format:
```typescript
{
  "detail": "Error message here"
}
```

## Other Useful Endpoints

### Health Check
```typescript
GET http://localhost:8000/health

// Response
{
  "status": "healthy",
  "model_loaded": true,
  "features_count": 58
}
```

### Get Required Features
```typescript
GET http://localhost:8000/features

// Response
{
  "features": ["Tracking_F_1", "Tracking_F_2", ...],
  "count": 58
}
```

## CORS

The API allows requests from:
- `http://localhost:3000` (React default)
- `http://localhost:3001`
- `http://localhost:5173` (Vite default)
- `http://localhost:8080`

For production, update the `allow_origins` list in `api.py`.

## Production URL

When deploying, replace `http://localhost:8000` with your production API URL:

```typescript
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
```

