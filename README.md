# Neurogaze ASD Classification Model

## Files Overview

### Datasets
- **`final_training_dataset.csv`** - Main training dataset (78,209 samples, 61 features)
- **`eyetrackingsample1.csv`** - Sample data for prediction

### Model Training
- **`train_models.py`** - Train Random Forest, XGBoost, and LightGBM models
  - Run: `python3 train_models.py`
  - Saves trained models to `model_results/`

### Prediction
- **`predict_sample.py`** - Quick prediction on new samples
  - Run: `python3 predict_sample.py`
  - Outputs: Class prediction and probabilities
  
- **`predict_sample_detailed.py`** - Detailed prediction with feature analysis
  - Run: `python3 predict_sample_detailed.py`
  - Shows which features influenced the prediction

### Model Results
- **`model_results/`** - Directory containing:
  - `random_forest_model.pkl` - Trained Random Forest model
  - `prediction_results.csv` - Prediction results
  - `model_comparison.csv` - Model performance comparison
  - Visualization files (PNG images)

## Quick Start

1. **Train models:**
   ```bash
   python3 train_models.py
   ```

2. **Make predictions:**
   ```bash
   python3 predict_sample.py
   ```

## Model Performance

- **Random Forest:** 97.53% accuracy, 0.9981 ROC-AUC
- Trained on 78,209 samples
- Binary classification: ASD vs TD (Typical Development)

## Notes

- Models are trained on pediatric data (ages 2.7-12.9 years)
- Predictions for individuals outside this age range may be less reliable
- This is a screening tool - use alongside professional clinical assessment

