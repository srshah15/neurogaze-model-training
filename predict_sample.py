"""
Predict ASD/TD classification on a new sample
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("="*70)
print("ASD CLASSIFICATION PREDICTION")
print("="*70)

# Load training data to get feature structure
print("\n1. Loading training data...")
df_train = pd.read_csv('final_training_dataset.csv')
print(f"   Training dataset shape: {df_train.shape}")

# Prepare features and target
feature_cols = [col for col in df_train.columns if col not in ['Class']]
X_train_full = df_train[feature_cols]
y_train_full = df_train['Class'].map({'ASD': 1, 'TD': 0})

# Remove zero variance features (same as training)
zero_var_features = X_train_full.columns[X_train_full.nunique() == 1]
if len(zero_var_features) > 0:
    print(f"   Removing zero variance features: {list(zero_var_features)}")
    X_train_full = X_train_full.drop(columns=zero_var_features)
    feature_cols = [col for col in feature_cols if col not in zero_var_features]

print(f"   Final feature count: {len(feature_cols)}")

# Check if model is saved
model_path = Path('model_results/random_forest_model.pkl')
if model_path.exists():
    print("\n2. Loading saved model...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("   ✓ Model loaded from saved file")
else:
    print("\n2. Training model (this may take a minute)...")
    # Train model with same parameters as training script
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Use full dataset for training
    model.fit(X_train_full, y_train_full)
    print("   ✓ Model trained")
    
    # Save model
    model_path.parent.mkdir(exist_ok=True, parents=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"   ✓ Model saved to {model_path}")

# Load sample data
print("\n3. Loading sample data...")
try:
    df_sample = pd.read_csv('eyetrackingsample1.csv')
    print(f"   Sample shape: {df_sample.shape}")
    print(f"   Sample columns: {len(df_sample.columns)}")
except FileNotFoundError:
    print("   ❌ Error: eyetrackingsample1.csv not found!")
    exit(1)

# Check if sample has required features
missing_features = [col for col in feature_cols if col not in df_sample.columns]
if missing_features:
    print(f"\n   ⚠️  Warning: Missing {len(missing_features)} features:")
    for feat in missing_features[:10]:
        print(f"      - {feat}")
    if len(missing_features) > 10:
        print(f"      ... and {len(missing_features) - 10} more")
    
    # Use available features only
    available_features = [col for col in feature_cols if col in df_sample.columns]
    print(f"\n   Using {len(available_features)} available features")
    X_sample = df_sample[available_features]
    
    # Fill missing features with training data median
    for feat in missing_features:
        median_val = X_train_full[feat].median()
        X_sample[feat] = median_val
        print(f"   Filled {feat} with median: {median_val:.4f}")
    
    # Reorder to match training
    X_sample = X_sample[feature_cols]
else:
    X_sample = df_sample[feature_cols]
    print("   ✓ All required features present")

# Make predictions
print("\n4. Making predictions...")
predictions = model.predict(X_sample)
probabilities = model.predict_proba(X_sample)

# Display results
print("\n" + "="*70)
print("PREDICTION RESULTS")
print("="*70)

for i in range(len(X_sample)):
    pred_class = "ASD" if predictions[i] == 1 else "TD"
    prob_asd = probabilities[i][1] * 100
    prob_td = probabilities[i][0] * 100
    
    print(f"\nSample {i+1}:")
    print(f"  Predicted Class: {pred_class}")
    print(f"  Probability of ASD: {prob_asd:.2f}%")
    print(f"  Probability of TD:  {prob_td:.2f}%")
    
    # Confidence level
    if prob_asd > 70 or prob_td > 70:
        confidence = "High"
    elif prob_asd > 60 or prob_td > 60:
        confidence = "Moderate"
    else:
        confidence = "Low"
    
    print(f"  Confidence: {confidence}")
    
    # Visual bar
    bar_length = 50
    asd_bars = int(bar_length * prob_asd / 100)
    td_bars = bar_length - asd_bars
    
    print(f"  [{'█' * asd_bars}{'░' * td_bars}]")
    print(f"   ASD{' ' * (asd_bars-3) if asd_bars > 3 else ''}TD")

# Save results
print("\n" + "="*70)
results_df = pd.DataFrame({
    'Sample': range(1, len(X_sample) + 1),
    'Predicted_Class': ['ASD' if p == 1 else 'TD' for p in predictions],
    'Probability_ASD': [prob[1] * 100 for prob in probabilities],
    'Probability_TD': [prob[0] * 100 for prob in probabilities]
})

output_file = Path('model_results/prediction_results.csv')
results_df.to_csv(output_file, index=False)
print(f"✓ Results saved to: {output_file}")

print("\n" + "="*70)
print("PREDICTION COMPLETE!")
print("="*70)

