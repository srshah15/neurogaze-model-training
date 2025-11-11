"""
Detailed prediction with feature analysis
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

print("="*70)
print("DETAILED ASD CLASSIFICATION PREDICTION")
print("="*70)

# Load model
model_path = Path('model_results/random_forest_model.pkl')
if not model_path.exists():
    print("❌ Model not found. Please run predict_sample.py first to train the model.")
    exit(1)

print("\n1. Loading model...")
with open(model_path, 'rb') as f:
    model = pickle.load(f)
print("   ✓ Model loaded")

# Load training data for comparison
df_train = pd.read_csv('final_training_dataset.csv')
feature_cols = [col for col in df_train.columns if col not in ['Class']]
zero_var_features = df_train[feature_cols].columns[df_train[feature_cols].nunique() == 1]
feature_cols = [col for col in feature_cols if col not in zero_var_features]

# Load sample
print("\n2. Loading sample...")
df_sample = pd.read_csv('eyetrackingsample1.csv')
X_sample = df_sample[feature_cols]

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Make prediction
prediction = model.predict(X_sample)[0]
probabilities = model.predict_proba(X_sample)[0]
prob_asd = probabilities[1] * 100
prob_td = probabilities[0] * 100

print("\n" + "="*70)
print("PREDICTION RESULT")
print("="*70)
print(f"\nPredicted Class: {'ASD' if prediction == 1 else 'TD'}")
print(f"Probability of ASD: {prob_asd:.2f}%")
print(f"Probability of TD:  {prob_td:.2f}%")

if prob_asd > 70 or prob_td > 70:
    confidence = "High"
elif prob_asd > 60 or prob_td > 60:
    confidence = "Moderate"
else:
    confidence = "Low (Borderline case - model is uncertain)"

print(f"Confidence: {confidence}")

# Compare sample values with training data statistics
print("\n" + "="*70)
print("FEATURE ANALYSIS")
print("="*70)
print("\nTop 10 Most Important Features and Sample Values:")
print("-" * 70)

for i, row in feature_importance.head(10).iterrows():
    feat = row['feature']
    importance = row['importance']
    sample_val = X_sample[feat].iloc[0]
    
    # Get training statistics
    train_mean = df_train[feat].mean()
    train_std = df_train[feat].std()
    train_asd_mean = df_train[df_train['Class'] == 'ASD'][feat].mean()
    train_td_mean = df_train[df_train['Class'] == 'TD'][feat].mean()
    
    # Calculate z-score
    if train_std > 0:
        z_score = (sample_val - train_mean) / train_std
    else:
        z_score = 0
    
    # Determine if closer to ASD or TD
    dist_to_asd = abs(sample_val - train_asd_mean)
    dist_to_td = abs(sample_val - train_td_mean)
    
    if dist_to_asd < dist_to_td:
        closer_to = "ASD"
    else:
        closer_to = "TD"
    
    print(f"{feat:30s}")
    print(f"  Importance: {importance:.4f}")
    print(f"  Sample Value: {sample_val:.4f}")
    print(f"  Training Mean: {train_mean:.4f} (ASD: {train_asd_mean:.4f}, TD: {train_td_mean:.4f})")
    print(f"  Z-score: {z_score:.2f} ({'closer to ' + closer_to if abs(z_score) < 2 else 'outlier'})")
    print()

# Show demographic info if available
print("\n" + "="*70)
print("DEMOGRAPHIC INFORMATION")
print("="*70)
if 'Age' in df_sample.columns:
    age = df_sample['Age'].iloc[0]
    print(f"Age: {age:.1f} years")
    
    # Age distribution in training
    age_asd_mean = df_train[df_train['Class'] == 'ASD']['Age'].mean()
    age_td_mean = df_train[df_train['Class'] == 'TD']['Age'].mean()
    print(f"Training Age Mean - ASD: {age_asd_mean:.1f}, TD: {age_td_mean:.1f}")

if 'Gender_encoded' in df_sample.columns:
    gender = df_sample['Gender_encoded'].iloc[0]
    gender_str = "Male" if gender == 1 else "Female"
    print(f"Gender: {gender_str} (encoded: {gender})")
    
    # Gender distribution
    gender_asd_dist = df_train[df_train['Class'] == 'ASD']['Gender_encoded'].value_counts(normalize=True)
    gender_td_dist = df_train[df_train['Class'] == 'TD']['Gender_encoded'].value_counts(normalize=True)
    print(f"Training Gender Distribution:")
    print(f"  ASD - Male: {gender_asd_dist.get(1, 0)*100:.1f}%, Female: {gender_asd_dist.get(0, 0)*100:.1f}%")
    print(f"  TD - Male: {gender_td_dist.get(1, 0)*100:.1f}%, Female: {gender_td_dist.get(0, 0)*100:.1f}%")

# Summary
print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
if confidence == "Low (Borderline case - model is uncertain)":
    print("\n⚠️  This is a borderline case with nearly equal probabilities.")
    print("   The model is uncertain, which could indicate:")
    print("   - The sample has features that are mixed between ASD and TD patterns")
    print("   - The individual may be in a transitional or ambiguous state")
    print("   - Additional clinical assessment may be recommended")
    print("   - Consider collecting more data points for better prediction")
elif prediction == 1:
    print(f"\n✓ The model predicts ASD with {prob_asd:.1f}% confidence.")
    print("   However, this is a screening tool and should be used alongside")
    print("   professional clinical assessment.")
else:
    print(f"\n✓ The model predicts TD (Typical Development) with {prob_td:.1f}% confidence.")
    print("   However, this is a screening tool and should be used alongside")
    print("   professional clinical assessment.")

print("\n" + "="*70)

