"""
Quick Start Model Training Script
Implements the top 3 recommended models for ASD classification
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Try importing models (install if needed)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    print("⚠️  XGBoost not available.")
    if "libomp" in str(e) or "OpenMP" in str(e):
        print("   XGBoost requires OpenMP runtime on macOS.")
        print("   Fix: Run 'brew install libomp' then reinstall xgboost: 'pip install --upgrade --force-reinstall xgboost'")
    else:
        print("   Install with: pip install xgboost")
    print(f"   Error: {str(e)[:100]}")
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except (ImportError, Exception) as e:
    print("⚠️  LightGBM not available.")
    if "libomp" in str(e) or "OpenMP" in str(e):
        print("   LightGBM requires OpenMP runtime on macOS.")
        print("   Fix: Run 'brew install libomp' then reinstall lightgbm: 'pip install --upgrade --force-reinstall lightgbm'")
    else:
        print("   Install with: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    RF_AVAILABLE = True
except ImportError:
    RF_AVAILABLE = False

# Check if at least one model is available
if not (XGBOOST_AVAILABLE or LIGHTGBM_AVAILABLE or RF_AVAILABLE):
    print("\n❌ ERROR: No models available!")
    print("Please install at least one of:")
    print("  - Random Forest (included with scikit-learn)")
    print("  - XGBoost: pip install xgboost")
    print("  - LightGBM: pip install lightgbm")
    print("\nFor XGBoost on macOS, you may need:")
    print("  1. brew install libomp")
    print("  2. pip install --upgrade --force-reinstall xgboost")
    exit(1)

# Load data
print("="*70)
print("LOADING DATA")
print("="*70)
df = pd.read_csv('final_training_dataset.csv')
print(f"Dataset shape: {df.shape}")

# Prepare features and target
feature_cols = [col for col in df.columns if col not in ['Class']]
X = df[feature_cols]
y = df['Class'].map({'ASD': 1, 'TD': 0})

# Remove zero variance features
zero_var_features = X.columns[X.nunique() == 1]
if len(zero_var_features) > 0:
    print(f"\nRemoving {len(zero_var_features)} zero variance features: {list(zero_var_features)}")
    X = X.drop(columns=zero_var_features)
    feature_cols = [col for col in feature_cols if col not in zero_var_features]

print(f"Features after cleaning: {len(feature_cols)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain set: {X_train.shape[0]:,} samples")
print(f"Test set: {X_test.shape[0]:,} samples")

# Create output directory
output_dir = Path('model_results')
output_dir.mkdir(exist_ok=True)

# Store results
results = {}
saved_models = {}  # Store models for ensemble

# ============================================================================
# MODEL 1: XGBoost
# ============================================================================
if XGBOOST_AVAILABLE:
    print("\n" + "="*70)
    print("TRAINING XGBOOST")
    print("="*70)
    
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    xgb_model.fit(X_train, y_train)
    
    # Predictions
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
    
    print(f"\nXGBoost Results:")
    print(f"  Accuracy: {acc_xgb:.4f} ({acc_xgb*100:.2f}%)")
    print(f"  ROC-AUC: {auc_xgb:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_xgb, target_names=['TD', 'ASD']))
    
    results['XGBoost'] = {
        'model': xgb_model,
        'accuracy': acc_xgb,
        'auc': auc_xgb,
        'predictions': y_pred_xgb,
        'probabilities': y_pred_proba_xgb
    }
    saved_models['xgb'] = xgb_model
    
    # Save XGBoost model
    with open(output_dir / 'xgb_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    print(f"   ✓ Model saved to {output_dir / 'xgb_model.pkl'}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.4f}")
    
    # Save feature importance
    feature_importance.to_csv(output_dir / 'xgb_feature_importance.csv', index=False)

# ============================================================================
# MODEL 2: LightGBM
# ============================================================================
if LIGHTGBM_AVAILABLE:
    print("\n" + "="*70)
    print("TRAINING LIGHTGBM")
    print("="*70)
    
    lgbm_model = LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    
    lgbm_model.fit(X_train, y_train)
    
    # Predictions
    y_pred_lgbm = lgbm_model.predict(X_test)
    y_pred_proba_lgbm = lgbm_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc_lgbm = accuracy_score(y_test, y_pred_lgbm)
    auc_lgbm = roc_auc_score(y_test, y_pred_proba_lgbm)
    
    print(f"\nLightGBM Results:")
    print(f"  Accuracy: {acc_lgbm:.4f} ({acc_lgbm*100:.2f}%)")
    print(f"  ROC-AUC: {auc_lgbm:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_lgbm, target_names=['TD', 'ASD']))
    
    results['LightGBM'] = {
        'model': lgbm_model,
        'accuracy': acc_lgbm,
        'auc': auc_lgbm,
        'predictions': y_pred_lgbm,
        'probabilities': y_pred_proba_lgbm
    }
    saved_models['lgbm'] = lgbm_model
    
    # Save LightGBM model
    with open(output_dir / 'lgbm_model.pkl', 'wb') as f:
        pickle.dump(lgbm_model, f)
    print(f"   ✓ Model saved to {output_dir / 'lgbm_model.pkl'}")

# ============================================================================
# MODEL 3: Random Forest
# ============================================================================
if RF_AVAILABLE:
    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST")
    print("="*70)
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc_rf = accuracy_score(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
    
    print(f"\nRandom Forest Results:")
    print(f"  Accuracy: {acc_rf:.4f} ({acc_rf*100:.2f}%)")
    print(f"  ROC-AUC: {auc_rf:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_rf, target_names=['TD', 'ASD']))
    
    results['Random Forest'] = {
        'model': rf_model,
        'accuracy': acc_rf,
        'auc': auc_rf,
        'predictions': y_pred_rf,
        'probabilities': y_pred_proba_rf
    }
    saved_models['rf'] = rf_model
    
    # Save Random Forest model
    with open(output_dir / 'random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"   ✓ Model saved to {output_dir / 'random_forest_model.pkl'}")

# ============================================================================
# COMPARISON & VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)

if len(results) > 0:
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [r['accuracy'] for r in results.values()],
        'ROC-AUC': [r['auc'] for r in results.values()]
    }).sort_values('Accuracy', ascending=False)
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    axes[0].barh(comparison_df['Model'], comparison_df['Accuracy'], color=['#3498db', '#e74c3c', '#2ecc71'][:len(results)])
    axes[0].set_xlabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    for i, v in enumerate(comparison_df['Accuracy']):
        axes[0].text(v + 0.001, i, f'{v:.4f}', va='center', fontweight='bold')
    
    # ROC-AUC comparison
    axes[1].barh(comparison_df['Model'], comparison_df['ROC-AUC'], color=['#3498db', '#e74c3c', '#2ecc71'][:len(results)])
    axes[1].set_xlabel('ROC-AUC', fontsize=12)
    axes[1].set_title('Model ROC-AUC Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    for i, v in enumerate(comparison_df['ROC-AUC']):
        axes[1].text(v + 0.001, i, f'{v:.4f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Comparison plot saved to: {output_dir / 'model_comparison.png'}")
    
    # ROC Curves
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, (name, result) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {result['auc']:.4f})", 
                linewidth=2, color=colors[i % len(colors)])
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC curves saved to: {output_dir / 'roc_curves.png'}")
    
    # Confusion matrices
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for i, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(y_test, result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=['TD', 'ASD'], yticklabels=['TD', 'ASD'])
        axes[i].set_title(f'{name}\nAccuracy: {result["accuracy"]:.4f}', 
                         fontsize=12, fontweight='bold')
        axes[i].set_ylabel('True Label', fontsize=10)
        axes[i].set_xlabel('Predicted Label', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrices saved to: {output_dir / 'confusion_matrices.png'}")

# ============================================================================
# ENSEMBLE MODEL (Voting Classifier)
# ============================================================================
if len(saved_models) >= 2:
    print("\n" + "="*70)
    print("CREATING ENSEMBLE MODEL")
    print("="*70)
    
    from sklearn.ensemble import VotingClassifier
    
    # Create list of (name, model) tuples for voting
    estimators = []
    if 'xgb' in saved_models:
        estimators.append(('xgb', saved_models['xgb']))
    if 'lgbm' in saved_models:
        estimators.append(('lgbm', saved_models['lgbm']))
    if 'rf' in saved_models:
        estimators.append(('rf', saved_models['rf']))
    
    if len(estimators) >= 2:
        # Use soft voting (averages probabilities) for better performance
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probability voting
            n_jobs=-1
        )
        
        print(f"\nTraining ensemble with {len(estimators)} models:")
        for name, _ in estimators:
            print(f"  - {name.upper()}")
        
        # Fit ensemble (models are already trained, but VotingClassifier needs refit)
        ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        y_pred_ensemble = ensemble.predict(X_test)
        y_pred_proba_ensemble = ensemble.predict_proba(X_test)[:, 1]
        
        acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
        auc_ensemble = roc_auc_score(y_test, y_pred_proba_ensemble)
        
        print(f"\nEnsemble Results:")
        print(f"  Accuracy: {acc_ensemble:.4f} ({acc_ensemble*100:.2f}%)")
        print(f"  ROC-AUC: {auc_ensemble:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred_ensemble, target_names=['TD', 'ASD']))
        
        # Save ensemble
        with open(output_dir / 'ensemble_model.pkl', 'wb') as f:
            pickle.dump(ensemble, f)
        print(f"\n✓ Ensemble model saved to {output_dir / 'ensemble_model.pkl'}")
        
        # Add to results for comparison
        results['Ensemble'] = {
            'model': ensemble,
            'accuracy': acc_ensemble,
            'auc': auc_ensemble,
            'predictions': y_pred_ensemble,
            'probabilities': y_pred_proba_ensemble
        }
        
        # Update comparison
        comparison_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [r['accuracy'] for r in results.values()],
            'ROC-AUC': [r['auc'] for r in results.values()]
        }).sort_values('Accuracy', ascending=False)
        
        print(f"\n📊 Final Model Comparison (including Ensemble):")
        print(comparison_df.to_string(index=False))
        comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\nAll results saved to: {output_dir.absolute()}")
print("\nNext steps:")
print("  1. Review model comparison results")
print("  2. Tune hyperparameters for best model")
print("  3. Consider ensemble of top 2-3 models")
print("  4. Perform cross-validation for robust evaluation")

