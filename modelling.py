import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.lightgbm
import dagshub
import warnings
warnings.filterwarnings('ignore')


# ========== DAGSHUB INTEGRATION ==========
dagshub.init(repo_owner='anggapradanaa', 
             repo_name='Membangun_model', 
             mlflow=True)

# Set MLflow tracking URI ke DagsHub
mlflow.set_tracking_uri("https://dagshub.com/anggapradanaa/Membangun_model.mlflow")
# ==========================================


def load_data():
    """Load preprocessed data"""
    print("=" * 60)
    print("Loading Preprocessed Data...")
    print("=" * 60)
    
    X_train = pd.read_csv('diabetes_preprocessing/X_train.csv')
    X_test = pd.read_csv('diabetes_preprocessing/X_test.csv')
    y_train = pd.read_csv('diabetes_preprocessing/y_train.csv').values.ravel()
    y_test = pd.read_csv('diabetes_preprocessing/y_test.csv').values.ravel()
    
    print(f"Data loaded successfully!")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")
    
    # Print class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\n  Class distribution in training:")
    print(f"    Class 0 (No Diabetes): {counts[0]} ({counts[0]/len(y_train)*100:.1f}%)")
    print(f"    Class 1 (Diabetes): {counts[1]} ({counts[1]/len(y_train)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test


def train_model_autolog(X_train, X_test, y_train, y_test):
    """Train LightGBM model with MLflow autolog"""
    print("\n" + "=" * 60)
    print("Training LightGBM Model (Baseline - MLflow Autolog)")
    print("=" * 60)
    
    # Start MLflow run
    mlflow.set_experiment("diabetes-lgbm-baseline")
    
    # Enable autolog
    mlflow.lightgbm.autolog()
    
    with mlflow.start_run(run_name="lgbm_baseline_dagshub"):
        print("\nMLflow Autolog Enabled")
        print("   - Automatically logs: parameters, metrics, model, signature")
        print("   - Tracking: DagsHub")
        
        # Check class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\nClass Distribution (Already Balanced by SMOTE):")
        print(f"   - Class 0 (No Diabetes): {counts[0]} ({counts[0]/len(y_train)*100:.1f}%)")
        print(f"   - Class 1 (Diabetes): {counts[1]} ({counts[1]/len(y_train)*100:.1f}%)")
        print(f"   - Total samples: {len(y_train)}")
        print(f"   - No additional class weighting needed (data already balanced)")
        
        # Initialize model
        model = LGBMClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=20,
            min_child_samples=15,
            subsample=0.85,
            colsample_bytree=0.95,
            reg_alpha=0.15,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1
        )
        
        # Train model
        print("\nTraining LightGBM...")
        model.fit(X_train, y_train)
        print("Training completed!")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Manual log metrics
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("test_auc_roc", auc)
        mlflow.log_param("tracking", "DagsHub")

        # Print metrics
        print("\nModel Performance (Baseline):")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   AUC-ROC:   {auc:.4f}")
        
        run_id = mlflow.active_run().info.run_id
        print("\n" + "=" * 60)
        print("Model and metrics logged to DagsHub via MLflow autolog")
        print("=" * 60)
        print(f"Run ID: {run_id}")
        print(f"\nView in DagsHub:")
        print(f"https://dagshub.com/anggapradanaa/Membangun_model.mlflow")
        print("=" * 60)
        
        return model, {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc
        }


def main():
    """Main training pipeline"""
    print("\n" + "=" * 60)
    print("MODELLING - LIGHTGBM WITH MLFLOW AUTOLOG")
    print("TRACKING: DAGSHUB")
    print("=" * 60)
    print("Author: Angga Yulian Adi Pradana")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Train model with autolog
    model, metrics = train_model_autolog(X_train, X_test, y_train, y_test)
    
    print("\n" + "=" * 60)
    print("MODELLING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nSUMMARY:")
    print("  LightGBM model trained (baseline)")
    print("  Class imbalance handled (SMOTE)")
    print("  MLflow autolog enabled")
    print("  Logged to DagsHub")
    print("  Metrics logged automatically:")
    print(f"     - Accuracy:  {metrics['accuracy']:.4f}")
    print(f"     - Precision: {metrics['precision']:.4f}")
    print(f"     - Recall:    {metrics['recall']:.4f}")
    print(f"     - F1-Score:  {metrics['f1_score']:.4f}")
    print(f"     - AUC-ROC:   {metrics['auc_roc']:.4f}")
    print("  Model artifact saved to DagsHub MLflow")
    print("\nView results:")
    print("  https://dagshub.com/anggapradanaa/Membangun_model.mlflow")
    print("=" * 60)


if __name__ == "__main__":
    main()
