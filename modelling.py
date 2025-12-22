import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


# Set MLflow tracking URI (local storage)
mlflow.set_tracking_uri("file:./mlruns")

# Set experiment name
mlflow.set_experiment("Diabetes_Prediction_Basic")


def load_data():
    """Load preprocessed training and testing data"""
    print("Loading preprocessed data...")
    
    # Load training data
    X_train = pd.read_csv('diabetes_preprocessing/X_train.csv')
    y_train = pd.read_csv('diabetes_preprocessing/y_train.csv')
    
    # Load testing data
    X_test = pd.read_csv('diabetes_preprocessing/X_test.csv')
    y_test = pd.read_csv('diabetes_preprocessing/y_test.csv')
    
    # Convert to numpy arrays if needed
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    
    print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing data shape: X_test={X_test.shape}, y_test={y_test.shape}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test):
    """Train Random Forest model with MLflow autolog"""
    
    # Enable MLflow autolog for scikit-learn
    # Disable log_models to prevent auto-registration
    mlflow.sklearn.autolog(log_models=False)
    
    print("\nStarting MLflow run...")
    
    with mlflow.start_run(run_name="RandomForest_Basic_Model"):
        
        print("Training Random Forest model...")
        
        # Create and train the model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Explicitly log model to run artifacts ONLY
        print("Logging model to artifacts...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=None
        )
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        
        print("\nModel training completed successfully!")
        print(f"Model and metrics saved to MLflow tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        
        return model


def main():
    """Main function to orchestrate the training process"""
    print("=" * 60)
    print("Diabetes Prediction Model Training")
    print("Using MLflow with Autolog (Basic Level)")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Train model
    model = train_model(X_train, X_test, y_train, y_test)
    
    print("\n" + "=" * 60)
    print("Training process completed!")
    print("To view results, run: mlflow ui")
    print("Then open http://localhost:5000 in your browser")
    print("=" * 60)


if __name__ == "__main__":
    main()
