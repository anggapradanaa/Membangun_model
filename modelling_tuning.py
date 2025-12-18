import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, matthews_corrcoef, cohen_kappa_score)
import mlflow
import mlflow.lightgbm
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load preprocessed data"""
    print("=" * 60)
    print("Loading Preprocessed Data...")
    print("=" * 60)
    
    X_train = pd.read_csv('namadataset_preprocessing/X_train.csv')
    X_test = pd.read_csv('namadataset_preprocessing/X_test.csv')
    y_train = pd.read_csv('namadataset_preprocessing/y_train.csv').values.ravel()
    y_test = pd.read_csv('namadataset_preprocessing/y_test.csv').values.ravel()
    
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


def tune_hyperparameters(X_train, y_train):
    """Hyperparameter tuning dengan GridSearchCV"""
    print("\n" + "=" * 60)
    print("Hyperparameter Tuning - GridSearchCV")
    print("=" * 60)
    
    # Check class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\nClass Distribution:")
    print(f"   - Class 0: {counts[0]} ({counts[0]/len(y_train)*100:.1f}%)")
    print(f"   - Class 1: {counts[1]} ({counts[1]/len(y_train)*100:.1f}%)")

    param_grid = {
        'n_estimators': [40, 50, 60],
        'max_depth': [2, 3],
        'learning_rate': [0.12, 0.15, 0.18],
        'num_leaves': [10, 15, 20],
        'min_child_samples': [25, 30, 35],
        'subsample': [0.95, 1.0],
        'colsample_bytree': [0.95, 1.0],
        'reg_alpha': [0, 0.01],
        'reg_lambda': [0, 0.01]
    }
    
    print("\nParameter Grid:")
    total_combinations = 1
    for param, values in param_grid.items():
        print(f"   {param}: {values}")
        total_combinations *= len(values)
    
    # Initialize base model
    base_model = LGBMClassifier(
        random_state=42,
        verbose=-1
    )
    
    # Use StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\nRunning GridSearchCV...")
    print(f"   Testing all {total_combinations} combinations...")
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\nGridSearchCV completed!")
    print(f"\nBest Parameters Found:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    
    print(f"\nCross-Validation Performance:")
    print(f"   Best CV ROC-AUC Score: {grid_search.best_score_:.4f}")
    
    # Overfitting check
    cv_results = pd.DataFrame(grid_search.cv_results_)
    best_idx = grid_search.best_index_
    train_score = cv_results.loc[best_idx, 'mean_train_score']
    test_score = cv_results.loc[best_idx, 'mean_test_score']
    
    print(f"\nOverfitting Check:")
    print(f"   Mean Train Score: {train_score:.4f}")
    print(f"   Mean CV Score:    {test_score:.4f}")
    print(f"   Gap:              {train_score - test_score:.4f}")
    
    if train_score - test_score > 0.05:
        print(f"   Possible overfitting")
    else:
        print(f"   Good generalization")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def evaluate_model(model, X_test, y_test):
    """Evaluate model dan calculate all metrics"""
    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Standard metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Custom metrics (tambahan)
    mcc = matthews_corrcoef(y_test, y_pred)  # Matthews Correlation Coefficient
    kappa = cohen_kappa_score(y_test, y_pred)  # Cohen's Kappa
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Specificity
    specificity = tn / (tn + fp)
    
    # Print metrics
    print("\nModel Performance:")
    print(f"   Accuracy:   {accuracy:.4f}")
    print(f"   Precision:  {precision:.4f}")
    print(f"   Recall:     {recall:.4f}")
    print(f"   F1-Score:   {f1:.4f}")
    print(f"   AUC-ROC:    {auc:.4f}")
    print(f"   Specificity: {specificity:.4f}")
    print(f"   MCC:        {mcc:.4f}")
    print(f"   Cohen Kappa: {kappa:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"   TN: {tn}  FP: {fp}")
    print(f"   FN: {fn}  TP: {tp}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc,
        'specificity': specificity,
        'mcc': mcc,
        'cohen_kappa': kappa,
        'confusion_matrix': cm.tolist()
    }


def log_to_mlflow(model, best_params, best_cv_score, metrics):
    """Manual logging to MLflow"""
    print("\n" + "=" * 60)
    print("Logging to MLflow (Manual)")
    print("=" * 60)
    
    # Set experiment
    mlflow.set_experiment("diabetes-lgbm-tuned")
    
    with mlflow.start_run(run_name="lgbm_tuned_manual"):
        print("\nLogging hyperparameters...")
        # Log hyperparameters
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        mlflow.log_param("model_type", "LightGBM")
        mlflow.log_param("tuning_method", "GridSearchCV")
        mlflow.log_param("cv_folds", 3)
        mlflow.log_param("cv_strategy", "StratifiedKFold")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("scoring", "roc_auc")
        
        print("Hyperparameters logged")
        
        print("\nLogging metrics...")
        # Log standard metrics
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("precision", metrics['precision'])
        mlflow.log_metric("recall", metrics['recall'])
        mlflow.log_metric("f1_score", metrics['f1_score'])
        mlflow.log_metric("auc_roc", metrics['auc_roc'])
        
        # Log custom metrics
        mlflow.log_metric("specificity", metrics['specificity'])
        mlflow.log_metric("mcc", metrics['mcc'])
        mlflow.log_metric("cohen_kappa", metrics['cohen_kappa'])
        
        # Log CV score
        mlflow.log_metric("best_cv_roc_auc", best_cv_score)
        
        print("All metrics logged")
        
        print("\nLogging model...")
        # Log model
        mlflow.lightgbm.log_model(
            lgb_model=model,
            artifact_path="model",
            registered_model_name="lgbm_diabetes_tuned"
        )
        print("Model logged")
        
        # Log confusion matrix as artifact
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = np.array(metrics['confusion_matrix'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Diabetes', 'Diabetes'],
                   yticklabels=['No Diabetes', 'Diabetes'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        mlflow.log_artifact('confusion_matrix.png')
        print("Confusion matrix plot logged")
        
        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow logging completed!")
        print(f"   Run ID: {run_id}")
        
        return run_id


def main():
    """Main training pipeline with tuning"""
    print("\n" + "=" * 60)
    print("MODELLING WITH HYPERPARAMETER TUNING")
    print("=" * 60)
    print("Author: Angga Yulian Adi Pradana")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Hyperparameter tuning
    best_model, best_params, best_cv_score = tune_hyperparameters(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(best_model, X_test, y_test)
    
    # Log to MLflow
    run_id = log_to_mlflow(best_model, best_params, best_cv_score, metrics)
    
    print("\n" + "=" * 60)
    print("TUNING & MODELLING COMPLETED!")
    print("=" * 60)
    print("\nSUMMARY:")
    print("  Hyperparameter tuning completed (GridSearchCV)")
    print("  Class imbalance handled (class_weight='balanced', StratifiedKFold, SMOTE)")
    print("  Best model selected and evaluated")
    print("  All metrics calculated:")
    print(f"     - Accuracy:   {metrics['accuracy']:.4f}")
    print(f"     - Precision:  {metrics['precision']:.4f}")
    print(f"     - Recall:     {metrics['recall']:.4f}")
    print(f"     - F1-Score:   {metrics['f1_score']:.4f}")
    print(f"     - AUC-ROC:    {metrics['auc_roc']:.4f}")
    print(f"     - Specificity: {metrics['specificity']:.4f}")
    print(f"     - MCC:        {metrics['mcc']:.4f}")
    print(f"     - Cohen Kappa: {metrics['cohen_kappa']:.4f}")
    print("  Manual logging to MLflow completed")
    print(f" MLflow Run ID: {run_id}")
    print("\nView results: mlflow ui")
    print("=" * 60)


if __name__ == "__main__":
    main()