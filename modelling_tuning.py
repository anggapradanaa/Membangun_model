import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, matthews_corrcoef, cohen_kappa_score)
import mlflow
import mlflow.sklearn
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
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt'],
        'class_weight': [None, 'balanced']
    }
    
    print("\nParameter Grid:")
    total_combinations = 1
    for param, values in param_grid.items():
        print(f"   {param}: {values}")
        total_combinations *= len(values)
    
    # Initialize base model
    base_model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
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


def log_to_mlflow(model, best_params, best_cv_score, metrics, X_train, X_test, y_test):
    """Manual logging to MLflow (DagsHub)"""
    print("\n" + "=" * 60)
    print("Logging to MLflow (DagsHub)")
    print("=" * 60)
    
    # Create artifact directory
    import os
    artifact_dir = 'artifact_tuning'
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)
        print(f"\n✓ Created directory: {artifact_dir}/")
    
    mlflow.set_experiment("diabetes-rf-tuned")
    
    with mlflow.start_run(run_name="rf_tuned_dagshub"):
        print("\nLogging hyperparameters...")
        # Log hyperparameters
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("tuning_method", "GridSearchCV")
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("cv_strategy", "StratifiedKFold")
        mlflow.log_param("scoring", "roc_auc")
        mlflow.log_param("tracking", "DagsHub")
        
        print("Hyperparameters logged")
        
        print("\nLogging metrics...")
    
        mlflow.log_metric("test_accuracy", metrics['accuracy'])      
        mlflow.log_metric("test_precision", metrics['precision'])    
        mlflow.log_metric("test_recall", metrics['recall'])          
        mlflow.log_metric("test_f1_score", metrics['f1_score'])      
        mlflow.log_metric("test_auc_roc", metrics['auc_roc'])        
        
        # Tambahan metrics
        mlflow.log_metric("test_specificity", metrics['specificity'])
        mlflow.log_metric("test_mcc", metrics['mcc'])
        mlflow.log_metric("test_cohen_kappa", metrics['cohen_kappa'])
        mlflow.log_metric("best_cv_roc_auc", best_cv_score)
        
        print("All metrics logged")
        
        print("\nLogging model...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="rf_diabetes_tuned_dagshub"
        )
        print("Model logged")
        
        # ===== ARTIFACTS =====
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        print("\nCreating and logging artifacts...")
        
        # 1. Confusion Matrix PNG
        cm = np.array(metrics['confusion_matrix'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Diabetes', 'Diabetes'],
                   yticklabels=['No Diabetes', 'Diabetes'])
        plt.title('Confusion Matrix - Random Forest')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        confusion_matrix_path = os.path.join(artifact_dir, 'confusion_matrix.png')
        plt.savefig(confusion_matrix_path)
        plt.close()
        mlflow.log_artifact(confusion_matrix_path)
        print("   ✓ Confusion matrix PNG logged")
        
        # 2. Confusion Matrix TXT
        confusion_matrix_txt_path = os.path.join(artifact_dir, 'confusion_matrix.txt')
        with open(confusion_matrix_txt_path, 'w') as f:
            f.write("Confusion Matrix - Random Forest\n")
            f.write("=" * 50 + "\n\n")
            tn, fp, fn, tp = cm.ravel()
            f.write(f"True Negatives  (TN): {tn}\n")
            f.write(f"False Positives (FP): {fp}\n")
            f.write(f"False Negatives (FN): {fn}\n")
            f.write(f"True Positives  (TP): {tp}\n\n")
            f.write(f"Total: {tn + fp + fn + tp}\n")
        mlflow.log_artifact(confusion_matrix_txt_path)
        print("   ✓ Confusion matrix TXT logged")
        
        # 3. Classification Report
        y_pred = model.predict(X_test)
        report_text = classification_report(y_test, y_pred, 
                                           target_names=['No Diabetes', 'Diabetes'])
        classification_report_path = os.path.join(artifact_dir, 'classification_report.txt')
        with open(classification_report_path, 'w') as f:
            f.write("Classification Report - Random Forest\n")
            f.write("=" * 50 + "\n\n")
            f.write(report_text)
        mlflow.log_artifact(classification_report_path)
        print("   ✓ Classification report logged")
        
        # 4. Feature Importance - JSON
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance_json_path = os.path.join(artifact_dir, 'feature_importance.json')
        feature_importance.to_json(feature_importance_json_path, 
                                   orient='records', indent=2)
        mlflow.log_artifact(feature_importance_json_path)
        print("   ✓ Feature importance JSON logged")
        
        # 5. Feature Importance - PNG
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance (Gini)')
        plt.title('Top 20 Feature Importances - Random Forest')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        feature_importance_png_path = os.path.join(artifact_dir, 'feature_importance.png')
        plt.savefig(feature_importance_png_path, dpi=100, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(feature_importance_png_path)
        print("   ✓ Feature importance PNG logged")
        
        # 6. ROC Curve - PNG (ARTEFAK TAMBAHAN #1)
        from sklearn.metrics import roc_curve, auc
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        roc_curve_path = os.path.join(artifact_dir, 'roc_curve.png')
        plt.savefig(roc_curve_path, dpi=100, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(roc_curve_path)
        print("   ✓ ROC curve PNG logged (EXTRA ARTIFACT #1)")
        
        # 7. Precision-Recall Curve - PNG (ARTEFAK TAMBAHAN #2)
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall_vals, precision_vals, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        pr_curve_path = f'{artifact_dir}/precision_recall_curve.png'
        plt.savefig(pr_curve_path, dpi=100, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(pr_curve_path)
        print("   ✓ Precision-Recall curve PNG logged (EXTRA ARTIFACT #2)")
        
        # 8. CV Results Summary
        cv_results_path = f'{artifact_dir}/cv_results_summary.txt'
        with open(cv_results_path, 'w') as f:
            f.write("Cross-Validation & Test Results Summary\n")
            f.write("Model: Random Forest with Hyperparameter Tuning\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Best CV ROC-AUC Score: {best_cv_score:.4f}\n\n")
            f.write("Test Set Metrics:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Accuracy:      {metrics['accuracy']:.4f}\n")
            f.write(f"Precision:     {metrics['precision']:.4f}\n")
            f.write(f"Recall:        {metrics['recall']:.4f}\n")
            f.write(f"F1-Score:      {metrics['f1_score']:.4f}\n")
            f.write(f"AUC-ROC:       {metrics['auc_roc']:.4f}\n")
            f.write(f"Specificity:   {metrics['specificity']:.4f}\n")
            f.write(f"MCC:           {metrics['mcc']:.4f}\n")
            f.write(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}\n")
        mlflow.log_artifact(cv_results_path)
        print("   ✓ CV results summary logged")
        
        # 9. Tree Statistics - TXT (ARTEFAK TAMBAHAN #3)
        tree_stats_path = f'{artifact_dir}/tree_statistics.txt'
        with open(tree_stats_path, 'w') as f:
            f.write("Random Forest Tree Statistics\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Number of Trees: {model.n_estimators}\n")
            f.write(f"Number of Features: {model.n_features_in_}\n")
            f.write(f"Number of Classes: {model.n_classes_}\n")
            f.write(f"Out-of-Bag Score: {model.oob_score_ if hasattr(model, 'oob_score_') else 'N/A'}\n\n")
            
            # Tree depths
            tree_depths = [tree.get_depth() for tree in model.estimators_]
            f.write("Tree Depth Statistics:\n")
            f.write(f"   Min Depth: {np.min(tree_depths)}\n")
            f.write(f"   Max Depth: {np.max(tree_depths)}\n")
            f.write(f"   Mean Depth: {np.mean(tree_depths):.2f}\n")
            f.write(f"   Median Depth: {np.median(tree_depths):.2f}\n\n")
            
            # Number of leaves
            n_leaves = [tree.get_n_leaves() for tree in model.estimators_]
            f.write("Number of Leaves Statistics:\n")
            f.write(f"   Min Leaves: {np.min(n_leaves)}\n")
            f.write(f"   Max Leaves: {np.max(n_leaves)}\n")
            f.write(f"   Mean Leaves: {np.mean(n_leaves):.2f}\n")
            f.write(f"   Median Leaves: {np.median(n_leaves):.2f}\n")
        mlflow.log_artifact(tree_stats_path)
        print("   ✓ Tree statistics logged (EXTRA ARTIFACT #3)")
        
        run_id = mlflow.active_run().info.run_id
        print(f"\n{'='*60}")
        print(f"MLflow logging to DagsHub completed!")
        print(f"{'='*60}")
        print(f"Run ID: {run_id}")
        print(f"Artifacts saved to: {artifact_dir}/")
        print(f"Total artifacts logged: 9 files")
        print(f"   • Standard artifacts: 6 files")
        print(f"   • Extra artifacts (beyond autolog): 3 files")
        print(f"     1. ROC Curve PNG")
        print(f"     2. Precision-Recall Curve PNG")
        print(f"     3. Tree Statistics TXT")
        print(f"\nView in DagsHub:")
        print(f"https://dagshub.com/anggapradanaa/Membangun_model.mlflow")
        print(f"{'='*60}")
        
        return run_id


def main():
    """Main training pipeline with tuning"""
    print("\n" + "=" * 60)
    print("MODELLING WITH HYPERPARAMETER TUNING")
    print("MODEL: RANDOM FOREST")
    print("TRACKING: DAGSHUB")
    print("=" * 60)
    print("Author: Angga Yulian Adi Pradana")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Hyperparameter tuning
    best_model, best_params, best_cv_score = tune_hyperparameters(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(best_model, X_test, y_test)
    
    # Log to MLflow (DagsHub)
    run_id = log_to_mlflow(best_model, best_params, best_cv_score, metrics, 
                           X_train, X_test, y_test)
    
    print("\n" + "=" * 60)
    print("TUNING & MODELLING COMPLETED!")
    print("=" * 60)
    print("\nSUMMARY:")
    print("  ✓ Hyperparameter tuning completed (GridSearchCV)")
    print("  ✓ Best Random Forest model selected and evaluated")
    print("  ✓ Manual logging to DagsHub completed")
    print(f"  ✓ MLflow Run ID: {run_id}")
    print("\n  All metrics calculated:")
    print(f"     - Accuracy:    {metrics['accuracy']:.4f}")
    print(f"     - Precision:   {metrics['precision']:.4f}")
    print(f"     - Recall:      {metrics['recall']:.4f}")
    print(f"     - F1-Score:    {metrics['f1_score']:.4f}")
    print(f"     - AUC-ROC:     {metrics['auc_roc']:.4f}")
    print(f"     - Specificity: {metrics['specificity']:.4f}")
    print(f"     - MCC:         {metrics['mcc']:.4f}")
    print(f"     - Cohen Kappa: {metrics['cohen_kappa']:.4f}")
    print("\n  Artifacts logged (9 files):")
    print("     1. confusion_matrix.png")
    print("     2. confusion_matrix.txt")
    print("     3. classification_report.txt")
    print("     4. feature_importance.json")
    print("     5. feature_importance.png")
    print("     6. roc_curve.png (EXTRA)")
    print("     7. precision_recall_curve.png (EXTRA)")
    print("     8. tree_statistics.txt (EXTRA)")
    print("     9. cv_results_summary.txt")
    print("\n  View in DagsHub:")
    print("  https://dagshub.com/anggapradanaa/Membangun_model.mlflow")
    print("=" * 60)


if __name__ == "__main__":
    main()
