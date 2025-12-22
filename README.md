# Membangun Model - Diabetes Prediction

## 📁 Deskripsi Folder
Folder ini berisi implementasi model Machine Learning untuk prediksi diabetes menggunakan Random Forest dengan tracking menggunakan MLflow (local storage).

## 📂 Struktur Folder
```
Membangun_model/
├── diabetes_preprocessing/              # Data hasil preprocessing
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
├── modelling.py                         # Baseline model dengan MLflow autolog
├── modelling_tuning.py                  # Advanced: Hyperparameter tuning
├── requirements.txt                     # Dependencies
├── mlruns/                             # MLflow tracking data (local)
└── artifact_tuning/                    # Generated artifacts from tuning
    ├── classification_report.txt
    ├── confusion_matrix.png
    ├── confusion_matrix.txt
    ├── cv_results_summary.txt
    ├── feature_importance.json
    ├── feature_importance.png
    ├── roc_curve.png
    ├── precision_recall_curve.png
    └── tree_statistics.txt
```

## 🤖 Model & Algoritma
**Algorithm**: Random Forest Classifier
- Efisien untuk dataset tabular
- Robust terhadap overfitting dengan ensemble learning
- Support hyperparameter tuning dengan GridSearchCV

## 📊 Dua Pendekatan Modelling

### 1. **modelling.py** - Baseline Model
Pendekatan baseline dengan MLflow autolog untuk tracking otomatis menggunakan local storage.

**Features:**
- ✅ MLflow autolog enabled
- ✅ Local tracking (file:./mlruns)
- ✅ Automatic logging (params, metrics, model)
- ✅ Simple & straightforward implementation

**Hyperparameters (Baseline):**
```python
n_estimators=100
random_state=42
n_jobs=-1
```

**Model Performance (Baseline):**
| Metric | Value |
|--------|-------|
| **Accuracy** | **0.7273** (72.73%) |
| Precision | 0.7354 |
| Recall | 0.7273 |
| F1-Score | 0.7302 |
| AUC-ROC | 0.8188 |

**Cara Menjalankan:**
```bash
python modelling.py
```

**Melihat Hasil di MLflow UI:**
```bash
mlflow ui
```
Kemudian buka http://localhost:5000 di browser Anda.

---

### 2. **modelling_tuning.py** - Advanced Model (Tuned)
Pendekatan advanced dengan hyperparameter tuning menggunakan GridSearchCV dan tracking ke DagsHub.

**Features:**
- ✅ GridSearchCV untuk hyperparameter tuning
- ✅ StratifiedKFold (5-fold cross-validation)
- ✅ Manual MLflow logging
- ✅ 9 artifacts logged (3 extra beyond standard)
- ✅ Extended metrics (MCC, Cohen's Kappa, Specificity)
- ✅ Tracking ke DagsHub

**Parameter Grid:**
```python
'n_estimators': [50, 100]
'max_depth': [5, 10]
'min_samples_split': [2, 5]
'min_samples_leaf': [1, 2]
'max_features': ['sqrt']
'class_weight': [None, 'balanced']
```

**Metrics Logged:**
- Standard: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Extended: Specificity, MCC, Cohen's Kappa
- CV: Best CV ROC-AUC Score

**Model Performance (Tuned):**
| Metric | Value | Improvement |
|--------|-------|-------------|
| **Accuracy** | **0.7273** (72.73%) | Baseline |
| Precision | 0.5909 | - |
| Recall | 0.7222 | - |
| F1-Score | 0.6500 | - |
| AUC-ROC | 0.8172 | -0.16% |
| **Specificity** | **0.73** | - |
| **MCC** | **0.4360** | - |
| **Cohen's Kappa** | **0.4302** | - |
| **Best CV ROC-AUC** | **0.8930** | - |

> ⚠️ **Note**: Tuned model menunjukkan performa yang berbeda. Model baseline memiliki precision lebih tinggi (0.7354 vs 0.5909), sementara tuned model memiliki recall lebih baik (0.7222 vs 0.7273). Pilihan model tergantung pada kebutuhan: precision tinggi untuk mengurangi false positive, atau recall tinggi untuk mendeteksi lebih banyak kasus diabetes.

**Artifacts Generated (9 files):**
1. `confusion_matrix.png` - Visual confusion matrix
2. `confusion_matrix.txt` - Text confusion matrix
3. `classification_report.txt` - Detailed classification report
4. `feature_importance.json` - Feature importance data
5. `feature_importance.png` - Feature importance plot
6. `roc_curve.png` - ROC curve visualization (EXTRA #1)
7. `precision_recall_curve.png` - Precision-Recall curve (EXTRA #2)
8. `tree_statistics.txt` - Random Forest tree statistics (EXTRA #3)
9. `cv_results_summary.txt` - CV and test results summary

**Cara Menjalankan:**
```bash
python modelling_tuning.py
```

## 🔗 MLflow Tracking

### Baseline Model (Local)
- **Tracking URI**: `file:./mlruns`
- **Experiment**: Diabetes_Prediction_Basic
- **View UI**: `mlflow ui` → http://localhost:5000

### Tuned Model (DagsHub)
- **Repository**: https://dagshub.com/anggapradanaa/Membangun_model.mlflow
- **Experiment**: diabetes-rf-tuned
- Terintegrasi dengan DagsHub untuk:
  - Experiment tracking
  - Model versioning
  - Artifact storage
  - Metrics visualization

## 📈 Evaluasi Model

**Confusion Matrix Components:**
- TN (True Negative): Correctly predicted No Diabetes
- TP (True Positive): Correctly predicted Diabetes
- FP (False Positive): Incorrectly predicted Diabetes
- FN (False Negative): Incorrectly predicted No Diabetes

**Additional Metrics (Tuned Model):**
- **Specificity**: TN / (TN + FP) - True negative rate
- **MCC**: Matthews Correlation Coefficient (-1 to +1)
- **Cohen's Kappa**: Agreement measure between predictions

## 🔧 Requirements

```
pandas
numpy
scikit-learn
mlflow
dagshub
matplotlib
seaborn
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 📝 Workflow

### Baseline Model:
1. **Load Data** → Load preprocessed data dari folder diabetes_preprocessing
2. **Enable Autolog** → MLflow sklearn autolog
3. **Train Model** → Training Random Forest dengan parameter default
4. **Evaluate** → Calculate metrics otomatis
5. **Log to MLflow** → Track experiments ke local storage

### Tuned Model:
1. **Load Data** → Load preprocessed data dari folder diabetes_preprocessing
2. **Hyperparameter Tuning** → GridSearchCV dengan StratifiedKFold
3. **Train Model** → Training dengan best parameters
4. **Evaluate** → Calculate extended metrics
5. **Generate Artifacts** → Create visualizations dan reports
6. **Log to MLflow** → Manual logging ke DagsHub

## 🎯 Output

**Baseline Model:**
- Model tersimpan di `mlruns/` (local)
- Automatic logging via autolog
- View via MLflow UI

**Tuned Model:**
- Hasil tuning tersimpan di DagsHub MLflow
- Best parameters otomatis terseleksi
- Model terbaik tersimpan dengan metrics lengkap
- 9 artifacts comprehensive

## 📊 Monitoring

### Baseline:
```bash
mlflow ui
```
Access at: http://localhost:5000

### Tuned Model:
- **DagsHub MLflow UI**: Metrics, parameters, artifacts
- **DagsHub Repository**: https://dagshub.com/anggapradanaa/Membangun_model.mlflow

## 🔍 Perbandingan Model

| Aspek | Baseline | Tuned |
|-------|----------|-------|
| **Tracking** | Local (mlruns) | DagsHub |
| **Hyperparameter** | Fixed | GridSearchCV |
| **Accuracy** | 0.7273 | 0.7273 |
| **Precision** | 0.7354 | 0.5909 |
| **Recall** | 0.7273 | 0.7222 |
| **AUC-ROC** | 0.8188 | 0.8172 |
| **Artifacts** | Auto (basic) | 9 files (detailed) |
| **CV Score** | - | 0.8930 |

---

**Author**: Angga Yulian Adi Pradana  
**Algorithm**: Random Forest Classifier  
**Tracking**: MLflow (Local & DagsHub)
