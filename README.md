# TurtleBot3 Navigation Anomaly Detection  
Machine Learning Models + Digital Twin Control  
*(Raspberry Pi Compatible)*

---

## 📌 1. Introduction  

This project builds a **navigation anomaly detection system** for TurtleBot3, based on a labeled dataset of 10,000 samples including:

- `timestamp`  
- 360-beam LiDAR scan  
- `linear_velocity` command  
- `angular_velocity` command  
- `actual_velocity` measured  
- `status` label  
  - **0 = Normal**  
  - **1 = Stuck (High Friction / Low Surface Density)**  
  - **2 = Sliding on Steep Descent**  
  - **3 = Unlabeled (on test dataset)**  

We train **four classical ML classifiers**:

1. **Logistic Regression** (baseline)  
2. **Random Forest**  
3. **SVM + PCA**  
4. **XGBoost**  

All models report:

- Validation accuracy  
- Per-sample inference time  
- Memory usage during inference  

A **Digital Twin post-processing module (`dt_control.py`)** translates ML predictions into robot control actions (e.g., braking, slow-crawl correction).

---

## 📦 2. Repository Structure  

```text
RD/
│
├── train_baseline_logreg.py        # Logistic Regression
├── train_rf_speed.py               # Random Forest
├── train_svm_pca.py                # SVM + PCA (with GridSearchCV)
├── train_xgboost.py                # XGBoost
│
├── dt_control.py                   # Digital Twin post-control logic
│
├── run_all_models.sh               # Train all 4 models + generate summary
├── results_summary_simple.csv      # Final summarized comparison table
│
├── turtlebot3_navigation_dataset_10000_with_actual_velocity_labelled.csv
│
├── artifacts_logreg/
├── artifacts_rf_speed/
├── artifacts_svm/
├── artifacts_xgb/
└── (Automatically generated model outputs)
