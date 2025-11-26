#!/bin/bash
# =========================================
# Unified Runner for 4 ML Models
# =========================================

CSV="./turtlebot3_navigation_dataset_10000_with_actual_velocity_labelled.csv"

rm -f results_summary_simple.csv

echo "========================================="
echo " Running Logistic Regression"
echo "========================================="
python train_baseline_logreg.py --csv "$CSV" --out_dir "./artifacts_logreg"

echo "========================================="
echo " Running Random Forest"
echo "========================================="
python train_rf_speed.py --csv "$CSV" --out_dir "./artifacts_rf_speed"

echo "========================================="
echo " Running SVM + PCA"
echo "========================================="
python train_svm_pca.py --csv "$CSV" --out_dir "./artifacts_svm" --pca_dim 14

echo "========================================="
echo " Running XGBoost"
echo "========================================="
python train_xgboost.py --csv "$CSV" --out_dir "./artifacts_xgb"

echo "========================================="
echo " All models finished."
echo "Combined summary saved to results_summary_simple.csv"
echo "========================================="
echo ""
echo "=========== OVERALL COMPARISON TABLE ==========="
column -t -s, results_summary_simple.csv
echo "==============================================="
