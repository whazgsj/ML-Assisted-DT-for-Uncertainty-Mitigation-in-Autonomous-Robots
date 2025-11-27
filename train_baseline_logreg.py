import argparse
import json
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import accuracy_score
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
import time

def detect_columns(df):
    label_col = None
    for c in df.columns:
        if c.lower() == "status":
            label_col = c
            break
    time_col = None
    for c in df.columns:
        if "time" in c.lower():
            time_col = c
            break

    lidar_cols = [c for c in df.columns if c.lower().startswith("lidar_")]
    linear_col = None
    angular_col = None
    actual_col = None
    for c in df.columns:
        cl = c.lower()
        if "linear" in cl and "velocity" in cl:
            linear_col = c
        if "angular" in cl and "velocity" in cl:
            angular_col = c
        if "actual" in cl and "velocity" in cl:
            actual_col = c

    return {
        "label": label_col,
        "time": time_col,
        "lidar": lidar_cols,
        "linear_velocity": linear_col,
        "angular_velocity": angular_col,
        "actual_velocity": actual_col,
    }

# Compute mean/min/std for each sector to obtain 18×3 = 54 spatial-distribution features
def make_sector_indices(n_points=360, n_sectors=18):
    per = n_points // n_sectors
    sectors = []
    start = 0
    for i in range(n_sectors):
        end = start + per if i < n_sectors - 1 else n_points
        sectors.append((start, end))
        start = end
    return sectors

def build_features(df, cols):
    lidar_cols = cols["lidar"]
    lin_c = cols["linear_velocity"]
    ang_c = cols["angular_velocity"]
    act_c = cols["actual_velocity"]

    if len(lidar_cols) != 360:
        raise ValueError(f"Expected 360 lidar columns, got {len(lidar_cols)}")

    lidar = df[lidar_cols].values.astype(float)

    # Global statistics
    global_min = np.min(lidar, axis=1)
    global_max = np.max(lidar, axis=1)
    global_mean = np.mean(lidar, axis=1)
    global_std = np.std(lidar, axis=1)
    q10 = np.quantile(lidar, 0.10, axis=1)
    q25 = np.quantile(lidar, 0.25, axis=1)
    q50 = np.quantile(lidar, 0.50, axis=1)
    q75 = np.quantile(lidar, 0.75, axis=1)
    q90 = np.quantile(lidar, 0.90, axis=1)

    # Heuristic windows
    def idx_range(start, end):
        if start <= end:
            return list(range(start, end+1))
        else:
            return list(range(start, 360)) + list(range(0, end+1))

    front_idx = np.array(idx_range(330, 359) + idx_range(0, 30))
    left_idx  = np.array(idx_range(60, 120))
    right_idx = np.array(idx_range(240, 300))

    front_min = np.min(lidar[:, front_idx], axis=1)
    left_min  = np.min(lidar[:, left_idx], axis=1)
    right_min = np.min(lidar[:, right_idx], axis=1)

    # Direction-derived features (asymmetry and relative front-facing hazard level)
    lr_asym_min = left_min - right_min
    front_vs_mean = front_min - global_mean

    # Count how many LiDAR points in each frame fall below the given distance thresholds
    thr_02 = np.sum(lidar < 0.2, axis=1)
    thr_03 = np.sum(lidar < 0.3, axis=1)
    thr_05 = np.sum(lidar < 0.5, axis=1)

    sectors = make_sector_indices(n_points=360, n_sectors=18)
    sec_means, sec_mins, sec_stds = [], [], []
    for (s, e) in sectors:
        block = lidar[:, s:e]
        sec_means.append(np.mean(block, axis=1))
        sec_mins.append(np.min(block, axis=1))
        sec_stds.append(np.std(block, axis=1))
    sec_means = np.vstack(sec_means).T
    sec_mins  = np.vstack(sec_mins).T
    sec_stds  = np.vstack(sec_stds).T

    # linear_velocity / angular_velocity / actual_velocity
    v_lin = df[lin_c].values.astype(float)
    v_ang = df[ang_c].values.astype(float)
    v_act = df[act_c].values.astype(float)
    v_ratio = np.divide(np.abs(v_ang), (np.abs(v_lin) + 1e-6))

    features = {
        "global_min": global_min,
        "global_max": global_max,
        "global_mean": global_mean,
        "global_std": global_std,
        "q10": q10, "q25": q25, "q50": q50, "q75": q75, "q90": q90,
        "front_min": front_min, "left_min": left_min, "right_min": right_min,
        "lr_asym_min": lr_asym_min, "front_vs_mean": front_vs_mean,
        "count_lt_0.2": thr_02, "count_lt_0.3": thr_03, "count_lt_0.5": thr_05,
        "linear_velocity": v_lin, "angular_velocity": v_ang, "actual_velocity": v_act,
        "ang_over_lin": v_ratio
    }

    for i in range(sec_means.shape[1]):
        features[f"sec{i:02d}_mean"] = sec_means[:, i]
        features[f"sec{i:02d}_min"]  = sec_mins[:, i]
        features[f"sec{i:02d}_std"]  = sec_stds[:, i]

    X = pd.DataFrame(features)
    meta = {
        "feature_names": list(X.columns),
        "n_features": X.shape[1],
        "sectors": sectors
    }
    return X, meta

def main():
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--out_dir", type=str, default="artifacts_logreg")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CSV and automatically detect key columns
    df = pd.read_csv(args.csv)
    cols = detect_columns(df)

    if cols["label"] is None:
        raise ValueError("Cannot find 'status' column.")

    # Normalize labels: -1 -> 3 (unknown)
    df.loc[df[cols["label"]] == -1, cols["label"]] = 3

    # Use first 7000 rows with labels in {0,1,2} for supervised training
    df_trainable = df.iloc[:7000].copy()
    df_trainable = df_trainable[df_trainable[cols["label"]].isin([0,1,2])].copy()

    # Feature engineering
    X_all, meta = build_features(df_trainable, cols)
    y_all = df_trainable[cols["label"]].astype(int).values

    # Train/validation split (stratified)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=args.seed, stratify=y_all
    )

    # For reproducibility: if switching models later (e.g., XGBoost/SVM/RandomForest),
    # these class weights can be reused for consistent comparison
    classes = np.unique(y_all)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_all)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}

    # Build training pipeline: Standardization + Multinomial Logistic Regression
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            multi_class="multinomial",
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            n_jobs=-1
        ))
    ])

    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    train_time = time.time() - t0
    print(f"\nTraining time: {train_time:.3f} seconds")

    # Confusion matrix explanation:
    # precision: among predicted class X samples, how many are correct
    # recall: among true class X samples, how many are found
    # f1-score: harmonic mean of precision and recall
    # support: number of samples per class
    # accuracy: overall correctness ratio
    # macro avg: unweighted average across classes
    # weighted avg: weighted average by class sample count

    # Validation inference + timing
    t1 = time.time()
    y_pred = pipe.predict(X_val)
    val_time = time.time() - t1
    avg_val_time = val_time / len(X_val)
    print(f"Validation inference time: {val_time:.3f} sec total, {avg_val_time:.6f} sec/sample")

    print("\nConfusion matrix (val) [labels 0,1,2]:")
    print(confusion_matrix(y_val, y_pred, labels=[0, 1, 2]))
    print("\nClassification report (val):")
    print(classification_report(y_val, y_pred, digits=4))

    # Save model and metadata
    model_path = out_dir / "logreg_pipeline.joblib"
    joblib.dump(pipe, model_path)

    meta["class_weight"] = class_weight_dict
    meta["label_mapping"] = {
        0: "safe/normal",
        1: "stuck (low friction/density)",
        2: "sliding on steep descent"
    }
    meta["columns"] = cols
    meta["timing"] = {"train_sec": train_time, "val_sec_per_sample": avg_val_time}

    with open(out_dir / "feature_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved feature meta to: {out_dir/'feature_meta.json'}")

    # Test inference + timing
    df_test = df.iloc[7000:].copy()
    X_test_all, _ = build_features(df_test, cols)

    t2 = time.time()
    proba = pipe.predict_proba(X_test_all)
    pred = pipe.predict(X_test_all)
    test_time = time.time() - t2
    avg_test_time = test_time / len(X_test_all)
    print(f"Test inference time: {test_time:.3f} sec total, {avg_test_time:.6f} sec/sample")

    out_pred = pd.DataFrame()
    if "timestamp" in df_test.columns:
        out_pred["timestamp"] = df_test["timestamp"].values
    out_pred["pred_class"] = pred
    out_pred["p0"], out_pred["p1"], out_pred["p2"] = proba[:, 0], proba[:, 1], proba[:, 2]
    out_pred.to_csv(out_dir / "test_predictions.csv", index=False)
    print(f"\nSaved test predictions to: {out_dir/'test_predictions.csv'}")

    # Print distribution of predicted classes in the test set
    vals, cnts = np.unique(pred, return_counts=True)
    print("\n================= TEST (last 3000, unlabeled) =================")
    for v, c in zip(vals, cnts):
        print(f"class {v}: {c:4d} ({100 * c / len(pred):6.2f}%)")
    print("===============================================================")

    # Execution summary (simplified)
    process = psutil.Process()
    mem_usage = process.memory_info().rss / (1024 ** 2)  # MB
    val_acc = accuracy_score(y_val, y_pred)
    model_name = "Logistic Regression"

    results = {
        "Model": model_name,
        "ValInferTime_sec_per_sample": f"{avg_val_time * 1:.2e} s",
        "ValAccuracy": f"{val_acc:.4f}",
        "MemUsage_MB_per_sample": f"{(mem_usage / len(y_val)):.3f} MB",
    }

    # Print summary
    print("\n===== EXECUTION SUMMARY (Simplified) =====")
    for k, v in results.items():
        print(f"{k:30s}: {v}")
    print("==========================================")

    # Save summary CSV
    import csv
    summary_file = "results_summary_simple.csv"
    with open(summary_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(results)

if __name__ == "__main__":
    main()
