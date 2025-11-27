"""
train_xgboost_ratio.py
Improved XGBoost version with non-leaky velocity features.
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier
import joblib
import time

# === utils ===
def detect_columns(df):
    cols = {}
    cols["label"] = "status" if "status" in df.columns else None
    cols["timestamp"] = "timestamp" if "timestamp" in df.columns else None
    # detect lidar
    lidar_cols = [c for c in df.columns if "lidar" in c]
    cols["lidar_cols"] = lidar_cols
    # velocity columns
    for key in ["linear_velocity", "angular_velocity", "actual_velocity"]:
        found = [c for c in df.columns if key in c]
        cols[key] = found[0] if found else None
    return cols


def build_features(df, cols):
    """
    Basic feature engineering from LiDAR + velocity inputs
    """
    feats = {}
    lidar = df[cols["lidar_cols"]].to_numpy()
    feats["global_mean"] = lidar.mean(axis=1)
    feats["global_min"] = lidar.min(axis=1)
    feats["global_max"] = lidar.max(axis=1)
    feats["global_std"] = lidar.std(axis=1)

    # Example sector features
    n = len(cols["lidar_cols"])
    n_sectors = 18
    per_sector = n // n_sectors
    for i in range(n_sectors):
        s, e = i * per_sector, (i + 1) * per_sector
        sector = lidar[:, s:e]
        feats[f"sec{i:02d}_mean"] = sector.mean(axis=1)
        feats[f"sec{i:02d}_min"] = sector.min(axis=1)
        feats[f"sec{i:02d}_std"] = sector.std(axis=1)

    # velocity-based features
    for key in ["linear_velocity", "angular_velocity", "actual_velocity"]:
        if cols[key]:
            feats[key] = df[cols[key]]

    feats["ang_over_lin"] = df[cols["angular_velocity"]] / (df[cols["linear_velocity"]] + 1e-3)

    X = pd.DataFrame(feats)
    meta = {
        "n_sectors": n_sectors,
        "lidar_cols": cols["lidar_cols"],
        "feature_names": list(X.columns)
    }
    return X, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--out_dir", type=str, default="artifacts_xgb_ratio")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    cols = detect_columns(df)
    if cols["label"] is None:
        raise ValueError("Cannot find label/status column in CSV")

    # Unify unknown labels: convert -1 → 3
    df.loc[df[cols["label"]] == -1, cols["label"]] = 3

    # Only use labeled subset (0, 1, 2)
    df_trainable = df.iloc[:7000].copy()
    df_trainable = df_trainable[df_trainable[cols["label"]].isin([0, 1, 2])].copy()

    # features
    X_all, meta = build_features(df_trainable, cols)

    # --- remove leaky features if they exist ---
    for bad in ["v_diff", "v_diff_abs"]:
        if bad in X_all.columns:
            X_all.drop(columns=[bad], inplace=True)

    # --- Add new non-leaky velocity-based features ---
    # State | Description | actual vs commanded speed | ratio/log_ratio | absdiff
    # 0 Normal driving | Motor moves robot normally | actual ≈ linear | ratio ≈ 1, log ≈ 0 | absdiff small
    # 1 Stuck (low friction / spinning in place) | Motor spins but robot doesn't move | actual ≪ linear | ratio < 1, log < 0 | absdiff large
    # 2 Sliding downhill | Robot sliding due to gravity | actual ≫ linear | ratio > 1, log > 0 | absdiff large
    #
    # log_ratio sign tells you stuck (<0) or sliding (>0)
    # absdiff magnitude tells you whether anomaly is significant

    eps = 1e-3
    lin_col = cols["linear_velocity"]
    act_col = cols["actual_velocity"]
    print("Detected velocity columns:", lin_col, act_col)

    X_all["speed_ratio"] = (df_trainable[act_col] + eps) / (df_trainable[lin_col] + eps)
    X_all["speed_absdiff"] = abs(df_trainable[act_col] - df_trainable[lin_col])
    X_all["log_speed_ratio"] = np.log(X_all["speed_ratio"])

    y_all = df_trainable[cols["label"]].astype(int).values

    # train/validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=args.seed, stratify=y_all
    )

    # class weights
    classes = np.unique(y_all)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_all)
    weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}
    sample_weight_tr = np.array([weight_dict[int(c)] for c in y_tr])

    # model definition
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        random_state=args.seed
    )

    # === Training time ===
    t0 = time.time()
    model.fit(
        X_tr, y_tr,
        sample_weight=sample_weight_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    train_time = time.time() - t0
    print(f"\nTraining time: {train_time:.3f} seconds")

    # === Validation inference time ===
    t1 = time.time()
    y_pred = model.predict(X_val)
    val_time = time.time() - t1
    avg_val_time = val_time / len(X_val)
    print(f"Validation inference time: {val_time:.3f} sec total, {avg_val_time:.6f} sec/sample")

    print("\n================= VALIDATION (from first 7000) =================")
    acc = (y_pred == y_val).mean()
    f1_macro = np.mean([
        classification_report(y_val, y_pred, output_dict=True, zero_division=0)[str(k)]["f1-score"]
        for k in np.unique(y_val)
    ])
    print(f"Accuracy: {acc:.4f}   Macro-F1: {f1_macro:.4f}\n")
    print("Confusion matrix (labels 0,1,2):")
    print(confusion_matrix(y_val, y_pred, labels=[0, 1, 2]))
    print("\nClassification report:")
    print(classification_report(y_val, y_pred, digits=4))

    # save model + metadata
    joblib.dump(model, out_dir / "xgb_model.joblib")
    with open(out_dir / "feature_meta.json", "w") as f:
        meta["timing"] = {"train_sec": train_time, "val_sec_per_sample": avg_val_time}
        json.dump(meta, f, indent=2)
    print(f"\nSaved model to: {out_dir / 'xgb_model.joblib'}")
    print(f"Saved feature meta to: {out_dir / 'feature_meta.json'}\n")

    # feature importance
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1][:20]
    print("Top-20 feature importance:")
    for rank, i in enumerate(sorted_idx, 1):
        print(f"{rank:2d}. {X_all.columns[i]:24s} {importance[i]:.5f}")

    # === Test inference time ===
    df_test = df.iloc[7000:].copy()
    X_test, _ = build_features(df_test, cols)
    X_test["speed_ratio"] = (df_test[act_col] + eps) / (df_test[lin_col] + eps)
    X_test["speed_absdiff"] = abs(df_test[act_col] - df_test[lin_col])
    X_test["log_speed_ratio"] = np.log(X_test["speed_ratio"])

    t2 = time.time()
    proba = model.predict_proba(X_test)
    pred = model.predict(X_test)
    test_time = time.time() - t2
    avg_test_time = test_time / len(X_test)
    print(f"Test inference time: {test_time:.3f} sec total, {avg_test_time:.6f} sec/sample")

    out_pred = pd.DataFrame({
        "timestamp": df_test[cols["timestamp"]] if cols["timestamp"] else np.arange(len(df_test)),
        "pred_class": pred,
        "p0": proba[:, 0],
        "p1": proba[:, 1],
        "p2": proba[:, 2],
    })
    out_pred.to_csv(out_dir / "test_predictions.csv", index=False)
    print(f"\nSaved test predictions to: {out_pred}\n")

    # distribution of predictions
    print("================= TEST (last 3000, unlabeled) =================")
    vals, cnts = np.unique(pred, return_counts=True)
    total = cnts.sum()
    for v, c in zip(vals, cnts):
        print(f"class {v}: {c:4d} ({100*c/total:6.2f}%)")
    print("===============================================================")

    # ===== Execution Summary (simplified) =====
    process = psutil.Process()
    mem_usage = process.memory_info().rss / (1024 ** 2)  # MB
    mem_per_sample = mem_usage / len(y_val)
    val_acc = accuracy_score(y_val, y_pred)
    model_name = "XGBoost"

    results = {
        "Model": model_name,
        "ValInferTime_sec_per_sample": f"{avg_val_time * 1:.2e} s",
        "ValAccuracy": f"{val_acc:.4f}",
        "MemUsage_MB_per_sample": f"{(mem_usage / len(y_val)):.3f} MB",
    }

    # ==== Print concise summary ====
    print("\n===== EXECUTION SUMMARY (Simplified) =====")
    for k, v in results.items():
        print(f"{k:30s}: {v}")
    print("==========================================")

    # save summary file
    import csv
    summary_file = "results_summary_simple.csv"
    with open(summary_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(results)

if __name__ == "__main__":
    main()
