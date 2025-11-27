import pandas as pd
import numpy as np
import joblib, json, argparse
from pathlib import Path
import psutil
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import time


# ====================== Automatically detect key columns ======================
def detect_columns(df):
    cols = {}
    for c in df.columns:
        cl = c.lower()
        if "status" in cl or "label" in cl:
            cols["label"] = c
        elif "timestamp" in cl or "time" in cl:
            cols["timestamp"] = c
    lidar_cols = [c for c in df.columns if "lidar" in c.lower()]
    cols["lidar_cols"] = lidar_cols
    for k in ["linear", "angular", "actual"]:
        found = [c for c in df.columns if k in c.lower() and "velocity" in c.lower()]
        cols[f"{k}_velocity"] = found[0] if found else None
    return cols


# ====================== Feature construction function ======================
def build_features(df, cols):
    lidar = df[cols["lidar_cols"]].values
    global_min = np.min(lidar, axis=1)
    global_max = np.max(lidar, axis=1)
    global_mean = np.mean(lidar, axis=1)
    global_std = np.std(lidar, axis=1)
    q25 = np.quantile(lidar, 0.25, axis=1)
    q75 = np.quantile(lidar, 0.75, axis=1)
    front = lidar[:, :30]
    front_min = np.min(front, axis=1)
    thr_02 = np.sum(lidar < 0.2, axis=1)
    thr_05 = np.sum(lidar < 0.5, axis=1)

    v_lin = df[cols["linear_velocity"]].values
    v_ang = df[cols["angular_velocity"]].values
    v_act = df[cols["actual_velocity"]].values

    eps = 1e-3
    speed_ratio = (v_act + eps) / (v_lin + eps)
    log_speed_ratio = np.log(speed_ratio)
    speed_absdiff = np.abs(v_act - v_lin)

    ang_over_lin = np.abs(v_ang) / (np.abs(v_lin) + eps)

    feat = {
        "global_min": global_min,
        "global_max": global_max,
        "global_mean": global_mean,
        "global_std": global_std,
        "q25": q25, "q75": q75,
        "front_min": front_min,
        "count_lt_0.2": thr_02,
        "count_lt_0.5": thr_05,
        "linear_velocity": v_lin,
        "angular_velocity": v_ang,
        "actual_velocity": v_act,
        "ang_over_lin": ang_over_lin,
        "speed_ratio": speed_ratio,
        "log_speed_ratio": log_speed_ratio,
        "speed_absdiff": speed_absdiff
    }
    X = pd.DataFrame(feat)
    meta = {"feature_names": list(X.columns)}
    return X, meta


# ====================== Main function ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", default="artifacts_svm_speed")
    parser.add_argument("--pca_dim", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ==== 1. Load data ====
    df = pd.read_csv(args.csv)
    cols = detect_columns(df)
    df_trainable = df.iloc[:7000].copy()
    df_trainable = df_trainable[df_trainable[cols["label"]].isin([0, 1, 2])].copy()

    # ==== 2. Build features ====
    X_all, meta = build_features(df_trainable, cols)
    y_all = df_trainable[cols["label"]].astype(int).values

    # ==== 3. Train/validation split ====
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=args.seed, stratify=y_all
    )

    # ==== 4. Class weights ====
    classes = np.unique(y_all)
    cw = compute_class_weight("balanced", classes=classes, y=y_all)
    weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}
    sample_weight_tr = np.array([weight_dict[int(c)] for c in y_tr])

    # ==== 5. Dynamic PCA dimension ====
    n_feats = X_tr.shape[1]
    pca_dim_eff = min(args.pca_dim, n_feats - 1)
    if pca_dim_eff < 1:
        pca_dim_eff = n_feats
    print(f"[INFO] n_features={n_feats}, using pca_dim={pca_dim_eff}")

    # ==== 6. Pipeline ====
    # Radial Basis Function kernel maps original features into a high-dimensional space 
    # where the data may become linearly separable.
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pca_dim_eff, random_state=args.seed)),
        ("svm", SVC(kernel="rbf", probability=True, class_weight=None))
    ])

    # ==== 7. Grid search ====
    param_grid = {
        "svm__C": [0.5, 1.0, 2.0, 5.0, 10.0],
        "svm__gamma": ["scale", 0.5, 0.2, 0.1, 0.05, 0.02]
    }
    # 3-fold cross-validation: each parameter set is evaluated across 3 splits
    gs = GridSearchCV(
        pipe, param_grid, scoring="f1_macro", n_jobs=-1, cv=3, verbose=1, error_score="raise"
    )

    # ==== 8. Training (timed) ====
    t0 = time.time()
    gs.fit(X_tr, y_tr, svm__sample_weight=sample_weight_tr)
    train_time = time.time() - t0
    print(f"\nTraining time: {train_time:.3f} seconds")

    # ==== 9. Validation results (timed) ====
    t1 = time.time()
    y_pred = gs.predict(X_val)
    val_time = time.time() - t1
    avg_val_time = val_time / len(X_val)
    print(f"Validation inference time: {val_time:.3f} sec total, {avg_val_time:.6f} sec/sample")

    print("\nBest params:", gs.best_params_)
    print("\nConfusion matrix (val) [labels 0,1,2]:")
    print(confusion_matrix(y_val, y_pred, labels=[0, 1, 2]))
    print("\nClassification report (val):")
    print(classification_report(y_val, y_pred, digits=4))

    # ==== 10. Save model ====
    joblib.dump(gs.best_estimator_, out_dir / "svm_pca_speed.joblib")
    meta["columns"] = cols
    meta["class_weight"] = weight_dict
    meta["pca_dim"] = pca_dim_eff
    meta["timing"] = {"train_sec": train_time, "val_sec_per_sample": avg_val_time}
    with open(out_dir / "feature_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved model to: {out_dir/'svm_pca_speed.joblib'}")

    # ==== 11. Predict on last 3000 samples (timed) ====
    df_test = df.iloc[7000:].copy()
    X_test, _ = build_features(df_test, cols)

    t2 = time.time()
    proba = gs.predict_proba(X_test)
    pred = gs.predict(X_test)
    test_time = time.time() - t2
    avg_test_time = test_time / len(X_test)
    print(f"Test inference time: {test_time:.3f} sec total, {avg_test_time:.6f} sec/sample")

    out = pd.DataFrame({
        "timestamp": df_test[cols["timestamp"]] if cols.get("timestamp") else np.arange(len(df_test)),
        "pred_class": pred,
        "p0": proba[:, 0],
        "p1": proba[:, 1],
        "p2": proba[:, 2]
    })
    out.to_csv(out_dir / "test_predictions.csv", index=False)
    print(f"Saved test predictions to: {out_dir/'test_predictions.csv'}")

    vals, cnts = np.unique(pred, return_counts=True)
    print("\n================= TEST (last 3000, unlabeled) =================")
    for v, c in zip(vals, cnts):
        print(f"class {v}: {c:4d} ({100 * c / len(pred):6.2f}%)")
    print("===============================================================")

    # ===== Execution Summary (simplified) =====
    process = psutil.Process()
    mem_usage = process.memory_info().rss / (1024 ** 2)  # MB
    mem_per_sample = mem_usage / len(y_val)
    val_acc = accuracy_score(y_val, y_pred)
    model_name = "SVM + PCA"

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

    # Save summary
    import csv
    summary_file = "results_summary_simple.csv"
    with open(summary_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(results)

if __name__ == "__main__":
    main()
