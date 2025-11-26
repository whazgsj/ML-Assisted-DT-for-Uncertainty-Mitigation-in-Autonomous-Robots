import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import time
import psutil
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier


# ============== 列自动检测 ==============
def detect_columns(df: pd.DataFrame):
    cols = {}
    for c in df.columns:
        cl = c.lower()
        if "status" in cl or "label" in cl:
            cols["label"] = c
        elif "timestamp" in cl or cl == "time":
            cols["timestamp"] = c

    lidar_cols = [c for c in df.columns if "lidar" in c.lower()]
    if not lidar_cols:
        raise ValueError("No lidar columns detected (expect columns containing 'lidar').")
    cols["lidar_cols"] = lidar_cols

    for k in ["linear", "angular", "actual"]:
        found = [c for c in df.columns if (k in c.lower() and "velocity" in c.lower())]
        if not found:
            raise ValueError(f"Cannot find column for {k}_velocity")
        cols[f"{k}_velocity"] = found[0]
    return cols


# ============== 特征工程 ==============
def build_features(df: pd.DataFrame, cols: dict):
    lidar = df[cols["lidar_cols"]].to_numpy(dtype=np.float32)
    global_min = lidar.min(axis=1)
    global_max = lidar.max(axis=1)
    global_mean = lidar.mean(axis=1)
    global_std = lidar.std(axis=1)
    q10 = np.quantile(lidar, 0.10, axis=1)
    q25 = np.quantile(lidar, 0.25, axis=1)
    q50 = np.quantile(lidar, 0.50, axis=1)
    q75 = np.quantile(lidar, 0.75, axis=1)
    q90 = np.quantile(lidar, 0.90, axis=1)

    front = lidar[:, :30]
    front_min = front.min(axis=1)
    left_min = lidar[:, 90:150].min(axis=1)
    right_min = lidar[:, 210:270].min(axis=1)
    lr_asym_min = left_min - right_min
    front_vs_mean = front_min - global_mean

    thr_02 = (lidar < 0.2).sum(axis=1)
    thr_03 = (lidar < 0.3).sum(axis=1)
    thr_05 = (lidar < 0.5).sum(axis=1)

    v_lin = df[cols["linear_velocity"]].to_numpy(dtype=np.float32)
    v_ang = df[cols["angular_velocity"]].to_numpy(dtype=np.float32)
    v_act = df[cols["actual_velocity"]].to_numpy(dtype=np.float32)
    eps = 1e-3
    speed_ratio = (v_act + eps) / (v_lin + eps)
    log_speed_ratio = np.log(speed_ratio)
    speed_absdiff = np.abs(v_act - v_lin)
    ang_over_lin = np.abs(v_ang) / (np.abs(v_lin) + eps)

    n_beams = lidar.shape[1]
    n_sectors = 18
    w = n_beams // n_sectors
    sec_means, sec_mins, sec_stds = [], [], []
    for i in range(n_sectors):
        s, e = i * w, (i + 1) * w
        sector = lidar[:, s:e]
        sec_means.append(sector.mean(axis=1))
        sec_mins.append(sector.min(axis=1))
        sec_stds.append(sector.std(axis=1))
    sec_means = np.stack(sec_means, axis=1)
    sec_mins = np.stack(sec_mins, axis=1)
    sec_stds = np.stack(sec_stds, axis=1)

    feat = {
        "global_min": global_min, "global_max": global_max,
        "global_mean": global_mean, "global_std": global_std,
        "q10": q10, "q25": q25, "q50": q50, "q75": q75, "q90": q90,
        "front_min": front_min, "left_min": left_min, "right_min": right_min,
        "lr_asym_min": lr_asym_min, "front_vs_mean": front_vs_mean,
        "count_lt_0.2": thr_02, "count_lt_0.3": thr_03, "count_lt_0.5": thr_05,
        "linear_velocity": v_lin, "angular_velocity": v_ang, "actual_velocity": v_act,
        "ang_over_lin": ang_over_lin,
        "speed_ratio": speed_ratio, "log_speed_ratio": log_speed_ratio,
        "speed_absdiff": speed_absdiff,
    }

    for i in range(sec_means.shape[1]):
        feat[f"sec{i:02d}_mean"] = sec_means[:, i]
        feat[f"sec{i:02d}_min"] = sec_mins[:, i]
        feat[f"sec{i:02d}_std"] = sec_stds[:, i]

    X = pd.DataFrame(feat).astype(np.float32)
    meta = {
        "feature_names": list(X.columns),
        "n_features": int(X.shape[1]),
        "n_sectors": n_sectors,
        "lidar_cols": cols["lidar_cols"],
    }
    return X, meta


# ============== 主流程 ==============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", default="artifacts_rf_timesplit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--max_features", type=str, default="sqrt")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    cols = detect_columns(df)

    # 前 7000 行有标签；仅取 0/1/2
    df_trainable = df.iloc[:7000].copy()
    df_trainable = df_trainable[df_trainable[cols["label"]].isin([0, 1, 2])].copy()

    # 特征 + 标签
    X_all, meta = build_features(df_trainable, cols)
    y_all = df_trainable[cols["label"]].astype(int).values

    # ==== 🕒 时间顺序切分 ====
    split_idx = int(len(X_all) * 0.8)
    X_tr, X_val = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
    y_tr, y_val = y_all[:split_idx], y_all[split_idx:]
    print(f"[INFO] Time-based split: train={len(X_tr)}, val={len(X_val)}")

    # 类别权重
    classes = np.unique(y_all)
    cw = compute_class_weight("balanced", classes=classes, y=y_all)
    weight_map = {int(c): float(w) for c, w in zip(classes, cw)}
    sample_weight_tr = np.array([weight_map[int(c)] for c in y_tr], dtype=np.float32)

    # RF 模型
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        class_weight=None,
        n_jobs=-1,
        random_state=args.seed,
        oob_score=False,
    )

    # ========== ⏱ 训练计时 ==========
    t0 = time.time()
    rf.fit(X_tr, y_tr, sample_weight=sample_weight_tr)
    train_time = time.time() - t0
    print(f"\nTraining time: {train_time:.3f} seconds")

    # ========== ⏱ 验证推理计时 ==========
    t1 = time.time()
    y_pred = rf.predict(X_val)
    val_time = time.time() - t1
    avg_val_time = val_time / len(X_val)
    print(f"Validation inference time: {val_time:.3f} sec total, {avg_val_time:.6f} sec/sample")

    # 验证结果
    print("\n================= VALIDATION (time split) =================")
    print("Confusion matrix (labels 0,1,2):")
    print(confusion_matrix(y_val, y_pred, labels=[0, 1, 2]))
    print("\nClassification report:")
    print(classification_report(y_val, y_pred, digits=4))

    # 保存模型与元数据
    joblib.dump(rf, out_dir / "rf_speed_timesplit.joblib")
    meta_out = {
        "columns": cols,
        "feature_meta": meta,
        "class_weight": weight_map,
        "rf_params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "random_state": args.seed,
        },
        "split_type": "time_ordered",
        "timing": {"train_sec": train_time, "val_sec_per_sample": avg_val_time},
    }
    with open(out_dir / "feature_meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"\nSaved model to: {out_dir/'rf_speed_timesplit.joblib'}")

    # 特征重要性
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1][:20]
    print("\nTop-20 feature importance:")
    for rank, i in enumerate(idx, 1):
        print(f"{rank:2d}. {X_all.columns[i]:24s} {importances[i]:.5f}")

    # 无标签后 3000 样本推断
    df_test = df.iloc[7000:].copy()
    X_test, _ = build_features(df_test, cols)

    # ========== ⏱ 测试推理计时 ==========
    t2 = time.time()
    proba = rf.predict_proba(X_test)
    pred = rf.predict(X_test)
    test_time = time.time() - t2
    avg_test_time = test_time / len(X_test)
    print(f"Test inference time: {test_time:.3f} sec total, {avg_test_time:.6f} sec/sample")

    out = pd.DataFrame({
        "timestamp": df_test[cols["timestamp"]] if "timestamp" in cols and cols["timestamp"] in df_test.columns else np.arange(len(df_test)),
        "pred_class": pred.astype(int),
        "p0": proba[:, 0],
        "p1": proba[:, 1],
        "p2": proba[:, 2],
    })
    out.to_csv(out_dir / "test_predictions.csv", index=False)
    print(f"\nSaved test predictions to: {out_dir/'test_predictions.csv'}")

    vals, cnts = np.unique(pred, return_counts=True)
    total = cnts.sum()
    print("\n================= TEST (last 3000, unlabeled) =================")
    for v, c in zip(vals, cnts):
        print(f"class {int(v)}: {c:4d} ({100*c/total:6.2f}%)")
    print("===============================================================")

    # ===== Execution Summary (simplified) =====
    process = psutil.Process()
    mem_usage = process.memory_info().rss / (1024 ** 2)  # MB
    mem_per_sample = mem_usage / len(y_val)
    val_acc = accuracy_score(y_val, y_pred)
    model_name = "Random Forest"

    results = {
        "Model": model_name, 
        "ValInferTime_sec_per_sample": f"{avg_val_time * 1:.2e} s",  # 科学计数法，保留 2 位
        "ValAccuracy": f"{val_acc:.4f}",  # 小数点后 4 位
        "MemUsage_MB_per_sample": f"{(mem_usage / len(y_val)):.3f} MB",  # 小数点后 3 位
    }

    # ==== 打印简洁结果 ====
    print("\n===== EXECUTION SUMMARY (Simplified) =====")
    for k, v in results.items():
        print(f"{k:30s}: {v}")
    print("==========================================")

    # 保存汇总结果
    import csv
    summary_file = "results_summary_simple.csv"
    with open(summary_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(results)

if __name__ == "__main__":
    main()