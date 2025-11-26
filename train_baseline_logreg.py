
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

# 对每个扇区做 mean/min/std, 得到 18*3=54 个“空间分布”特征
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

    # Global stats
    global_min = np.min(lidar, axis=1)
    global_max = np.max(lidar, axis=1)
    global_mean = np.mean(lidar, axis=1)
    global_std = np.std(lidar, axis=1)
    q10 = np.quantile(lidar, 0.10, axis=1)
    q25 = np.quantile(lidar, 0.25, axis=1)
    q50 = np.quantile(lidar, 0.50, axis=1)
    q75 = np.quantile(lidar, 0.75, axis=1)
    q90 = np.quantile(lidar, 0.90, axis=1)

    # Heuristic windows(方向窗口)
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

    # 方向派生特征（不对称与前方相对危险度）
    lr_asym_min = left_min - right_min
    front_vs_mean = front_min - global_mean

    # 统计每帧里，比阈值还近的雷达点有多少
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

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--out_dir", type=str, default="artifacts_logreg")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 创建输出目录
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读 CSV + 自动识别关键列
    df = pd.read_csv(args.csv)
    cols = detect_columns(df)

    if cols["label"] is None:
        raise ValueError("Cannot find 'status' column.")
    # normalize labels: -1 -> 3 (unknown)
    df.loc[df[cols["label"]] == -1, cols["label"]] = 3

    # supervised train on first 7000 with labels in {0,1,2}
    df_trainable = df.iloc[:7000].copy()
    df_trainable = df_trainable[df_trainable[cols["label"]].isin([0,1,2])].copy()
    # 特征工程
    X_all, meta = build_features(df_trainable, cols)
    y_all = df_trainable[cols["label"]].astype(int).values

    # 训练/验证划分（分层
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=args.seed, stratify=y_all
    )

    # 记录与复现：以后你换模型（比如 XGBoost/SVM/RandomForest）时，可以用同样的权重作为 sample_weight，保证对比公平
    classes = np.unique(y_all)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_all)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}

    # 建立训练管线：标准化 + 多项逻辑回归
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

    # 混淆矩阵是一张表，表示模型在每个真实类别上预测对了多少、错了多少
    # 指标	               含义	                          计算方式	              通俗解释
    # precision（精确率）	预测为该类的样本中有多少是真的	   TP / (TP + FP)	     “我说是类1的，有多少真的是类1？”
    # recall（召回率）	    实际属于该类的样本中有多少被找出来	TP / (TP + FN)	      “所有类1样本中我找出了多少？”
    # f1-score（F1分数）    精确率和召回率的调和平均	      2 × P × R / (P + R)	“兼顾精确和召回的平衡指标”
    # support	          每个类别在验证集中的样本数	     —	                   用于参考，样本多少
    # accuracy（准确率）	全部预测正确的比例	             (TP总和) / 总样本	      整体正确率
    # macro avg	          各类别指标的简单平均	            —	                  各类一视同仁，适合不均衡任务
    # weighted avg	      各类指标按样本数加权平均	         —	                   多数类影响更大
    # ===============================
    # 验证集推理 + 计时
    # ===============================
    t1 = time.time()
    y_pred = pipe.predict(X_val)
    val_time = time.time() - t1
    avg_val_time = val_time / len(X_val)
    print(f"Validation inference time: {val_time:.3f} sec total, {avg_val_time:.6f} sec/sample")

    print("\nConfusion matrix (val) [labels 0,1,2]:")
    print(confusion_matrix(y_val, y_pred, labels=[0, 1, 2]))
    print("\nClassification report (val):")
    print(classification_report(y_val, y_pred, digits=4))

    # ===============================
    # 保存模型与元数据
    # ===============================
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

    # ===============================
    # 测试集预测 + 计时
    # ===============================
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

    # 打印测试集类别分布
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
    model_name = "Logistic Regression"

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
