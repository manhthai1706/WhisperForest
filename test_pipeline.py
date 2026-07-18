"""
test_pipeline.py — Đánh giá toàn bộ luồng 4 nhánh của WhisperForest
Sử dụng dữ liệu heart.csv thực tế, không inject biến can thiệp nhân tạo.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from src.whisper_forest import WhisperForest

SEPARATOR = "=" * 70

def run():
    print(SEPARATOR)
    print(" WhisperForest — 4-Branch Pipeline Evaluation".center(70))
    print(SEPARATOR)

    # ── Chuẩn bị dữ liệu ─────────────────────────────────────────────────
    csv_path = "heart.csv"
    if not os.path.exists(csv_path):
        print(f"[ERROR] {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"\nDataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Target distribution:\n{df['target'].value_counts().to_string()}")

    target   = "target"
    features = [c for c in df.columns if c != target]

    # Ràng buộc tiên nghiệm (domain knowledge)
    # age/sex không thể bị tác động bởi bất kỳ biến nào trong mô hình
    constraints = {
        "tiers": {
            0: ["age", "sex"],
            1: ["trestbps", "chol", "fbs", "thalach"],
            2: ["cp", "restecg", "exang", "oldpeak", "slope", "ca", "thal"],
            3: [target]
        },
        "whitelist": [
            ("age",     target),
            ("sex",     target),
            ("chol",    target),
            ("oldpeak", target),
            ("ca",      target),
        ]
    }

    wf = WhisperForest(data=df, target=target, features=features)

    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEPARATOR}")
    print(" NHÁNH 1 — Predictive: Học dự đoán / phân loại".center(70))
    print(SEPARATOR)

    # Train / test split để đánh giá khách quan
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target])
    wf_eval = WhisperForest(data=df_train, target=target, features=features)

    wf_eval.predictive.fit(model_type="classifier")

    y_true = df_test[target].values
    y_pred = wf_eval.predictive.predict(df_test[features])
    y_prob = wf_eval.predictive.predict_proba(df_test[features])[:, 1]

    acc  = accuracy_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)

    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  ROC-AUC   : {auc:.4f}")

    # Tiếp tục fit toàn bộ dữ liệu cho các nhánh sau
    wf.predictive.fit(model_type="classifier")

    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEPARATOR}")
    print(" NHÁNH 2 — Causal Discovery + RCA".center(70))
    print(SEPARATOR)

    print("\n[2a] Khám phá đồ thị nhân quả (Stability Selection, 10 runs)...")
    dag = wf.causal.get_dag(
        method="ensemble",
        constraints=constraints,
        bootstrap_runs=10,
        confidence_threshold=0.40
    )
    print(f"     DAG edges discovered: {len(dag)}")
    for u, v in sorted(dag)[:8]:
        print(f"       {u} → {v}")
    if len(dag) > 8:
        print(f"       ... ({len(dag) - 8} more)")

    print("\n[2b] Root Cause Analysis trên một ca bất thường...")
    wf.rca.fit_scm()

    sick_cohort    = df[df[target] == 1]
    healthy_cohort = df[df[target] == 0]

    # Chọn ca bất thường: bệnh nặng nhất (oldpeak cao, ca nhiều)
    anomaly = sick_cohort.sort_values(["oldpeak", "ca"], ascending=False).iloc[[0]]
    baseline = healthy_cohort.sample(n=min(20, len(healthy_cohort)), random_state=42)

    print(f"     Anomaly patient: oldpeak={anomaly['oldpeak'].iloc[0]}, "
          f"ca={anomaly['ca'].iloc[0]}, chol={anomaly['chol'].iloc[0]}")

    rca_report = wf.rca.analyze_anomaly(
        anomaly_data=anomaly,
        baseline_data=baseline,
        causal_graph=dag,
        method="intervention"
    )

    print("\n     Top contributing root causes:")
    top5 = rca_report.sort_values("Attribution_Mean", ascending=False).head(5)
    for _, row in top5.iterrows():
        bar = "█" * int(abs(row["Attribution_Mean"]) * 2)
        print(f"       {row.name:<12} {row['Attribution_Mean']:+.4f}  {bar}")

    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEPARATOR}")
    print(" NHÁNH 3 — SCM / MoSCM + Counterfactual + Policy".center(70))
    print(SEPARATOR)

    print("\n[3a] Counterfactual: Trạng thái phản thực tế của ca bất thường")
    print("     (Nếu bệnh nhân này ở trạng thái không bệnh, các chỉ số sẽ thế nào?)\n")

    cf_table = wf.counterfactual(
        patient_df=anomaly.reset_index(drop=True),
        outcome=0,
        k_neighbors=5,
        top_k_causes=3
    )

    print(f"     {'Feature':<14} {'Original':>10} {'Counter.':>10} {'Delta':>10}  Intervened")
    print("     " + "-" * 52)
    for feat, row in cf_table.iterrows():
        marker = "  ◄ intervened" if row["Intervened"] else ""
        print(f"     {feat:<14} {row['Original']:>10.3f} {row['Counterfactual']:>10.3f} "
              f"{row['Delta']:>+10.3f}{marker}")

    print("\n[3b] EM-MoSCM: Học phân cụm cơ chế nhân quả (3 experts)...")
    wf.rca.fit_mixture_scm(n_clusters=3, n_iterations=5)

    routing = wf.rca.mixture_engine.predict_proba_membership(
        anomaly.reset_index(drop=True)[features]
    )[0]
    print("     Dynamic routing weights (P(Mechanism | Patient)):")
    for k, w in enumerate(routing):
        bar = "▓" * int(w * 30)
        print(f"       Expert {k}: {w:5.1%}  {bar}")

    print("\n[3c] Policy: Ghi nhận thử nghiệm và đúc kết chính sách...")
    print(f"     Experiments logged: {len(wf.policy.experiment_log)}")

    # Chạy counterfactual trên nhiều ca bệnh để làm giàu bộ nhớ policy
    sick_sample = sick_cohort.sample(n=min(10, len(sick_cohort)), random_state=0).reset_index(drop=True)
    for i in range(len(sick_sample)):
        patient_row = sick_sample.iloc[[i]].copy()
        try:
            wf.counterfactual(patient_row, outcome=0, k_neighbors=5, top_k_causes=3)
        except Exception:
            pass

    print(f"     Experiments logged after batch: {len(wf.policy.experiment_log)}")

    policy_tree = wf.policy.distill_policy_from_experiments(max_depth=3)
    if policy_tree is not None:
        importances = pd.Series(
            policy_tree.feature_importances_, index=features
        ).sort_values(ascending=False)
        print("\n     Most decisive features for intervention success (Policy Tree):")
        for feat, imp in importances[importances > 0.01].head(5).items():
            bar = "█" * int(imp * 40)
            print(f"       {feat:<14} {imp:.4f}  {bar}")

    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEPARATOR}")
    print(" NHÁNH 4 — Causal Auditor: Kiểm chứng & biểu quyết".center(70))
    print(SEPARATOR)

    audit_report = wf.audit_consistency(
        patient=anomaly.reset_index(drop=True),
        threshold_conflict=0.05
    )
    print()
    print(wf.causal.auditor.format_report(audit_report))

    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEPARATOR}")
    print(" Tổng kết".center(70))
    print(SEPARATOR)
    print(f"  Nhánh 1 — Predictive  : Accuracy={acc:.3f}, AUC={auc:.3f}")
    print(f"  Nhánh 2 — DAG edges   : {len(dag)}")
    print(f"  Nhánh 2 — Top cause   : {top5.index[0]} (attr={top5['Attribution_Mean'].iloc[0]:+.4f})")
    print(f"  Nhánh 3 — CF delta    : target Δ={cf_table.loc[target, 'Delta']:+.4f}" if target in cf_table.index else "")
    print(f"  Nhánh 3 — Policy log  : {len(wf.policy.experiment_log)} experiments")
    print(f"  Nhánh 4 — Safety      : {audit_report['Safety_Status']}")
    print(f"\n{SEPARATOR}")
    print(" Pipeline completed successfully.".center(70))
    print(SEPARATOR)

if __name__ == "__main__":
    run()
