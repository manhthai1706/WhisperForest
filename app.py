"""
WhisperForest — Causal Decision Intelligence Engine
Giao diện web trực quan hoá kiến trúc đa tầng dạng cây.
"""

import os
import io
import base64
import sys
import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask.json.provider import DefaultJSONProvider


class NumpyJSONProvider(DefaultJSONProvider):
    @staticmethod
    def default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

app = Flask(__name__)
app.json = NumpyJSONProvider(app)


# ── Pipeline tree definition ──────────────────────────────────────────
PIPELINE_TREE = {
    "data": {
        "id": "data",
        "label": "Dữ liệu Quan sát",
        "en": "Observational Data",
        "icon": "database",
        "description": "303 bệnh nhân, 14 chỉ số lâm sàng (heart.csv)",
        "row": 0,
        "col": 0,
        "colspan": 2,
        "children": ["predictive", "causal"],
    },
    "predictive": {
        "id": "predictive",
        "label": "Predictive Branch",
        "en": "Predictive Branch",
        "icon": "brain",
        "description": "RandomForest classifier — dự đoán nguy cơ bệnh tim",
        "row": 1,
        "col": 0,
        "children": ["rca"],
    },
    "causal": {
        "id": "causal",
        "label": "Causal Discovery",
        "en": "Causal Discovery",
        "icon": "graph",
        "description": "PC + DirectLiNGAM ensemble — khám phá đồ thị nhân quả DAG",
        "row": 1,
        "col": 1,
        "children": ["rca"],
    },
    "rca": {
        "id": "rca",
        "label": "WhisperTrace RCA",
        "en": "WhisperTrace RCA",
        "icon": "search",
        "description": "Root Cause Analysis — truy vết nguyên nhân từng ca bệnh",
        "row": 2,
        "col": 0,
        "colspan": 2,
        "children": ["trace"],
    },
    "trace": {
        "id": "trace",
        "label": "Per-patient Causal Trace",
        "en": "Per-patient Causal Trace",
        "icon": "file",
        "description": "Hồ sơ causal từng bệnh nhân: RCA scores, CATE, interventions",
        "row": 3,
        "col": 0,
        "colspan": 2,
        "children": ["cate"],
    },
    "cate": {
        "id": "cate",
        "label": "CATE + Intervention Effects",
        "en": "CATE + Intervention Effects",
        "icon": "chart",
        "description": "Ước lượng hiệu quả can thiệp từng biến (CausalForestDML / T-Learner)",
        "row": 4,
        "col": 0,
        "colspan": 2,
        "children": ["memory"],
    },
    "memory": {
        "id": "memory",
        "label": "Causal Experience Bank",
        "en": "Causal Experience Bank",
        "icon": "database",
        "description": "Bộ nhớ dài hạn — lưu causal traces, tra cứu theo causal profile",
        "row": 5,
        "col": 0,
        "colspan": 2,
        "children": ["reasoner"],
    },
    "reasoner": {
        "id": "reasoner",
        "label": "SCM Reasoning Engine",
        "en": "SCM Reasoning Engine",
        "icon": "cogs",
        "description": "Sinh đa giả thuyết → simulate → xếp hạng → policy vote",
        "row": 6,
        "col": 0,
        "colspan": 2,
        "children": ["counterfactual", "policy"],
    },
    "counterfactual": {
        "id": "counterfactual",
        "label": "Counterfactual Simulation",
        "en": "Counterfactual Simulation",
        "icon": "flask",
        "description": "Mô phỏng phản thực tế: Abduction → Action → Propagation (Pearl)",
        "row": 7,
        "col": 0,
        "children": ["auditor"],
    },
    "policy": {
        "id": "policy",
        "label": "Policy Generator",
        "en": "Policy Generator",
        "icon": "book",
        "description": "Đúc kết chính sách từ full reasoning chain (features→RCA→CF→outcome)",
        "row": 7,
        "col": 1,
        "children": ["auditor"],
    },
    "auditor": {
        "id": "auditor",
        "label": "Causal Auditor",
        "en": "Causal Auditor",
        "icon": "shield",
        "description": "Kiểm chứng nhất quán giữa các tầng, phát hiện Simpson's Paradox",
        "row": 8,
        "col": 0,
        "colspan": 2,
        "children": [],
    },
}


def _run_pipeline():
    """Run the full pipeline and return structured results for visualization."""
    from src.whisper_forest import WhisperForest

    csv_path = "heart.csv"
    if not os.path.exists(csv_path):
        return {"error": "heart.csv not found"}

    df = pd.read_csv(csv_path)
    target = "target"
    features = [c for c in df.columns if c != target]

    constraints = {
        "tiers": {
            0: ["age", "sex"],
            1: ["trestbps", "chol", "fbs", "thalach"],
            2: ["cp", "restecg", "exang", "oldpeak", "slope", "ca", "thal"],
            3: [target],
        },
        "whitelist": [
            ("age", target), ("sex", target), ("chol", target),
            ("oldpeak", target), ("ca", target),
        ],
    }

    wf = WhisperForest(data=df, target=target, features=features)
    wf.predictive.fit(model_type="classifier")

    dag = wf.causal.get_dag(
        method="ensemble", constraints=constraints,
        bootstrap_runs=10, confidence_threshold=0.40,
    )

    sick = df[df[target] == 1]
    healthy = df[df[target] == 0]
    anomaly = sick.sort_values(["oldpeak", "ca"], ascending=False).iloc[[0]]
    baseline = healthy.sample(n=min(20, len(healthy)), random_state=42)

    rca_report = wf.rca.analyze_anomaly(
        anomaly_data=anomaly, baseline_data=baseline,
        causal_graph=dag, method="graph",
    )

    wf.rca.fit_scm()

    sick_sample = sick.sample(n=min(5, len(sick)), random_state=0)
    for i in range(len(sick_sample)):
        try:
            wf.counterfactual_trace(
                patient_df=sick_sample.iloc[[i]].copy(),
                outcome=0, k_neighbors=5, top_k_causes=3,
            )
        except Exception:
            pass

    ranked, trace, recommendations = wf.reason(
        rca_report=rca_report,
        patient=anomaly.reset_index(drop=True),
        cate_series=pd.Series(0.0, index=features),
        n_plans=6,
    )

    audit_report = wf.audit_consistency(
        patient=anomaly.reset_index(drop=True),
        threshold_conflict=0.05,
        rca_report=rca_report,
    )

    # Build structured result
    edges_list = [{"source": u, "target": v} for u, v in dag]

    rca_top = rca_report.sort_values("Attribution_Mean", ascending=False).head(5)
    rca_items = [
        {"feature": idx, "attribution": round(row["Attribution_Mean"], 4)}
        for idx, row in rca_top.iterrows()
    ]

    hypotheses_list = []
    for i, (hyp, rec) in enumerate(ranked, 1):
        hypotheses_list.append({
            "rank": i,
            "source": hyp.source,
            "plan": {k: round(v, 1) for k, v in rec.plan.items()},
            "delta": round(rec.delta, 4),
            "success": rec.success,
            "rationale": hyp.rationale,
        })

    recs_list = []
    for plan, score in recommendations:
        recs_list.append({
            "plan": {k: round(v, 1) for k, v in plan.items()},
            "score": round(score, 3),
        })

    return {
        "dataset": {"rows": df.shape[0], "columns": df.shape[1]},
        "predictive": {
            "accuracy": None,
            "auc": None,
            "model": "RandomForestClassifier",
        },
        "dag": {
            "edges": len(dag),
            "edge_list": edges_list[:15],
            "confidence": [
                {"edge": f"{u} → {v}", "score": 1.0}
                for u, v in dag[:5]
            ],
        },
        "rca": {"top_causes": rca_items},
        "trace": {
            "features": anomaly.iloc[0].to_dict(),
            "top_causes": [f for f in trace.top_causes(k=5)],
            "interventions_count": len(trace.interventions_tried),
        },
        "memory": {"trace_count": len(wf.memory)},
        "reasoner": {
            "hypotheses_count": len(hypotheses_list),
            "hypotheses": hypotheses_list,
        },
        "counterfactual": {
            "original_outcome": float(anomaly[target].iloc[0]),
            "best_delta": hypotheses_list[0]["delta"] if hypotheses_list else None,
            "best_plan": hypotheses_list[0]["plan"] if hypotheses_list else {},
        },
        "policy": {"recommendations": recs_list},
        "auditor": {
            "consistency_score": round(audit_report["Consistency_Score"], 4),
            "safety": audit_report["Safety_Status"],
            "warnings": audit_report.get("Warnings", []),
        },
    }


# ── Routes ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/tree")
def get_tree():
    return jsonify(PIPELINE_TREE)


@app.route("/api/target")
def get_target():
    """Return the target feature info and feature names for the form."""
    import pandas as pd
    csv_path = "heart.csv"
    if not os.path.exists(csv_path):
        return jsonify({"error": "heart.csv not found"})
    df = pd.read_csv(csv_path)
    target = "target"
    features = [c for c in df.columns if c != target]
    sample = df.iloc[0].to_dict()
    info = {
        "target": target,
        "features": features,
        "sample": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in sample.items()},
    }
    return jsonify(info)


@app.route("/api/analyze", methods=["POST"])
def analyze_patient():
    """
    Accept patient data, run pipeline, return per-node results.
    Body: { "patient": { "age": 63, "sex": 1, ... } }
    """
    from src.whisper_forest import WhisperForest
    import pandas as pd

    try:
        body = request.get_json(force=True)
        patient_raw = body.get("patient", {})
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    csv_path = "heart.csv"
    if not os.path.exists(csv_path):
        return jsonify({"error": "heart.csv not found"}), 500

    df = pd.read_csv(csv_path)
    target = "target"
    features = [c for c in df.columns if c != target]

    constraints = {
        "tiers": {
            0: ["age", "sex"],
            1: ["trestbps", "chol", "fbs", "thalach"],
            2: ["cp", "restecg", "exang", "oldpeak", "slope", "ca", "thal"],
            3: [target],
        },
        "whitelist": [
            ("age", target), ("sex", target), ("chol", target),
            ("oldpeak", target), ("ca", target),
        ],
    }

    patient_df = pd.DataFrame([patient_raw])[features + [target]]
    wf = WhisperForest(data=df, target=target, features=features)
    wf.predictive.fit(model_type="classifier")

    # ── Step 1: Data → Predictive ──
    pred_risk = float(wf.predictive.predict_proba(patient_df[features])[0, 1])
    pred_class = int(wf.predictive.predict(patient_df[features])[0])

    # ── Step 2: Data → Causal Discovery ──
    dag = wf.causal.get_dag(
        method="ensemble", constraints=constraints,
        bootstrap_runs=10, confidence_threshold=0.40,
    )

    # ── Step 3: RCA (World DAG — không cần SCM) ──
    healthy = df[df[target] == 0]
    baseline = healthy.sample(n=min(20, len(healthy)), random_state=42)
    rca_report = wf.rca.analyze_anomaly(
        anomaly_data=patient_df, baseline_data=baseline,
        causal_graph=dag, method="graph",
    )
    rca_top = rca_report.sort_values("Attribution_Mean", ascending=False).head(5)
    rca_items = [
        {"feature": idx, "attribution": round(row["Attribution_Mean"], 4)}
        for idx, row in rca_top.iterrows()
    ]

    # ── Step 4: Causal Trace ──
    trace = wf.counterfactual_trace(
        patient_df=patient_df, outcome=0, k_neighbors=5, top_k_causes=3,
    )

    # ── Step 5: Reasoner (multi-hypothesis) ──
    ranked, updated_trace, recommendations = wf.reason(
        rca_report=rca_report,
        patient=patient_df.reset_index(drop=True),
        cate_series=pd.Series(0.0, index=features),
        n_plans=6,
    )

    hypotheses_list = []
    for i, (hyp, rec) in enumerate(ranked, 1):
        hypotheses_list.append({
            "rank": i,
            "source": hyp.source,
            "plan": {k: round(v, 1) for k, v in rec.plan.items()},
            "delta": round(rec.delta, 4),
            "success": rec.success,
            "rationale": hyp.rationale,
        })

    recs_list = []
    for plan, score in recommendations:
        recs_list.append({
            "plan": {k: round(v, 1) for k, v in plan.items()},
            "score": round(score, 3),
        })

    # ── Step 6: Auditor ──
    audit_report = wf.audit_consistency(
        patient=patient_df.reset_index(drop=True),
        threshold_conflict=0.05,
        rca_report=rca_report,
    )

    full_result = {
        "patient": {
            "features": {k: round(float(v), 2) for k, v in patient_raw.items() if k != target},
            "actual_outcome": int(patient_raw.get(target, -1)),
        },
        "predictive": {
            "risk_score": pred_risk,
            "predicted_class": pred_class,
        },
        "dag": {
            "edges": len(dag),
            "edge_list": [f"{u} → {v}" for u, v in sorted(dag)[:12]],
        },
        "rca": {
            "top_causes": rca_items,
        },
        "trace": {
            "top_causes": trace.top_causes(k=5) if hasattr(trace, "top_causes") else [],
        },
        "memory": {
            "trace_count": len(wf.memory),
        },
        "reasoner": {
            "hypotheses": hypotheses_list,
        },
        "counterfactual": {
            "best_delta": hypotheses_list[0]["delta"] if hypotheses_list else None,
            "best_plan": hypotheses_list[0]["plan"] if hypotheses_list else {},
        },
        "policy": {
            "recommendations": recs_list,
        },
        "auditor": {
            "consistency_score": round(audit_report.get("Consistency_Score", 0), 4),
            "safety": audit_report.get("Safety_Status", "UNKNOWN"),
            "warnings": audit_report.get("Warnings", []),
        },
    }

    return jsonify(full_result)


@app.route("/api/run")
def run_pipeline():
    try:
        result = _run_pipeline()
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
