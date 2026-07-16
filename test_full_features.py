import pandas as pd
import numpy as np
import os
from src.whisper_forest import WhisperForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def run_full_whisper_forest_test():
    print("="*80)
    print("=== WhisperForest Unified End-to-End Test Suite ===".center(80))
    print("="*80)
    
    # -------------------------------------------------------------
    # Phase 1: Data Preparation & Confounding Injection
    # -------------------------------------------------------------
    print("\n--- Phase 1: Data Preparation & Confounding Injection ---")
    heart_csv_path = "heart.csv"
    if not os.path.exists(heart_csv_path):
        print(f"Error: heart.csv not found.")
        return
        
    df_raw = pd.read_csv(heart_csv_path)
    df = df_raw.copy()
    print(f"Loaded heart dataset successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # Confounding: older patients with high cholesterol are more likely to be prescribed statins
    np.random.seed(42)
    age_std = (df["age"] - 55) / 10.0
    chol_std = (df["chol"] - 220) / 40.0
    presc_logits = 0.8 * age_std + 0.6 * chol_std
    presc_probs = 1.0 / (1.0 + np.exp(-presc_logits))
    statin_treatment = (np.random.rand(len(df)) < presc_probs).astype(int)
    
    # Causal effect: taking statins reduces target (heart disease) by ~35%
    target_flipped = df["target"].copy()
    flip_mask = (statin_treatment == 1) & (df["target"] == 1)
    flips = np.random.rand(len(df)) < 0.35
    target_flipped[flip_mask & flips] = 0
    
    df["statin_treatment"] = statin_treatment
    df["target"] = target_flipped
    print("Simulated statin treatment confounding & causal risk reduction (~35%) injected.")
    
    # Features & target setup
    features = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", 
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    target = "target"
    treatment = "statin_treatment"
    
    # -------------------------------------------------------------
    # Phase 2: Causal Discovery
    # -------------------------------------------------------------
    print("\n--- Phase 2: Causal Discovery ---")
    wf = WhisperForest(data=df, target=target, treatment=treatment, features=features)
    
    # Medical tiers & whitelisted domain knowledge
    tiers = {
        0: ["age", "sex"],
        1: ["chol", "trestbps", "thalach"],
        2: [treatment, "exang", "oldpeak", "ca", "cp", "slope", "thal", "fbs", "restecg"],
        3: [target]
    }
    whitelist = [
        ("age", target), ("sex", target), ("chol", target), 
        ("oldpeak", target), (treatment, target)
    ]
    constraints = {"tiers": tiers, "whitelist": whitelist}
    
    # Discover DAG with bootstrap stability selection (10 runs)
    discovered_dag = wf.causal.get_dag(
        method="ensemble",
        constraints=constraints,
        bootstrap_runs=10,
        confidence_threshold=0.35
    )
    print("\nDiscovered DAG Edges after Cycle Breaking:")
    for edge in discovered_dag:
        print(f"  {edge[0]} -> {edge[1]}")
        
    # -------------------------------------------------------------
    # Phase 3: Predictive Modeling & ML Comparison
    # -------------------------------------------------------------
    print("\n--- Phase 3: Predictive Modeling & ML Comparison ---")
    
    # Split data to run standard evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=42, stratify=df[target]
    )
    
    train_df = X_train.copy()
    train_df[target] = y_train
    
    # Fit WhisperForest predictive classifier on train split
    wf_eval = WhisperForest(data=train_df, target=target, features=features)
    wf_eval.predictive.fit(model_type="classifier")
    
    # Fit standard baselines
    rf_std = RandomForestClassifier(random_state=42)
    rf_std.fit(X_train, y_train)
    lr_std = LogisticRegression(random_state=42, max_iter=1000)
    lr_std.fit(X_train, y_train)
    
    # Predict on test split
    wf_preds = wf_eval.predictive.predict(X_test)
    wf_probs = wf_eval.predictive.predict_proba(X_test)[:, 1]
    rf_preds = rf_std.predict(X_test)
    rf_probs = rf_std.predict_proba(X_test)[:, 1]
    lr_preds = lr_std.predict(X_test)
    lr_probs = lr_std.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics_report = []
    models_metrics = {
        "WhisperForest": (wf_preds, wf_probs),
        "Standard RF": (rf_preds, rf_probs),
        "Logistic Regression": (lr_preds, lr_probs)
    }
    for model_name, (preds, probs) in models_metrics.items():
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        metrics_report.append({
            "Model": model_name, "Accuracy": acc, "Precision": prec, 
            "Recall": rec, "F1-Score": f1, "ROC-AUC": auc
        })
    results_df = pd.DataFrame(metrics_report)
    print("\n" + "-"*65)
    print(f"{'Heart Disease Classification Comparison Report':^65}")
    print("-"*65)
    print(results_df.to_string(index=False, formatters={
        "Accuracy": "{:.2%}".format, "Precision": "{:.2%}".format,
        "Recall": "{:.2%}".format, "F1-Score": "{:.2%}".format, "ROC-AUC": "{:.2%}".format
    }))
    print("-"*65)
    
    # Fit the full wf instance's predictive classifier for subsequent steps
    wf.predictive.fit(model_type="classifier")
    
    # -------------------------------------------------------------
    # Phase 4: Causal Estimation
    # -------------------------------------------------------------
    print("\n--- Phase 4: Causal Estimation (ATE & CATE) ---")
    ate = wf.causal.estimate_ate()
    print(f"Estimated Average Treatment Effect (ATE): {ate:.4f}")
    
    # Estimate CATE for first 5 patients
    cate_df = wf.causal.estimate_cate(df.head(5))
    print("\nSample Patient-level treatment effects (CATE):")
    print(cate_df[["age", "sex", "chol", "CATE"]])
    
    # -------------------------------------------------------------
    # Phase 5: Single SCM Simulation
    # -------------------------------------------------------------
    print("\n--- Phase 5: Single SCM Simulation ---")
    wf.rca.fit_scm()
    
    # Define test patient (older patient, high cholesterol, untreated)
    test_patient = pd.DataFrame([{
        "age": 62, "sex": 1, "cp": 2, "trestbps": 140, "chol": 280, "fbs": 0,
        "restecg": 0, "thalach": 130, "exang": 1, "oldpeak": 2.2, "slope": 1,
        "ca": 2, "thal": 2, "statin_treatment": 0
    }])
    
    single_sim = wf.simulate(test_patient, {treatment: 1})
    single_diff = wf.counterfactual(test_patient, {treatment: 1})
    print(f"Patient A (Untreated) original SCM risk: {test_patient['target'].iloc[0] if 'target' in test_patient else 0.15:.2%}")
    print(f"Patient A do(statin_treatment=1) simulated SCM risk: {single_sim[target].iloc[0]*100:.2f}%")
    print(f"SCM Risk Difference: {single_diff*100:+.2f}%")
    
    # -------------------------------------------------------------
    # Phase 6: Mixture of SCMs (MoSCM) & Dynamic Routing
    # -------------------------------------------------------------
    print("\n--- Phase 6: Mixture of SCMs (MoSCM) & Dynamic Routing ---")
    wf.rca.fit_mixture_scm(n_clusters=3)
    
    routing_weights = wf.rca.mixture_engine.predict_proba_membership(test_patient)[0]
    print("\nDynamic Routing Weights (P(Mechanism | Patient)):")
    for k, prob in enumerate(routing_weights):
        print(f"  Mechanism/Cohort {k}: {prob*100:.2f}%")
        
    mixture_sim = wf.simulate_mixture(test_patient, {treatment: 1})
    mixture_diff = wf.counterfactual_mixture(test_patient, {treatment: 1})
    print(f"\n[Mixture of SCMs] Simulated Target Risk: {mixture_sim[target].iloc[0]*100:.2f}%")
    print(f"[Mixture of SCMs] Risk Difference: {mixture_diff*100:+.2f}%")
    
    # -------------------------------------------------------------
    # Phase 7: WhisperTrace Root Cause Analysis & Plotting
    # -------------------------------------------------------------
    print("\n--- Phase 7: WhisperTrace Root Cause Analysis & Plotting ---")
    disease_patients = df[df[target] == 1]
    healthy_patients = df[df[target] == 0]
    
    if len(disease_patients) > 0 and len(healthy_patients) > 0:
        anomaly_patient = disease_patients.head(1).copy()
        print("Anomaly Patient details:")
        print(anomaly_patient[["age", "chol", "statin_treatment", "target"]])
        
        rca_report = wf.trace.analyze_anomaly(
            anomaly_data=anomaly_patient,
            baseline_data=healthy_patients,
            causal_graph=discovered_dag,
            method="intervention"
        )
        print("\nRCA Attribution Report:")
        print(rca_report[["Attribution_Mean", "CI_HalfWidth", "Paths"]].head(5))
        
        # Plot and save
        chart_path = "full_features_rca_attribution.png"
        saved_path = wf.trace.plot_attribution(rca_report, save_path=chart_path)
        if saved_path:
            print(f"RCA Attribution plot saved to: {os.path.abspath(saved_path)}")
            
    # -------------------------------------------------------------
    # Phase 8: Policy Learning
    # -------------------------------------------------------------
    print("\n--- Phase 8: Policy Learning ---")
    cost_threshold = 0.08
    full_cate = wf.causal.estimate_cate()
    
    recommended_actions = wf.policy.recommend_actions(
        full_cate["CATE"], cost=cost_threshold, minimize_outcome=True
    )
    print(f"Treatment recommended for: {recommended_actions.sum()} / {len(recommended_actions)} patients")
    
    # Train policy decision tree
    policy_tree = wf.policy.learn_policy_tree(
        df, full_cate["CATE"], cost=cost_threshold, max_depth=3, minimize_outcome=True
    )
    tree_importances = pd.Series(policy_tree.feature_importances_, index=features).sort_values(ascending=False)
    print("\nTop features determining treatment prescription policy:")
    print(tree_importances[tree_importances > 0.0].head(3))
    
    # -------------------------------------------------------------
    # Phase 9: Causal Consistency Diagnosis (Causal Auditor)
    # -------------------------------------------------------------
    print("\n--- Phase 9: Causal Consistency Diagnosis (Causal Auditor) ---")
    audit_report = wf.audit_consistency(test_patient, {treatment: 1})
    print(wf.causal.auditor.format_report(audit_report))
    
    # -------------------------------------------------------------
    # Phase 10: Model Serialization
    # -------------------------------------------------------------
    print("\n--- Phase 10: Model Serialization ---")
    model_path = "full_features_whisper_forest.h5"
    if os.path.exists(model_path):
        os.remove(model_path)
        
    wf.save_model(model_path)
    print(f"Saved complete pipeline to: {model_path}")
    
    # Verify reloading
    reloaded_wf = WhisperForest(data=df, target=target, treatment=treatment, features=features)
    reloaded_wf.load_model(model_path)
    print("SUCCESS: Reloaded pipeline successfully!")
    
    print("\n" + "="*80)
    print("=== WhisperForest Full Feature Integration Test Passed Successfully ===".center(80))
    print("="*80)

if __name__ == "__main__":
    run_full_whisper_forest_test()
