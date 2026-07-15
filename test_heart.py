import pandas as pd
import numpy as np
import os
from src.whisper_forest import WhisperForest

def run_heart_analysis():
    print("=== WhisperForest Heart Disease Causal Analysis (v4) ===")
    
    # 1. Load the heart dataset
    heart_csv_path = r"C:\Users\manht\Downloads\heart.csv"
    if not os.path.exists(heart_csv_path):
        print(f"Error: Heart dataset not found at {heart_csv_path}")
        return
        
    df_raw = pd.read_csv(heart_csv_path)
    df = df_raw.copy()
    print(f"Loaded heart dataset successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # 2. Simulate a realistic medical treatment: statin_treatment (Direction 3)
    # Confounding: older patients with high cholesterol are more likely to be prescribed statins
    np.random.seed(42)
    age_std = (df["age"] - 55) / 10.0
    chol_std = (df["chol"] - 220) / 40.0
    presc_logits = 0.8 * age_std + 0.6 * chol_std
    presc_probs = 1.0 / (1.0 + np.exp(-presc_logits))
    statin_treatment = (np.random.rand(len(df)) < presc_probs).astype(int)
    
    # Inject causal effect: taking statins reduces the probability of heart disease (target) by ~35%
    # If statin_treatment == 1 and target == 1, we randomly flip some targets to 0 with 35% probability
    target_flipped = df["target"].copy()
    flip_mask = (statin_treatment == 1) & (df["target"] == 1)
    flips = np.random.rand(len(df)) < 0.35
    target_flipped[flip_mask & flips] = 0
    
    df["statin_treatment"] = statin_treatment
    df["target"] = target_flipped
    
    print("Injected simulated actionable treatment: 'statin_treatment' (with confounding and causal effect).")
    print(f"Prescribed statins: {statin_treatment.sum()} / {len(statin_treatment)} patients.")
    
    # 3. Initialize WhisperForest
    features = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", 
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    
    wf = WhisperForest(
        data=df,
        target="target",
        treatment="statin_treatment",
        features=features
    )
    
    # 4. Branch 2: Traditional Predictive Model (Classifier)
    print("\n--- Branch 2: Training Predictive Classifier ---")
    wf.predictive.fit(model_type="classifier")
    
    prob_preds = wf.predictive.predict_proba(df.head(5))
    print("Predicted probabilities of heart disease (Healthy vs Disease) for first 5 patients:")
    print(prob_preds)
    
    importances = wf.predictive.explain(method="mdi")
    print("\nFeature Importances for Predicting Heart Disease:")
    print(importances.head(5))
    
    # 5. Branch 1: Causal Graph and Effect Estimation (Stability Selection & Prior Knowledge - Directions 1 & 2)
    print("\n--- Branch 1: Causal Graph and Effect Estimation ---")
    
    # Medical-grounded temporal tiers
    tiers = {
        0: ["age", "sex"],
        1: ["chol", "trestbps", "thalach"],
        2: ["statin_treatment", "exang", "oldpeak", "ca", "cp", "slope", "thal", "fbs", "restecg"],
        3: ["target"]
    }
    
    # Whitelist established clinical edges as domain-expert prior knowledge (Expert Validation)
    whitelist = [
        ("age", "target"),
        ("sex", "target"),
        ("chol", "target"),
        ("oldpeak", "target"),
        ("statin_treatment", "target")
    ]
    
    constraints = {
        "tiers": tiers,
        "whitelist": whitelist
    }
    
    # Run DAG discovery with stability selection (25 runs, threshold 0.35)
    discovered_dag = wf.causal.get_dag(
        method="ensemble",
        constraints=constraints, 
        bootstrap_runs=25, 
        confidence_threshold=0.35
    )
    
    print("\nFinal Discovered Causal Graph (DAG) Edges after Cycle Breaking:")
    for edge in discovered_dag:
        print(f"  {edge[0]} -> {edge[1]}")
        
    # Estimate ATE of Statin treatment on heart disease risk
    ate = wf.causal.estimate_ate()
    print(f"\nEstimated ATE of Statin Treatment on Heart Disease Risk: {ate:.4f}")
    print("Interpretation: Prescribing statins causes an average risk change of heart disease by this amount.")
    
    # Estimate CATE for each patient
    cate_df = wf.causal.estimate_cate()
    print("\nSample patient-level treatment effects (CATE):")
    print(cate_df[["age", "sex", "chol", "CATE"]].head())
    
    # 6. Branch 4: Policy Learning
    print("\n--- Branch 4: Policy Learning ---")
    # Benefit = risk reduction (-CATE). Cost threshold of statins = 0.08
    rec_actions = wf.policy.recommend_actions(cate_df["CATE"], cost=0.08, minimize_outcome=True)
    print(f"Patients recommended for statin treatment: {rec_actions.sum()} / {len(rec_actions)}")
    
    # Learn policy tree
    tree = wf.policy.learn_policy_tree(df, cate_df["CATE"], cost=0.08, max_depth=3, minimize_outcome=True)
    tree_importances = pd.Series(tree.feature_importances_, index=features).sort_values(ascending=False)
    print("Top features determining Statin prescription recommendation policy:")
    print(tree_importances[tree_importances > 0.0].head(3))
    
    # 7. Branch 3: WhisperTrace Causal RCA on a Heart Patient (Cohort-based v2)
    print("\n--- Branch 3: WhisperTrace Causal RCA ---")
    disease_patients = df[df["target"] == 1]
    healthy_patients = df[df["target"] == 0]
    
    if len(disease_patients) > 0 and len(healthy_patients) > 0:
        anomaly_patient = disease_patients.head(1)
        
        print("Anomaly Patient (Heart Disease):")
        print(anomaly_patient[['age', 'sex', 'chol', 'thalach', 'statin_treatment', 'target']])
        
        print("\nRunning RCA using WhisperTrace v2 (Cohort-Based Interventional Counterfactual)...")
        rca_report = wf.trace.analyze_anomaly(
            anomaly_data=anomaly_patient,
            baseline_data=healthy_patients,
            causal_graph=discovered_dag,
            method="intervention"
        )
        
        print("\nRCA Attribution Report (with Confidence Intervals & Causal Paths):")
        print(rca_report)
        
        # Save visualization plot
        chart_path = "heart_rca_attribution.png"
        saved_path = wf.trace.plot_attribution(rca_report, save_path=chart_path)
        if saved_path:
            print(f"\nRCA Attribution plot saved to: {os.path.abspath(saved_path)}")
            
        # 8. Export Model to H5
        print("\n--- Saving WhisperForest model to H5 ---")
        h5_model_path = "whisper_forest_model.h5"
        wf.save_model(h5_model_path)
        
        # Verify loading the model back
        print("\n--- Verifying H5 Loading ---")
        new_wf = WhisperForest(data=df, target="target", treatment="statin_treatment", features=features)
        new_wf.load_model(h5_model_path)

            
if __name__ == "__main__":
    run_heart_analysis()
