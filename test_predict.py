import pandas as pd
import numpy as np
import os
from src.whisper_forest import WhisperForest

def run_prediction_test():
    print("=== WhisperForest Causal & Predictive Testing Script ===")
    
    # 1. Load the original dataset to reference the healthy pool
    heart_csv_path = "heart.csv"
    if not os.path.exists(heart_csv_path):
        print(f"Error: heart.csv not found.")
        return
        
    df = pd.read_csv(heart_csv_path)
    
    # Simulate statin_treatment matching our trained model data structure
    np.random.seed(42)
    age_std = (df["age"] - 55) / 10.0
    chol_std = (df["chol"] - 220) / 40.0
    presc_logits = 0.8 * age_std + 0.6 * chol_std
    presc_probs = 1.0 / (1.0 + np.exp(-presc_logits))
    statin_treatment = (np.random.rand(len(df)) < presc_probs).astype(int)
    
    target_flipped = df["target"].copy()
    flip_mask = (statin_treatment == 1) & (df["target"] == 1)
    flips = np.random.rand(len(df)) < 0.35
    target_flipped[flip_mask & flips] = 0
    
    df["statin_treatment"] = statin_treatment
    df["target"] = target_flipped
    
    # 2. Instantiate new WhisperForest engine
    features = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", 
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    wf = WhisperForest(data=df, target="target", treatment="statin_treatment", features=features)
    
    # 3. Load the saved model pipeline
    model_path = "whisper_forest_model.h5"
    if not os.path.exists(model_path):
        print(f"Error: Trained model '{model_path}' not found. Please run 'python test_heart.py' first.")
        return
        
    print(f"\nLoading model pipeline from: {model_path}...")
    wf.load_model(model_path)
    
    # 4. Define two test patients (observing counterfactual treatment effect)
    # Patient A does NOT take statins, Patient B is the exact same patient but TAKING statins
    test_patients = pd.DataFrame([
        {
            "age": 62, "sex": 1, "cp": 2, "trestbps": 140, "chol": 280, "fbs": 0,
            "restecg": 0, "thalach": 130, "exang": 1, "oldpeak": 2.2, "slope": 1,
            "ca": 2, "thal": 2, "statin_treatment": 0 # Untreated
        },
        {
            "age": 62, "sex": 1, "cp": 2, "trestbps": 140, "chol": 280, "fbs": 0,
            "restecg": 0, "thalach": 130, "exang": 1, "oldpeak": 2.2, "slope": 1,
            "ca": 2, "thal": 2, "statin_treatment": 1 # Treated with statins
        }
    ])
    
    print("\nTest Patients Data:")
    print(test_patients[["age", "chol", "oldpeak", "statin_treatment"]])
    
    # 5. Run Predictive Risk Inference (Static conditional probability)
    print("\n--- Step 1: Predictive Risk Inference (Static Conditional P(Y|X)) ---")
    preds = wf.predictive.predict_proba(test_patients)
    print(f"Patient A (Untreated) Heart Disease Probability: {preds[0, 1]*100:.2f}%")
    print(f"Patient B (Treated with Statins - Static) Heart Disease Probability: {preds[1, 1]*100:.2f}%")
    
    # 5.1 Run Causal Decision Simulation (do-intervention propagation)
    print("\n--- Step 1.1: SCM Causal Decision Simulation (Pearl's do-intervention) ---")
    # We take Patient A (who is untreated, statin_treatment=0)
    patient_a = test_patients.iloc[[0]].copy()
    
    # Simulate do(statin_treatment = 1) and propagate downstream effects topologically using SCM
    simulated_patient_a = wf.simulate(patient_a, {"statin_treatment": 1})
    print("\nSimulated Counterfactual Patient A State (do(statin_treatment=1)):")
    print(simulated_patient_a[["age", "chol", "statin_treatment", "target"]])
    
    # Calculate SCM-propagated counterfactual risk difference
    cf_risk_diff = wf.counterfactual(patient_a, {"statin_treatment": 1})
    print(f"\nSCM-Propagated Counterfactual Risk Difference: {cf_risk_diff*100:+.2f}%")
    
    # 6. Run Causal Estimation (Observational CATE via EconML DML)
    print("\n--- Step 2: Causal CATE Estimation (DML) ---")
    cate_df = wf.causal.estimate_cate(test_patients)
    patient_cate = cate_df["CATE"].iloc[0]
    print(f"Estimated Statin Treatment Effect (CATE) for this patient: {patient_cate:.4f}")
    print(f"Risk Reduction: {-patient_cate*100:.2f}%")
    
    # 7. Run WhisperTrace RCA to trace why Patient A has high risk
    print("\n--- Step 3: WhisperTrace Root Cause Tracing (Patient A vs Healthy Cohort) ---")
    healthy_pool = df[df["target"] == 0]
    anomaly_patient = test_patients.iloc[[0]].copy()
    # set target to 1 to represent anomaly
    anomaly_patient["target"] = 1 
    
    rca_report = wf.trace.analyze_anomaly(
        anomaly_data=anomaly_patient,
        baseline_data=healthy_pool,
        causal_graph=wf.causal.get_dag(),
        method="intervention",
        k_neighbors=10
    )
    
    print("\nWhisperTrace RCA Report:")
    print(rca_report[["Attribution_Mean", "CI_HalfWidth", "Paths"]].head(5))
    
    # 8. Policy Recommendation
    print("\n--- Step 4: Policy Recommendation ---")
    cost_threshold = 0.08
    recommended = wf.policy.recommend_actions(pd.Series([patient_cate]), cost=cost_threshold, minimize_outcome=True).iloc[0]
    print(f"Treatment cost threshold: {cost_threshold}")
    print(f"Is statin treatment recommended for this patient? {'YES (Benefit exceeds cost)' if recommended == 1 else 'NO'}")
    
    # 9. Causal Consistency Evaluation
    print("\n--- Step 5: Multi-Layer Causal Consistency Diagnosis ---")
    consistency_results = wf.evaluate_causal_consistency(patient_a, {"statin_treatment": 1})
    print(f"Predictive Layer Effect:  {consistency_results['Predictive_Effect']*100:+.2f}%")
    print(f"SCM Structural Layer Effect: {consistency_results['SCM_Effect']*100:+.2f}%")
    print(f"Causal DML Layer Effect:  {consistency_results['DML_Effect']*100:+.2f}%")
    print(f"Consistency Score:        {consistency_results['Consistency_Score']:.4f}")
    print(f"Diagnostic Status:\n{consistency_results['Status']}")
    
    # 10. Deep SCM & Predictive Feature Diagnostics
    print("\n--- Deep Model Diagnostics (Simpson's Paradox Investigation) ---")
    
    # Predictive RF Importance
    pred_rf = wf.predictive.modeling.model
    pred_features = wf.features.copy()
    if wf.treatment and wf.treatment not in pred_features:
        pred_features.append(wf.treatment)
    if "statin_treatment" in pred_features:
        idx_pred = pred_features.index("statin_treatment")
        importances = pred_rf.feature_importances_
        print(f"Predictive RF - 'statin_treatment' Feature Importance: {importances[idx_pred]:.6f}")
        
    # SCM Target Classifier Importance
    target_scm_model = wf.rca.engine.models.get("target")
    target_scm_parents = wf.rca.engine.parents_map.get("target", [])
    if target_scm_model is not None and "statin_treatment" in target_scm_parents:
        idx_scm = target_scm_parents.index("statin_treatment")
        scm_importances = target_scm_model.feature_importances_
        print(f"SCM target Model - 'statin_treatment' Feature Importance: {scm_importances[idx_scm]:.6f}")
        
    # Print the exact SCM predicted probabilities under do-intervention
    if target_scm_model is not None:
        patient_do0 = patient_a.copy()
        patient_do0["statin_treatment"] = 0
        patient_do1 = patient_a.copy()
        patient_do1["statin_treatment"] = 1
        
        prob_do0 = target_scm_model.predict_proba(patient_do0[target_scm_parents])[0, 1]
        prob_do1 = target_scm_model.predict_proba(patient_do1[target_scm_parents])[0, 1]
        print(f"SCM target Probability (do(statin=0)): {prob_do0*100:.4f}%")
        print(f"SCM target Probability (do(statin=1)): {prob_do1*100:.4f}%")
        print(f"SCM target Probability Delta:        {(prob_do1 - prob_do0)*100:+.4f}%")

if __name__ == "__main__":
    run_prediction_test()
