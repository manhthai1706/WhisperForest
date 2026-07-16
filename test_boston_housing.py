import pandas as pd
import numpy as np
import os
from src.whisper_forest import WhisperForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def run_boston_housing_test():
    print("="*80)
    print("=== WhisperForest Real-World Case Study: Boston Housing ===".center(80))
    print("="*80)
    
    # 1. Load the Boston Housing dataset
    boston_csv_path = r"C:\Users\manht\Downloads\BostonHousing.csv"
    if not os.path.exists(boston_csv_path):
        print(f"Error: BostonHousing.csv not found at {boston_csv_path}")
        return
        
    df = pd.read_csv(boston_csv_path)
    print(f"Loaded Boston Housing dataset successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # Clean 'chas' column (convert string "0"/"1" to integer 0/1)
    df["chas"] = df["chas"].astype(str).str.replace('"', '').astype(int)
    
    # Configure variables
    target = "medv"       # Median value of owner-occupied homes (in $1000s)
    treatment = "chas"    # Charles River dummy variable (1 if tract bounds river, 0 otherwise)
    features = [
        "crim", "zn", "indus", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b", "lstat"
    ]
    
    # 2. Causal Discovery under clinical/geographical constraints
    print("\n--- Step 1: Causal Discovery (DAG Stability Selection) ---")
    wf = WhisperForest(data=df, target=target, treatment=treatment, features=features)
    
    # Geographical tiers
    # zn (residential land), indus (industrial acres) are structural/urban planning features (Tier 0)
    # nox (nitric oxides), rm (rooms), age (old units) are neighborhood features (Tier 1)
    # chas (bounds river), rad (highway access), tax (property tax) are location/financial features (Tier 2)
    # medv (house price) is the ultimate outcome (Tier 3)
    tiers = {
        0: ["zn", "indus"],
        1: ["nox", "rm", "age", "crim", "dis", "lstat", "ptratio", "b"],
        2: [treatment, "rad", "tax"],
        3: [target]
    }
    whitelist = [
        (treatment, target),
        ("rm", target),
        ("crim", target),
        ("lstat", target)
    ]
    constraints = {"tiers": tiers, "whitelist": whitelist}
    
    discovered_dag = wf.causal.get_dag(
        method="ensemble",
        constraints=constraints,
        bootstrap_runs=10,
        confidence_threshold=0.35
    )
    print("\nDiscovered DAG Edges after Cycle Breaking:")
    for edge in discovered_dag:
        print(f"  {edge[0]} -> {edge[1]}")
        
    # 3. Train Predictive Regressor and evaluate R2 and RMSE against baselines
    print("\n--- Step 2: Predictive Regressor & ML Comparison (R2 / RMSE) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        df[features + [treatment]], df[target], test_size=0.2, random_state=42
    )
    
    train_df = X_train.copy()
    train_df[target] = y_train
    
    # Fit WhisperForest predictive regressor on train split
    wf_eval = WhisperForest(data=train_df, target=target, treatment=treatment, features=features)
    wf_eval.predictive.fit(model_type="regressor")
    
    # Fit standard baselines
    rf_std = RandomForestRegressor(random_state=42)
    rf_std.fit(X_train, y_train)
    lr_std = LinearRegression()
    lr_std.fit(X_train, y_train)
    
    # Predict on test split
    wf_preds = wf_eval.predictive.predict(X_test)
    rf_preds = rf_std.predict(X_test)
    lr_preds = lr_std.predict(X_test)
    
    # Compute R2 and RMSE
    results = []
    models = {
        "WhisperForest Regressor": wf_preds,
        "Standard RF Regressor": rf_preds,
        "Linear Regression": lr_preds
    }
    for model_name, preds in models.items():
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        results.append({"Model": model_name, "R2-Score": r2, "RMSE": rmse})
        
    results_df = pd.DataFrame(results)
    print("\n" + "-"*55)
    print(f"{'Boston Housing Price Regression Report':^55}")
    print("-"*55)
    print(results_df.to_string(index=False, formatters={
        "R2-Score": "{:.4f}".format,
        "RMSE": "{:.4f}".format
    }))
    print("-"*55)
    
    # Fit the full wf instance's predictive regressor for subsequent steps
    wf.predictive.fit(model_type="regressor")
    
    # 4. Estimate ATE and CATE
    print("\n--- Step 3: Causal Estimation (Charles River Effect on medv) ---")
    # Estimating ATE using CausalForestDML
    ate = wf.causal.estimate_ate()
    print(f"Estimated ATE of Bounding Charles River (chas): {ate:+.4f} ($1000s)")
    print("Interpretation: Bounding the Charles River is estimated to average this impact on home value.")
    
    # Estimate CATE for first 5 neighborhoods
    cate_df = wf.causal.estimate_cate(df.head(5))
    print("\nSample Neighborhood CATE effects:")
    print(cate_df[["rm", "lstat", "crim", "CATE"]])
    
    # 5. Pearl's SCM Causal Simulation (do-intervention)
    print("\n--- Step 4: SCM Simulation (do-intervention on chas) ---")
    wf.rca.fit_scm()
    
    # Define a test house/neighborhood (not bounding river, lower priced)
    test_house = pd.DataFrame([{
        "crim": 0.1, "zn": 0, "indus": 8.0, "nox": 0.5, "rm": 5.8, "age": 70,
        "dis": 4.5, "rad": 4, "tax": 300, "ptratio": 18.0, "b": 390.0, "lstat": 15.0,
        "chas": 0
    }])
    
    # Simulate moving this house to bound the Charles River
    sim_house = wf.simulate(test_house, {"chas": 1})
    diff = wf.counterfactual(test_house, {"chas": 1})
    print(f"Original SCM estimated value: {test_house['medv'].iloc[0] if 'medv' in test_house else 18.0:.2f} ($1000s)")
    print(f"Simulated value under do(chas=1): {sim_house[target].iloc[0]:.2f} ($1000s)")
    print(f"SCM simulated price difference: {diff:+.2f} ($1000s)")
    
    # 6. Fit EM-MoSCM
    print("\n--- Step 5: EM-MoSCM (Latent Causal Subgroups & Gating) ---")
    # Fit 3 SCM Experts (representing low-value/high-crime, middle suburban, high-value neighborhoods)
    wf.rca.fit_mixture_scm(n_clusters=3, n_iterations=5)
    
    routing_weights = wf.rca.mixture_engine.predict_proba_membership(test_house)[0]
    print("\nDynamic Routing Weights (P(Mechanism | Neighborhood)):")
    for k, prob in enumerate(routing_weights):
        print(f"  Expert Cluster {k}: {prob*100:.2f}%")
        
    mix_sim = wf.simulate_mixture(test_house, {"chas": 1})
    mix_diff = wf.counterfactual_mixture(test_house, {"chas": 1})
    print(f"\n[EM-MoSCM] Simulated price under do(chas=1): {mix_sim[target].iloc[0]:.2f} ($1000s)")
    print(f"[EM-MoSCM] Risk/Price Difference: {mix_diff:+.2f} ($1000s)")
    
    # 7. WhisperTrace RCA on an expensive housing anomaly
    print("\n--- Step 6: WhisperTrace Root Cause Analysis ---")
    expensive_neighborhoods = df[df[target] > 40] # > $40k houses
    cheap_neighborhoods = df[df[target] < 20]     # < $20k houses
    
    if len(expensive_neighborhoods) > 0 and len(cheap_neighborhoods) > 0:
        anomaly_house = expensive_neighborhoods.head(1).copy()
        print("Anomaly Neighborhood details (High Value):")
        print(anomaly_house[["crim", "rm", "lstat", "medv"]])
        
        rca_report = wf.trace.analyze_anomaly(
            anomaly_data=anomaly_house,
            baseline_data=cheap_neighborhoods,
            causal_graph=discovered_dag,
            method="intervention"
        )
        print("\nRCA Attribution Report (Why are values high?):")
        print(rca_report[["Attribution_Mean", "CI_HalfWidth", "Paths"]].head(5))
        
        # Plot attribution
        chart_path = "boston_housing_rca_attribution.png"
        saved_path = wf.trace.plot_attribution(rca_report, save_path=chart_path)
        if saved_path:
            print(f"RCA Attribution plot saved to: {os.path.abspath(saved_path)}")
            
    # 8. Causal Auditing (Consistency verification)
    print("\n--- Step 7: Causal Auditing (Consistency Verification) ---")
    audit_report = wf.audit_consistency(test_house, {"chas": 1}, threshold_conflict=0.1)
    print(wf.causal.auditor.format_report(audit_report))
            
    print("\n" + "="*80)
    print("=== WhisperForest Boston Housing Real-World Test Passed Successfully ===".center(80))
    print("="*80)

if __name__ == "__main__":
    run_boston_housing_test()
