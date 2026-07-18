import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import List, Optional, Dict

class OptimizationBranch:
    def __init__(self, parent):
        self.parent = parent

    def optimize_treatment(
        self,
        X: pd.DataFrame,
        t_bounds=(0.0, 10.0),
        cost_per_unit: float = 0.0,
        grid_resolution: int = 100
    ) -> pd.Series:
        """
        Optimizes a continuous treatment variable for each sample in X.
        Finds T* in t_bounds that maximizes Y_predicted(X, T) - cost_per_unit * T.
        """
        import numpy as np
        predictive_branch = self.parent.predictive
        if predictive_branch.modeling.model is None:
            predictive_branch.fit()

        treatment_name = self.parent.treatment
        if not treatment_name:
            raise ValueError("Treatment variable name must be defined for optimization.")

        t_grid = np.linspace(t_bounds[0], t_bounds[1], grid_resolution)
        n_samples = len(X)
        features = self.parent.features

        batch_list = []
        for t_val in t_grid:
            sub_df = X[features].copy()
            sub_df[treatment_name] = t_val
            sub_df["_t_val"] = t_val
            sub_df["_sample_idx"] = np.arange(n_samples)
            batch_list.append(sub_df)

        large_df = pd.concat(batch_list, ignore_index=True)
        preds = predictive_branch.predict(large_df)
        large_df["y_pred"] = preds
        large_df["utility"] = large_df["y_pred"] - cost_per_unit * large_df["_t_val"]

        best_t_idx = large_df.groupby("_sample_idx")["utility"].idxmax()
        optimal_t = large_df.loc[best_t_idx, "_t_val"].values
        return pd.Series(optimal_t, index=X.index, name="optimal_treatment")

    def counterfactual(
        self,
        patient_df: pd.DataFrame,
        outcome: int = 0,
        k_neighbors: int = 5,
        top_k_causes: int = 3,
        use_mixture: bool = False
    ) -> pd.DataFrame:
        """
        Computes the counterfactual feature state for a patient.

        Given a patient with outcome=1 (e.g. sick), this function:
        1. Finds the K nearest neighbors in the data with the target outcome (e.g. healthy).
        2. Uses RCA attribution to identify the top root-cause features driving the difference.
        3. Runs the SCM simulation with those root-cause features set to the reference
           (healthy neighbor) level, keeping the patient's personal noise terms intact.
        4. Returns a comparison table: original vs counterfactual values for all features.

        Parameters
        ----------
        patient_df : pd.DataFrame
            A single-row DataFrame representing the patient.
        outcome : int
            The target outcome we want to simulate (e.g. 0 = no disease).
        k_neighbors : int
            Number of nearest reference neighbors to use.
        top_k_causes : int
            Number of top causal features to intervene on.
        use_mixture : bool
            Whether to use the MoSCM mixture engine for simulation.

        Returns
        -------
        pd.DataFrame
            Comparison DataFrame with columns: Original, Counterfactual, Delta.
        """
        target = self.parent.target
        features = self.parent.features
        data = self.parent.data

        # 1. Find nearest neighbors with the desired outcome
        reference_pool = data[data[target] == outcome]
        if len(reference_pool) == 0:
            raise ValueError(f"No reference samples found with target == {outcome}.")

        nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, len(reference_pool)), metric="euclidean")
        nbrs.fit(reference_pool[features].values)
        distances, indices = nbrs.kneighbors(patient_df[features].values)
        reference_cohort = reference_pool.iloc[indices[0]]
        reference_mean = reference_cohort[features].mean()

        # 2. Find top root-cause features (biggest delta between patient and reference)
        patient_vals = patient_df[features].iloc[0]
        delta = (patient_vals - reference_mean).abs().sort_values(ascending=False)
        top_causes = delta.head(top_k_causes).index.tolist()

        # 3. Build interventions: set top-cause features to reference level
        interventions = {feat: float(reference_mean[feat]) for feat in top_causes}

        # 4. Run SCM simulation (abduct noise from patient, apply interventions, propagate DAG)
        if use_mixture:
            if not self.parent.rca.mixture_engine.is_fitted:
                self.parent.rca.fit_mixture_scm()
            simulated_df = self.parent.simulate_mixture(patient_df, interventions)
        else:
            if not self.parent.rca.engine.is_fitted:
                self.parent.rca.fit_scm()
            simulated_df = self.parent.simulate(patient_df, interventions)

        # 5. Log this experiment to policy memory
        trial = {
            "patient_idx": patient_df.index[0] if len(patient_df.index) > 0 else -1,
            "top_causes": top_causes,
            "interventions": interventions,
            "original_outcome": float(patient_df[target].iloc[0]) if target in patient_df.columns else None,
            "simulated_outcome": float(simulated_df[target].iloc[0]) if target in simulated_df.columns else None,
            "outcome_target": outcome
        }
        self.parent.policy.log_experiment(trial)

        # 6. Build comparison table
        all_cols = features + ([target] if target in simulated_df.columns else [])
        original_vals = patient_df[all_cols].iloc[0]
        simulated_vals = simulated_df[all_cols].iloc[0]

        comparison = pd.DataFrame({
            "Original": original_vals,
            "Counterfactual": simulated_vals,
            "Delta": simulated_vals - original_vals,
            "Intervened": [feat in interventions for feat in all_cols]
        })
        return comparison
