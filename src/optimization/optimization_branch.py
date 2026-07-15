import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

class OptimizationBranch:
    def __init__(self, parent):
        self.parent = parent

    def optimize_treatment(
        self, 
        X: pd.DataFrame, 
        t_bounds: Tuple[float, float] = (0.0, 10.0), 
        cost_per_unit: float = 0.0,
        grid_resolution: int = 100
    ) -> pd.Series:
        """
        Optimizes a continuous treatment variable for each sample in X.
        Finds T* in t_bounds that maximizes Y_predicted(X, T) - cost_per_unit * T.
        """
        predictive_branch = self.parent.predictive
        if predictive_branch.modeling.model is None:
            print("Predictive model not trained yet. Training default model now...")
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
        
        # Predict Y
        preds = predictive_branch.predict(large_df)
        large_df["y_pred"] = preds
        
        # Calculate utility
        large_df["utility"] = large_df["y_pred"] - cost_per_unit * large_df["_t_val"]
        
        # Group by sample index and find argmax of utility
        best_t_idx = large_df.groupby("_sample_idx")["utility"].idxmax()
        optimal_t = large_df.loc[best_t_idx, "_t_val"].values
        
        opt_series = pd.Series(optimal_t, index=X.index, name="optimal_treatment")
        return opt_series
