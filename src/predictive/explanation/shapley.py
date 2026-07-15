import pandas as pd
import numpy as np

class ShapleyExplainer:
    def __init__(self, branch):
        self.branch = branch
        self.parent = branch.parent

    def explain(self, method: str = "shap", **kwargs) -> pd.Series:
        """
        Explain the model feature importances.
        If SHAP is installed and method is 'shap', use SHAP.
        Otherwise, fall back to Random Forest Mean Decrease Impurity (MDI).
        """
        model = self.branch.modeling.model
        features = self.parent.features.copy()
        treatment = self.parent.treatment
        if treatment and treatment not in features:
            features.append(treatment)
        
        if model is None:
            raise ValueError("Model must be trained before explaining. Call fit() first.")
            
        if method.lower() == "shap":
            try:
                import shap
                print("Calculating SHAP values using shap library...")
                X = self.parent.data[features]
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                if isinstance(shap_values, list):
                    mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                else:
                    mean_shap = np.abs(shap_values).mean(axis=0)
                return pd.Series(mean_shap, index=features).sort_values(ascending=False)
            except ImportError:
                print("SHAP package not installed. Falling back to default Feature Importances (MDI)...")
                
        # Fallback to feature_importances_
        importances = model.feature_importances_
        return pd.Series(importances, index=features).sort_values(ascending=False)
