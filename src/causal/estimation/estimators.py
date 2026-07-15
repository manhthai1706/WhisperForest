import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from typing import Optional

try:
    from econml.dml import CausalForestDML
    HAS_ECONML = True
except ImportError:
    HAS_ECONML = False

class CausalEstimation:
    def __init__(self, branch):
        self.branch = branch
        self.parent = branch.parent
        self.econml_estimator = None

    def estimate_ate(self) -> float:
        """
        Estimate the Average Treatment Effect (ATE).
        Uses EconML CausalForestDML if installed, otherwise falls back to a T-Learner.
        """
        data = self.parent.data
        target = self.parent.target
        treatment = self.parent.treatment
        features = self.parent.features
        
        if not treatment:
            raise ValueError("Treatment variable must be specified for ATE estimation.")
            
        if HAS_ECONML:
            print("Estimating ATE using EconML CausalForestDML...")
            try:
                # Detect treatment type
                is_discrete = data[treatment].nunique() <= 2
                
                model_y = RandomForestRegressor(random_state=42, n_jobs=1)
                if is_discrete:
                    model_t = RandomForestClassifier(random_state=42, n_jobs=1)
                else:
                    model_t = RandomForestRegressor(random_state=42, n_jobs=1)
                    
                est = CausalForestDML(
                    model_y=model_y,
                    model_t=model_t,
                    discrete_treatment=is_discrete,
                    n_jobs=1,
                    random_state=42
                )
                
                # Fit with float64 arrays to avoid type issues and deadlocks on Windows
                Y_arr = data[target].to_numpy().astype(np.float64)
                T_arr = data[treatment].to_numpy().astype(np.float64)
                X_arr = data[features].to_numpy().astype(np.float64)
                
                est.fit(Y_arr, T_arr, X=X_arr)
                self.econml_estimator = est
                
                # ATE is the average marginal effect over the population
                ate = est.ate(X_arr)
                return float(ate)
            except Exception as e:
                print(f"EconML fit failed: {e}. Falling back to T-Learner...")
                
        # T-Learner Fallback
        return self._estimate_ate_t_learner()

    def estimate_cate(self, new_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Estimate the Conditional Average Treatment Effect (CATE) for each sample.
        CATE(X) = E[Y|X, T=1] - E[Y|X, T=0]
        """
        data = self.parent.data if new_data is None else new_data
        target = self.parent.target
        treatment = self.parent.treatment
        features = self.parent.features
        
        if not treatment:
            raise ValueError("Treatment variable must be specified for CATE estimation.")
            
        if HAS_ECONML:
            try:
                if self.econml_estimator is None:
                    # Fit estimator if not already done
                    self.estimate_ate()
                    
                if self.econml_estimator is not None:
                    print("Estimating CATE using EconML CausalForestDML...")
                    X_arr = data[features].to_numpy().astype(np.float64)
                    cate = self.econml_estimator.effect(X_arr)
                    if len(cate.shape) > 1:
                        cate = cate.flatten()
                        
                    result = data.copy()
                    result["CATE"] = cate
                    return result
            except Exception as e:
                print(f"EconML CATE estimation failed: {e}. Falling back to T-Learner...")
                
        # T-Learner Fallback
        return self._estimate_cate_t_learner(data)

    def _estimate_ate_t_learner(self) -> float:
        data = self.parent.data
        target = self.parent.target
        treatment = self.parent.treatment
        features = self.parent.features
        
        # Binarize treatment if it is continuous
        is_discrete = data[treatment].nunique() <= 2
        if not is_discrete:
            median_val = data[treatment].median()
            t_bin = (data[treatment] > median_val).astype(int)
        else:
            t_bin = data[treatment]
            
        treated = data[t_bin == 1]
        control = data[t_bin == 0]
        
        if len(treated) == 0 or len(control) == 0:
            raise ValueError("Data must contain both treated and control samples after binarization.")
            
        model_treated = RandomForestRegressor(random_state=42)
        model_control = RandomForestRegressor(random_state=42)
        
        model_treated.fit(treated[features], treated[target])
        model_control.fit(control[features], control[target])
        
        pred_treated = model_treated.predict(data[features])
        pred_control = model_control.predict(data[features])
        
        ate = np.mean(pred_treated - pred_control)
        return float(ate)

    def _estimate_cate_t_learner(self, data: pd.DataFrame) -> pd.DataFrame:
        orig_data = self.parent.data
        target = self.parent.target
        treatment = self.parent.treatment
        features = self.parent.features
        
        # Binarize treatment if it is continuous
        is_discrete = orig_data[treatment].nunique() <= 2
        if not is_discrete:
            median_val = orig_data[treatment].median()
            t_bin = (orig_data[treatment] > median_val).astype(int)
        else:
            t_bin = orig_data[treatment]
            
        treated = orig_data[t_bin == 1]
        control = orig_data[t_bin == 0]
        
        model_treated = RandomForestRegressor(random_state=42)
        model_control = RandomForestRegressor(random_state=42)
        
        model_treated.fit(treated[features], treated[target])
        model_control.fit(control[features], control[target])
        
        pred_treated = model_treated.predict(data[features])
        pred_control = model_control.predict(data[features])
        
        cate = pred_treated - pred_control
        
        result = data.copy()
        result["CATE"] = cate
        return result

