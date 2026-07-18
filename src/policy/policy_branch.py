import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from typing import Union, List, Optional, Dict

class PolicyBranch:
    def __init__(self, parent):
        self.parent = parent
        self.experiment_log: List[Dict] = []  # Accumulator of SCM simulations/recourses

    def log_experiment(self, trial_dict: Dict):
        """
        Ghi nhận một lượt chạy thử nghiệm nhân quả (trajectories) làm cơ sở dữ liệu tri thức.
        """
        self.experiment_log.append(trial_dict)

    def distill_policy_from_experiments(self, max_depth: int = 3) -> Optional[DecisionTreeClassifier]:
        """
        Đúc kết lịch sử thử nghiệm nhân quả đã ghi nhận thành một cây chính sách (Policy Tree).
        Cây này sẽ mô hình hóa: Với thuộc tính X của bệnh nhân, hành động can thiệp nào có xác suất thành công cao nhất.
        """
        if len(self.experiment_log) == 0:
            print("No trials logged in the policy experiment buffer yet.")
            return None
            
        # Convert log to training DataFrame
        df_log = pd.DataFrame(self.experiment_log)
        
        # We want to map patients features to whether a treatment shift succeeded
        # Reconstruct patient state at the time of experiment
        train_rows = []
        for _, trial in df_log.iterrows():
            pat_idx = int(trial["patient_idx"])
            if 0 <= pat_idx < len(self.parent.data):
                patient_row = self.parent.data.iloc[pat_idx].copy()
            else:
                # patient_idx unavailable (-1) -> use a zero row as placeholder features
                patient_row = pd.Series(0, index=self.parent.features + [self.parent.target]).copy()
                patient_row = patient_row.drop(self.parent.target, errors="ignore")

            # Label: 1 if this trial was successful in flipping the simulated outcome
            # towards the target outcome compared to the original outcome.
            orig = trial.get("original_outcome")
            sim  = trial.get("simulated_outcome")
            tgt  = trial.get("outcome_target")
            success = 0
            if orig is not None and sim is not None and tgt is not None:
                # Distance reduction towards the desired outcome counts as success
                if abs(orig - tgt) > 0:
                    if abs(sim - tgt) < abs(orig - tgt):
                        success = 1
                else:
                    success = 1
            patient_row["_action_success"] = success

            interventions = trial.get("interventions", {}) or {}
            if isinstance(interventions, dict) and len(interventions) > 0:
                first_feat = next(iter(interventions.keys()))
                first_shift = interventions[first_feat]
            else:
                first_feat = ""
                first_shift = 0.0
            patient_row["_action_feature"] = first_feat
            patient_row["_action_shift"] = first_shift
            train_rows.append(patient_row)
            
        train_df = pd.DataFrame(train_rows)
        features = self.parent.features
        
        X = train_df[features]
        y = train_df["_action_success"]
        
        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        tree.fit(X, y)
        
        print(f"Policy Tree successfully distilled from {len(df_log)} historical trials.")
        return tree

    def recommend_actions(self, cate_series: pd.Series, cost: float = 0.0, minimize_outcome: bool = False) -> pd.Series:
        """
        Recommends whether to treat (1) or not (0) for each individual.
        """
        benefit = -cate_series if minimize_outcome else cate_series
        actions = (benefit > cost).astype(int)
        actions.name = "recommended_action"
        return actions

    def recommend_constrained_actions(
        self, 
        cate_series: pd.Series, 
        budget_fraction: float = 0.2,
        minimize_outcome: bool = False
    ) -> pd.Series:
        """
        Recommends treatment for individuals under a budget constraint.
        Treats only the top budget_fraction of the population who benefit the most.
        """
        n_samples = len(cate_series)
        n_treat = int(np.floor(budget_fraction * n_samples))
        
        actions = pd.Series(0, index=cate_series.index, name="constrained_action")
        
        if n_treat <= 0:
            return actions
            
        # Get indices of top beneficial treatments
        if minimize_outcome:
            # Most negative CATE means highest risk reduction
            top_indices = cate_series.nsmallest(n_treat).index
        else:
            # Highest positive CATE means highest increase
            top_indices = cate_series.nlargest(n_treat).index
            
        actions.loc[top_indices] = 1
        return actions

    def learn_policy_tree(
        self, 
        X: pd.DataFrame, 
        cate_series: pd.Series, 
        cost: float = 0.0, 
        max_depth: int = 3,
        minimize_outcome: bool = False
    ) -> DecisionTreeClassifier:
        """
        Learns an interpretable decision tree that maps features X to optimal treatment actions.
        """
        features = self.parent.features
        X_input = X[features]
        
        # Target action based on optimization direction
        optimal_actions = self.recommend_actions(cate_series, cost=cost, minimize_outcome=minimize_outcome).values
        
        # Fit a simple tree
        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        tree.fit(X_input, optimal_actions)
        
        print(f"Policy tree trained successfully (depth={max_depth}).")
        return tree
