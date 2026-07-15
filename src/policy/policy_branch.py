import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from typing import Union, List, Optional

class PolicyBranch:
    def __init__(self, parent):
        self.parent = parent

    def recommend_actions(self, cate_series: pd.Series, cost: float = 0.0, minimize_outcome: bool = False) -> pd.Series:
        """
        Recommends whether to treat (1) or not (0) for each individual.
        
        Parameters:
        -----------
        cate_series : pd.Series
            Conditional Average Treatment Effect (CATE) values.
        cost : float
            Threshold cost of treatment.
        minimize_outcome : bool
            If True, the outcome Y is something we want to minimize (e.g. disease risk).
            Treatment is recommended if risk reduction (-CATE) exceeds the cost.
            If False, treatment is recommended if risk increase (CATE) exceeds the cost.
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

