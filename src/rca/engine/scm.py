import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from typing import Dict, List, Tuple

class SCMEngine:
    def __init__(self, branch):
        self.branch = branch
        self.parent = branch.parent
        self.models = {}  # Map node -> RandomForestRegressor or RandomForestClassifier
        self.parents_map = {}  # Map node -> list of parent nodes
        self.is_fitted = False

    def fit(self):
        """
        Fits structural equations X_i = f_i(Pa_i) + U_i for each node in the DAG.
        Uses RandomForestClassifier for binary variables and RandomForestRegressor for continuous variables.
        """
        dag = self.parent.causal.get_dag()
        data = self.parent.data
        
        self.parents_map = {}
        all_nodes = set()
        
        for u, v in dag:
            all_nodes.add(u)
            all_nodes.add(v)
            if v not in self.parents_map:
                self.parents_map[v] = []
            self.parents_map[v].append(u)
            
        # Add target, treatment, and features to all_nodes
        all_nodes.add(self.parent.target)
        if self.parent.treatment:
            all_nodes.add(self.parent.treatment)
        for f in self.parent.features:
            all_nodes.add(f)
            
        for node in all_nodes:
            if node not in self.parents_map:
                self.parents_map[node] = []
                
        # Fit models for nodes that have parents
        for node in all_nodes:
            parents = self.parents_map.get(node, [])
            if len(parents) > 0:
                # Auto-detect binary variables (mixed-type SCM)
                is_binary = data[node].nunique() <= 2
                
                if is_binary:
                    print(f"WhisperTrace SCM: Fitting Classifier for binary node '{node}' using parents: {parents}")
                    model = RandomForestClassifier(random_state=42)
                else:
                    print(f"WhisperTrace SCM: Fitting Regressor for continuous node '{node}' using parents: {parents}")
                    model = RandomForestRegressor(random_state=42)
                    
                model.fit(data[parents], data[node])
                self.models[node] = model
            else:
                self.models[node] = None
                
        self.is_fitted = True
        print("SCM equations fitted successfully.")

    def compute_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the noise (exogenous) values U_i for each node in df.
        For binary variables: U_i = X_i - P(X_i = 1 | Pa_i)
        For continuous variables: U_i = X_i - f_i(Pa_i)
        """
        if not self.is_fitted:
            raise ValueError("SCM must be fitted first. Run fit().")
            
        noise_df = pd.DataFrame(index=df.index)
        
        for node, model in self.models.items():
            if node not in df.columns:
                noise_df[node] = 0.0
                continue
                
            parents = self.parents_map.get(node, [])
            if model is not None and len(parents) > 0:
                if isinstance(model, RandomForestClassifier):
                    prob = model.predict_proba(df[parents])
                    if prob.shape[1] == 2:
                        pred = prob[:, 1]
                    else:
                        pred = np.zeros(len(df)) if model.classes_[0] == 0 else np.ones(len(df))
                    
                    if node == self.branch.parent.target:
                        # For final target node, use continuous additive risk residue
                        noise_df[node] = df[node] - pred
                    else:
                        # For intermediate binary nodes, use threshold-based abduction
                        u_val = np.where(df[node] == 1, pred / 2.0, (pred + 1.0) / 2.0)
                        noise_df[node] = u_val
                else:
                    pred = model.predict(df[parents])
                    noise_df[node] = df[node] - pred
            else:
                noise_df[node] = df[node]
                
        return noise_df
