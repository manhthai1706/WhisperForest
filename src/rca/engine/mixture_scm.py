import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

class ClusterSCM:
    def __init__(self, features: List[str], target: str, treatment: Optional[str], dag: List[Tuple[str, str]]):
        self.features = features
        self.target = target
        self.treatment = treatment
        self.dag = dag
        self.models = {}
        self.parents_map = {}
        
        # Build parents map
        for u, v in self.dag:
            if v not in self.parents_map:
                self.parents_map[v] = []
            self.parents_map[v].append(u)
            
        all_nodes = set(features)
        if treatment:
            all_nodes.add(treatment)
        all_nodes.add(target)
        
        for node in all_nodes:
            if node not in self.parents_map:
                self.parents_map[node] = []
                
    def fit(self, data: pd.DataFrame, sample_weight: Optional[np.ndarray] = None):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        for node in self.parents_map:
            parents = self.parents_map.get(node, [])
            if len(parents) > 0:
                is_binary = data[node].nunique() <= 2
                if is_binary:
                    self.models[node] = RandomForestClassifier(random_state=42)
                else:
                    self.models[node] = RandomForestRegressor(random_state=42)
                
                if sample_weight is not None:
                    self.models[node].fit(data[parents], data[node], sample_weight=sample_weight)
                else:
                    self.models[node].fit(data[parents], data[node])
            else:
                self.models[node] = None
                
    def compute_log_likelihood(self, df: pd.DataFrame) -> np.ndarray:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        n_samples = len(df)
        log_like = np.zeros(n_samples)
        
        for node, model in self.models.items():
            parents = self.parents_map.get(node, [])
            if model is not None and len(parents) > 0:
                if isinstance(model, RandomForestClassifier):
                    prob = model.predict_proba(df[parents])
                    if prob.shape[1] == 2:
                        p1 = prob[:, 1]
                    else:
                        p1 = np.zeros(n_samples) if model.classes_[0] == 0 else np.ones(n_samples)
                    
                    p1 = np.clip(p1, 1e-15, 1.0 - 1e-15)
                    y = df[node].values
                    log_like += y * np.log(p1) + (1 - y) * np.log(1 - p1)
                else:
                    pred = model.predict(df[parents])
                    resid = df[node].values - pred
                    var_e = np.var(resid)
                    if var_e < 1e-3:
                        var_e = 1e-3
                    log_like += -0.5 * np.log(2 * np.pi * var_e) - (resid**2) / (2 * var_e)
        return log_like

    def compute_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        from sklearn.ensemble import RandomForestClassifier
        noise_df = pd.DataFrame(index=df.index)
        for node, model in self.models.items():
            if node not in df.columns:
                noise_df[node] = 0.0
                continue
            parents = self.parents_map.get(node, [])
            if model is not None and len(parents) > 0:
                if isinstance(model, RandomForestClassifier):
                    prob = model.predict_proba(df[parents])
                    pred = prob[:, 1] if prob.shape[1] == 2 else (np.zeros(len(df)) if model.classes_[0] == 0 else np.ones(len(df)))
                    if node == self.target:
                         noise_df[node] = df[node] - pred
                    else:
                         u_val = np.where(df[node] == 1, pred / 2.0, (pred + 1.0) / 2.0)
                         noise_df[node] = u_val
                else:
                    pred = model.predict(df[parents])
                    noise_df[node] = df[node] - pred
            else:
                noise_df[node] = df[node]
        return noise_df
        
    def simulate(self, patient: pd.DataFrame, interventions: Dict[str, float]) -> pd.DataFrame:
        from sklearn.ensemble import RandomForestClassifier
        noise_df = self.compute_noise(patient)
        
        topo_order = self._get_topological_order()
        simulated_df = patient.copy()
        
        for node in topo_order:
            if node in interventions:
                simulated_df[node] = interventions[node]
            else:
                parents = self.parents_map.get(node, [])
                if len(parents) > 0:
                    model = self.models.get(node)
                    if model is not None:
                        if isinstance(model, RandomForestClassifier):
                            prob = model.predict_proba(simulated_df[parents])
                            pred = prob[:, 1] if prob.shape[1] == 2 else (np.zeros(len(simulated_df)) if model.classes_[0] == 0 else np.ones(len(simulated_df)))
                            if node == self.target:
                                simulated_df[node] = pred + noise_df[node]
                            else:
                                simulated_df[node] = (pred >= noise_df[node]).astype(int)
                        else:
                            pred = model.predict(simulated_df[parents])
                            simulated_df[node] = pred + noise_df[node]
                    else:
                        simulated_df[node] = noise_df[node]
                else:
                    simulated_df[node] = noise_df[node]
        return simulated_df

    def _get_topological_order(self) -> List[str]:
        adj = {}
        in_degree = {}
        all_nodes = set(self.features)
        if self.treatment:
            all_nodes.add(self.treatment)
        all_nodes.add(self.target)
        
        for u, v in self.dag:
            if u not in adj:
                adj[u] = []
            adj[u].append(v)
            in_degree[v] = in_degree.get(v, 0) + 1
            if u not in in_degree:
                in_degree[u] = 0
                
        for n in all_nodes:
            if n not in in_degree:
                in_degree[n] = 0
            if n not in adj:
                adj[n] = []
                
        queue = [n for n in all_nodes if in_degree.get(n, 0) == 0]
        topo_order = []
        while queue:
            node = queue.pop(0)
            topo_order.append(node)
            for neighbor in adj.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        remaining = all_nodes - set(topo_order)
        topo_order.extend(list(remaining))
        return topo_order


class MixtureOfSCMEngine:
    def __init__(self, branch):
        self.branch = branch
        self.parent = branch.parent
        self.n_clusters = None
        self.scaler = None
        self.router = None
        self.scm_models = {}  # Map cluster_idx -> ClusterSCM
        self.is_fitted = False

    def fit(self, n_clusters: int = 3, n_iterations: int = 5):
        """
        Fits a Hierarchical Mixture of SCM Experts (EM-MoSCM) by:
        1. Initializing routing weights using GMM.
        2. Iteratively training ClusterSCMs (M-step) and updating routing weights (E-step).
        3. Fitting a supervised Gating Router on the final weights.
        """
        self.n_clusters = n_clusters
        data = self.parent.data
        features = self.parent.features
        target = self.parent.target
        treatment = self.parent.treatment
        dag = self.parent.causal.get_dag()
        
        # 1. Scale covariates and initialize weights using GMM
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(data[features])
        
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(X_scaled)
        weights = gmm.predict_proba(X_scaled)
        
        print(f"EM-MoSCM: Initializing {n_clusters} SCM Experts using GMM...")
        
        # 2. EM Loop
        for iteration in range(n_iterations):
            # M-step: Fit each ClusterSCM expert using soft weights
            for k in range(n_clusters):
                w_k = weights[:, k]
                w_k_norm = w_k / (w_k.sum() + 1e-15) * len(data)
                
                scm = ClusterSCM(features=features, target=target, treatment=treatment, dag=dag)
                scm.fit(data, sample_weight=w_k_norm)
                self.scm_models[k] = scm
                
            # E-step: Update weights based on SCM Likelihood & prior
            priors = weights.mean(axis=0)
            
            log_likelihoods = np.zeros((len(data), n_clusters))
            for k in range(n_clusters):
                log_likelihoods[:, k] = self.scm_models[k].compute_log_likelihood(data)
                
            log_posterior = log_likelihoods + np.log(priors + 1e-15)
            max_log = np.max(log_posterior, axis=1, keepdims=True)
            exp_posterior = np.exp(log_posterior - max_log)
            weights = exp_posterior / np.sum(exp_posterior, axis=1, keepdims=True)
            
            cluster_sizes = weights.sum(axis=0)
            print(f"    EM Iteration {iteration + 1}/{n_iterations} - SCM Experts Soft Sizes: {list(np.round(cluster_sizes, 2))}")
            
        # 3. Train a supervised Gating Router on final weights
        print("\nTraining Supervised Gating Router...")
        self.router = RandomForestRegressor(n_estimators=100, random_state=42)
        self.router.fit(X_scaled, weights)
        print("Gating Router trained successfully.")
        
        self.is_fitted = True

    def predict_proba_membership(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts the membership probabilities of each cluster for the given patients using the Gating Router.
        """
        if not self.is_fitted:
            raise ValueError("MixtureOfSCMEngine must be fitted first.")
        features = self.parent.features
        X_scaled = self.scaler.transform(X[features])
        weights = self.router.predict(X_scaled)
        
        if len(weights.shape) == 1:
            weights = weights.reshape(1, -1)
            
        weights = np.clip(weights, 0, 1)
        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return weights / row_sums

    def simulate(self, patient: pd.DataFrame, interventions: Dict[str, float]) -> pd.DataFrame:
        """
        Simulates the counterfactual state under interventions across the mixture of SCMs.
        Computes the weighted average of outcomes based on cluster membership probabilities.
        """
        if not self.is_fitted:
            raise ValueError("MixtureOfSCMEngine must be fitted first.")
            
        weights = self.predict_proba_membership(patient) # shape: (n_patients, n_clusters)
        
        target = self.parent.target
        n_patients = len(patient)
        
        accumulated_outcomes = np.zeros(n_patients)
        base_simulated = None
        
        for k in range(self.n_clusters):
            scm = self.scm_models[k]
            sim_k = scm.simulate(patient, interventions)
            
            if k == 0:
                base_simulated = sim_k.copy()
            
            accumulated_outcomes += weights[:, k] * sim_k[target].values
            
        final_simulated = base_simulated.copy()
        final_simulated[target] = accumulated_outcomes
        return final_simulated

    def counterfactual(self, patient: pd.DataFrame, interventions: Dict[str, float]) -> Union[float, pd.Series]:
        """
        Computes the counterfactual risk difference (Y_simulated - Y_original) for the given patient(s).
        """
        simulated_df = self.simulate(patient, interventions)
        target = self.parent.target
        
        weights = self.predict_proba_membership(patient)
        accumulated_originals = np.zeros(len(patient))
        
        for k in range(self.n_clusters):
            scm = self.scm_models[k]
            sim_k = scm.simulate(patient, {})
            accumulated_originals += weights[:, k] * sim_k[target].values
            
        diff = simulated_df[target] - accumulated_originals
        if len(diff) == 1:
            return float(diff.iloc[0])
        return diff
