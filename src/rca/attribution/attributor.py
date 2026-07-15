import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

class Attributor:
    def __init__(self, branch):
        self.branch = branch
        self.parent = branch.parent

    def attribute(
        self, 
        anomaly_data: pd.DataFrame, 
        baseline_data: pd.DataFrame, 
        causal_graph: List[Tuple[str, str]], 
        method: str = "intervention",
        k_neighbors: int = 10
    ) -> pd.DataFrame:
        """
        Attributes the difference in target variable (Y) between anomaly_data and baseline_data
        to the causal variables using a Cohort-based Reference Baseline (KNN).
        
        Parameters:
        -----------
        anomaly_data : pd.DataFrame
            The anomalous patient/datapoint (typically a single row).
        baseline_data : pd.DataFrame
            The healthy baseline pool. If size > k_neighbors, KNN matching is performed.
        causal_graph : List[Tuple[str, str]]
            The causal DAG.
        method : str
            The attribution method: 'intervention' (default) or 'noise'.
        k_neighbors : int
            Number of nearest healthy neighbors to select as the reference cohort.
        """
        engine = self.branch.engine
        target = self.parent.target
        
        anomaly_row = anomaly_data.mean().to_frame().T if len(anomaly_data) > 1 else anomaly_data.copy()
        
        # 1. Identify the reference cohort (KNN matching)
        if len(baseline_data) > k_neighbors:
            print(f"Finding {k_neighbors} nearest healthy neighbors as reference cohort...")
            cohort = self._find_nearest_neighbors(anomaly_row.iloc[0], baseline_data, k=k_neighbors)
        else:
            cohort = baseline_data.copy()
            
        print(f"Reference cohort size: {len(cohort)} patients.")
        
        # 2. Run attribution against each baseline patient in the cohort
        cohort_results = []
        for idx in range(len(cohort)):
            baseline_row = cohort.iloc[[idx]]
            res_series = self._attribute_single_baseline(anomaly_row, baseline_row, causal_graph, method)
            cohort_results.append(res_series)
            
        # Convert to DataFrame: rows = cohort patients, columns = nodes
        results_df = pd.DataFrame(cohort_results)
        
        # 3. Calculate statistics: Mean and 95% Confidence Interval (SEM * 1.96)
        mean_scores = results_df.mean()
        std_scores = results_df.std().fillna(0.0)
        sem_scores = std_scores / np.sqrt(len(results_df))
        ci_half = 1.96 * sem_scores
        
        # 4. Trace Causal Paths
        paths_map = {}
        for node in mean_scores.index:
            paths = self._find_causal_paths(node, target, causal_graph)
            paths_map[node] = " | ".join(paths) if paths else "No direct causal path"
            
        # 5. Assemble final report DataFrame
        report_df = pd.DataFrame({
            "Attribution_Mean": mean_scores,
            "CI_HalfWidth": ci_half,
            "Paths": pd.Series(paths_map)
        })
        
        # Sort by absolute mean attribution descending
        report_df = report_df.sort_values(by="Attribution_Mean", key=abs, ascending=False)
        return report_df

    def _attribute_single_baseline(
        self, 
        anomaly_row: pd.DataFrame, 
        baseline_row: pd.DataFrame, 
        causal_graph: List[Tuple[str, str]], 
        method: str
    ) -> pd.Series:
        """
        Calculates attribution score against a single baseline patient.
        """
        engine = self.branch.engine
        target = self.parent.target
        
        u_anomaly = engine.compute_noise(anomaly_row).iloc[0].to_dict()
        u_baseline = engine.compute_noise(baseline_row).iloc[0].to_dict()
        
        anomaly_vals = anomaly_row.iloc[0].to_dict()
        baseline_vals = baseline_row.iloc[0].to_dict()
        
        topo_order = self._get_topological_order(causal_graph)
        
        from sklearn.ensemble import RandomForestClassifier
        
        if method.lower() == "intervention":
            attributions = {}
            y_baseline = baseline_row[target].iloc[0]
            
            for node in topo_order:
                if node == target:
                    continue
                    
                # Run SCM forward starting from baseline, but intervene on `node`
                values = {}
                for n in topo_order:
                    parents = engine.parents_map.get(n, [])
                    if n == node:
                        values[n] = anomaly_vals.get(n, 0.0)
                    else:
                        u_val = u_baseline.get(n, 0.0)
                        if len(parents) == 0:
                            values[n] = u_val
                        else:
                            model = engine.models.get(n)
                            if model is not None:
                                parents_val = {p: values.get(p, u_baseline.get(p, 0.0)) for p in parents}
                                parents_df = pd.DataFrame([parents_val])[parents]
                                
                                if isinstance(model, RandomForestClassifier):
                                    prob = model.predict_proba(parents_df)
                                    if prob.shape[1] == 2:
                                        pred = prob[0, 1]
                                    else:
                                        pred = 0.0 if model.classes_[0] == 0 else 1.0
                                else:
                                    pred = model.predict(parents_df)[0]
                                    
                                values[n] = pred + u_val
                            else:
                                values[n] = u_val
                                
                y_intervened = values.get(target, 0.0)
                attributions[node] = y_intervened - y_baseline
                
            return pd.Series(attributions)
            
        else:
            # Noise-based
            def simulate(u_dict: Dict[str, float]) -> float:
                values = {}
                for node in topo_order:
                    parents = engine.parents_map.get(node, [])
                    if len(parents) == 0:
                        values[node] = u_dict.get(node, 0.0)
                    else:
                        model = engine.models.get(node)
                        if model is not None:
                            parents_val = {p: values.get(p, u_dict.get(p, 0.0)) for p in parents}
                            parents_df = pd.DataFrame([parents_val])[parents]
                            
                            if isinstance(model, RandomForestClassifier):
                                prob = model.predict_proba(parents_df)
                                if prob.shape[1] == 2:
                                    pred = prob[0, 1]
                                else:
                                    pred = 0.0 if model.classes_[0] == 0 else 1.0
                            else:
                                pred = model.predict(parents_df)[0]
                                
                            values[node] = pred + u_dict.get(node, 0.0)
                        else:
                            values[node] = u_dict.get(node, 0.0)
                return values.get(target, 0.0)
                
            y_anomaly = anomaly_row[target].iloc[0]
            y_baseline = baseline_row[target].iloc[0]
            y_diff = y_anomaly - y_baseline
            
            if abs(y_diff) < 1e-9:
                return pd.Series(0.0, index=topo_order).drop(target, errors='ignore')
                
            attributions = {}
            for node in topo_order:
                if node == target:
                    continue
                u_cf = u_anomaly.copy()
                u_cf[node] = u_baseline.get(node, 0.0)
                
                y_cf = simulate(u_cf)
                contribution = y_anomaly - y_cf
                attributions[node] = contribution
                
            return pd.Series(attributions)

    def _find_nearest_neighbors(self, anomaly_row: pd.Series, baseline_pool: pd.DataFrame, k: int = 10) -> pd.DataFrame:
        """
        Finds the K-nearest neighbors of anomaly_row in baseline_pool.
        Standardizes numerical features for Euclidean distance matching.
        """
        features = self.parent.features
        
        # Select numeric columns that exist in the features list
        numeric_cols = [c for c in features if pd.api.types.is_numeric_dtype(baseline_pool[c])]
        
        if len(numeric_cols) == 0:
            return baseline_pool.head(k)
            
        # Standardize baseline pool and row
        mean = baseline_pool[numeric_cols].mean()
        std = baseline_pool[numeric_cols].std().replace(0, 1.0)
        
        pool_norm = (baseline_pool[numeric_cols] - mean) / std
        row_norm = (anomaly_row[numeric_cols] - mean) / std
        
        # Calculate Euclidean distances
        distances = np.linalg.norm(pool_norm.values - row_norm.values, axis=1)
        
        # Sort and take top k
        closest_idx = np.argsort(distances)[:k]
        return baseline_pool.iloc[closest_idx]

    def _find_causal_paths(self, start_node: str, target_node: str, edges: List[Tuple[str, str]]) -> List[str]:
        """
        Traces all directed causal paths in the DAG from start_node to target_node using DFS.
        """
        adj = {}
        for u, v in edges:
            if u not in adj:
                adj[u] = []
            adj[u].append(v)
            
        paths = []
        
        def dfs(current: str, path: List[str], visited: set):
            if current == target_node:
                paths.append(" -> ".join(path))
                return
            visited.add(current)
            for neighbor in adj.get(current, []):
                if neighbor not in visited:
                    dfs(neighbor, path + [neighbor], visited.copy())
                    
        dfs(start_node, [start_node], set())
        return paths

    def plot_attribution(self, attribution_report: pd.DataFrame, save_path: Optional[str] = None):
        """
        Plots the cohort-based attribution report with Error Bars (Confidence Intervals).
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib is not installed. Skipping plot generation.")
            return None
            
        # Filter out rows with negligible attribution
        report = attribution_report[attribution_report["Attribution_Mean"].abs() > 1e-6].copy()
        if len(report) == 0:
            print("No significant attribution values to plot.")
            return None
            
        # Sort ascending for horizontal bar chart
        report = report.sort_values(by="Attribution_Mean", ascending=True)
        
        plt.figure(figsize=(10, 6))
        
        colors = ['#ff4d4d' if val < 0 else '#2ecc71' for val in report["Attribution_Mean"].values]
        
        # Plot horizontal bars with xerr (Confidence Intervals)
        bars = plt.barh(
            report.index, 
            report["Attribution_Mean"].values, 
            xerr=report["CI_HalfWidth"].values,
            color=colors, 
            edgecolor='grey', 
            height=0.6,
            error_kw={'ecolor': '#34495e', 'capsize': 4, 'lw': 1.2}
        )
        
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        
        # Add labels beside the bars
        for bar in bars:
            width = bar.get_width()
            label_x = width + (0.05 if width >= 0 else -0.25)
            # Find the CI value for display
            node_name = bar.get_path().vertices[0][1] # Get row name
            # For simplicity, just display mean value
            plt.text(
                label_x, 
                bar.get_y() + bar.get_height()/2, 
                f"{width:+.4f}", 
                va='center', 
                ha='left' if width >= 0 else 'right',
                fontsize=10,
                fontweight='bold',
                color='#2c3e50'
            )
            
        plt.title("WhisperTrace - Cohort-Based Root Cause Analysis (95% CI)", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Average Causal Contribution to Target Change", fontsize=12, labelpad=10)
        plt.ylabel("Causal Nodes / Variables", fontsize=12)
        plt.grid(axis='x', linestyle=':', alpha=0.6)
        
        for spine in ['top', 'right']:
            plt.gca().spines[spine].set_visible(False)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"WhisperTrace Attribution chart saved successfully to: {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None

    def _get_topological_order(self, edges: List[Tuple[str, str]]) -> List[str]:
        """
        Get topological order of nodes from DAG edges.
        """
        adj = {}
        in_degree = {}
        all_nodes = set()
        
        for u, v in edges:
            all_nodes.add(u)
            all_nodes.add(v)
            if u not in adj:
                adj[u] = []
            adj[u].append(v)
            in_degree[v] = in_degree.get(v, 0) + 1
            if u not in in_degree:
                in_degree[u] = 0
                
        # Add target and features
        all_nodes.add(self.parent.target)
        if self.parent.treatment:
            all_nodes.add(self.parent.treatment)
        for f in self.parent.features:
            all_nodes.add(f)
            if f not in in_degree:
                in_degree[f] = 0
            if f not in adj:
                adj[f] = []
                
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
