from typing import List, Tuple, Dict, Optional, Any
import pandas as pd
import numpy as np

try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.graph.Endpoint import Endpoint
    HAS_CAUSAL_LEARN = True
except ImportError:
    HAS_CAUSAL_LEARN = False

class CausalDiscovery:
    def __init__(self, branch):
        self.branch = branch
        self.parent = branch.parent
        self._dag = None

    def set_dag(self, edges: List[Tuple[str, str]]):
        """
        Set a manually specified DAG.
        """
        self._dag = edges

    def discover(
        self, 
        method: str = "pc", 
        constraints: Optional[Dict[str, Any]] = None,
        bootstrap_runs: int = 20,
        confidence_threshold: float = 0.35
    ) -> List[Tuple[str, str]]:
        """
        Discover the causal graph (DAG) from data using Bootstrap Stability Selection.
        If a manual DAG is already set, returns that.
        
        Parameters:
        -----------
        method : str
            The causal discovery method ('pc', 'lingam', etc.)
        constraints : Optional[Dict]
            Ràng buộc chứa 'whitelist', 'blacklist' và 'tiers' của các cạnh.
        bootstrap_runs : int
            Number of bootstrap samples to run for stability selection.
        confidence_threshold : float
            Minimum selection frequency to retain an edge.
        """
        if self._dag is not None:
            return self._dag
            
        data = self.parent.data
        target = self.parent.target
        treatment = self.parent.treatment
        features = self.parent.features
        
        # Collect all variables in play
        all_vars = features.copy()
        if treatment and treatment not in all_vars:
            all_vars.append(treatment)
        if target not in all_vars:
            all_vars.append(target)
            
        blacklist = list(constraints.get("blacklist", [])) if constraints else []
        whitelist = list(constraints.get("whitelist", [])) if constraints else []
        
        # Process temporal tiers if specified
        if constraints and "tiers" in constraints:
            tiers = constraints["tiers"]
            var_to_tier = {}
            for tier_idx, vars_in_tier in tiers.items():
                for var in vars_in_tier:
                    var_to_tier[var] = int(tier_idx)
            
            # Add reverse-tier edges to the blacklist
            for u in all_vars:
                for v in all_vars:
                    if u in var_to_tier and v in var_to_tier:
                        if var_to_tier[u] > var_to_tier[v]:
                            blacklist.append((u, v))
                            
        # Run bootstrap stability selection
        print(f"Running Bootstrap Stability Selection with {bootstrap_runs} runs using {method.upper()} method...")
        edge_counts = {}
        
        for run in range(bootstrap_runs):
            # Draw bootstrap sample (resampled with replacement)
            boot_data = data.sample(frac=1.0, replace=True, random_state=42 + run)
            discovered_edges_run = []
            
            # Sub-methods for voting/ensemble
            edges_pc = []
            edges_lingam = []
            edges_heuristic = []
            
            # 1. Run PC if requested/ensemble
            run_pc = (method.lower() in ["pc", "ensemble"]) and HAS_CAUSAL_LEARN
            if run_pc:
                df_sub = boot_data[all_vars].dropna()
                data_np = df_sub.to_numpy()
                try:
                    cg = pc(data_np, alpha=0.05, node_names=all_vars, show_progress=False)
                    for edge in cg.G.get_graph_edges():
                        node1 = edge.get_node1().get_name()
                        node2 = edge.get_node2().get_name()
                        endpoint1 = edge.get_endpoint1()
                        endpoint2 = edge.get_endpoint2()
                        if endpoint1 == Endpoint.TAIL and endpoint2 == Endpoint.ARROW:
                            edges_pc.append((node1, node2))
                        elif endpoint1 == Endpoint.ARROW and endpoint2 == Endpoint.TAIL:
                            edges_pc.append((node2, node1))
                        elif endpoint1 == Endpoint.TAIL and endpoint2 == Endpoint.TAIL:
                            if node2 == target or node1 == treatment:
                                edges_pc.append((node1, node2))
                            else:
                                edges_pc.append((node2, node1))
                except Exception:
                    pass
            
            # 2. Run DirectLiNGAM if requested/ensemble
            run_lingam = method.lower() in ["lingam", "ensemble"]
            if run_lingam:
                try:
                    edges_lingam = self._run_lingam_discovery(boot_data, all_vars)
                except Exception:
                    pass
                    
            # 3. Run Heuristic (correlation-based) if requested/ensemble/fallback
            run_heuristic = method.lower() in ["heuristic", "ensemble"] or (method.lower() == "pc" and not HAS_CAUSAL_LEARN)
            if run_heuristic:
                self._run_heuristic_discovery(edges_heuristic, all_vars, treatment, target, blacklist, boot_data)
                
            # Combine based on selected method
            if method.lower() == "ensemble":
                # Ensemble voting: edge passes if voted by at least 2 algorithms (or 1 if causal-learn missing)
                all_discovered = set(edges_pc + edges_lingam + edges_heuristic)
                min_votes = 2 if HAS_CAUSAL_LEARN else 2  # require at least 2 votes to be robust
                
                for edge in all_discovered:
                    votes = 0
                    if edge in edges_pc: votes += 1
                    if edge in edges_lingam: votes += 1
                    if edge in edges_heuristic: votes += 1
                    
                    if votes >= min_votes:
                        discovered_edges_run.append(edge)
            elif method.lower() == "lingam":
                discovered_edges_run = edges_lingam
            elif method.lower() == "pc":
                discovered_edges_run = edges_pc if HAS_CAUSAL_LEARN else edges_heuristic
            else:
                discovered_edges_run = edges_heuristic
                
            # Filter loops/blacklist for this run
            filtered_run = []
            for u, v in discovered_edges_run:
                if (u, v) not in blacklist and u != v:
                    if (u, v) not in filtered_run:
                        filtered_run.append((u, v))
                        
            # Accumulate edge frequency
            for edge in filtered_run:
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
                
        # Calculate edge confidence scores
        print("\nDiscovered Causal Edges Confidence Scores:")
        confident_edges = []
        
        # Whitelisted edges are always kept with confidence 1.0
        for edge in whitelist:
            if edge not in confident_edges:
                confident_edges.append(edge)
                print(f"  {edge[0]} -> {edge[1]}: 1.0000 (WHITELISTED)")
                
        # Sort all candidates by count descending
        sorted_candidates = sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)
        for edge, count in sorted_candidates:
            confidence = count / bootstrap_runs
            # Skip if already whitelisted
            if edge in whitelist:
                continue
            if confidence >= confidence_threshold:
                if edge not in confident_edges:
                    confident_edges.append(edge)
                print(f"  {edge[0]} -> {edge[1]}: {confidence:.4f} (PASSED)")
            else:
                print(f"  {edge[0]} -> {edge[1]}: {confidence:.4f} (DROPPED)")
                
        # Enforce acyclicity (break cycles) on the final selected edges
        final_edges = self._break_cycles(confident_edges, all_vars)
        
        self._dag = final_edges
        return self._dag

    def _break_cycles(self, edges: List[Tuple[str, str]], all_vars: List[str]) -> List[Tuple[str, str]]:
        """
        Detects directed cycles and breaks them by removing the edge with the lowest correlation.
        This guarantees the returned graph is a DAG.
        """
        data = self.parent.data
        corr_matrix = data[all_vars].corr().abs()
        
        edges_set = set(edges)
        
        while True:
            # Build adjacency list
            adj = {}
            for u, v in edges_set:
                if u not in adj:
                    adj[u] = []
                adj[u].append(v)
                
            visited = {}
            path = []
            
            def dfs(u):
                visited[u] = 1  # visiting
                path.append(u)
                for v in adj.get(u, []):
                    if visited.get(v, 0) == 1:
                        cycle_start_idx = path.index(v)
                        return path[cycle_start_idx:] + [v]
                    elif visited.get(v, 0) == 0:
                        cycle = dfs(v)
                        if cycle:
                            return cycle
                path.pop()
                visited[u] = 2  # visited
                return None
                
            detected_cycle = None
            for n in all_vars:
                if visited.get(n, 0) == 0:
                    detected_cycle = dfs(n)
                    if detected_cycle:
                        break
                        
            if not detected_cycle:
                break  # No cycles left!
                
            # Reconstruct the edges of this cycle
            cycle_edges = []
            for i in range(len(detected_cycle) - 1):
                cycle_edges.append((detected_cycle[i], detected_cycle[i+1]))
                
            # Find the edge in the cycle with the lowest absolute correlation
            weakest_edge = None
            min_corr = float('inf')
            
            for u, v in cycle_edges:
                corr_val = corr_matrix.loc[u, v] if (u in corr_matrix.index and v in corr_matrix.columns) else 0.0
                if corr_val < min_corr:
                    min_corr = corr_val
                    weakest_edge = (u, v)
                    
            if weakest_edge:
                print(f"WhisperTrace Causal Discovery: Cycle detected {' -> '.join(detected_cycle)}. Breaking cycle by removing weakest edge: {weakest_edge[0]} -> {weakest_edge[1]} (corr={min_corr:.4f})")
                edges_set.remove(weakest_edge)
                
        return list(edges_set)

    def _run_heuristic_discovery(self, discovered_edges: List[Tuple[str, str]], all_vars: List[str], 
                                 treatment: Optional[str], target: str, blacklist: List[Tuple[str, str]],
                                 data: pd.DataFrame):
        """
        Fallback heuristic based on correlation and causal tiers.
        """
        corr_matrix = data[all_vars].corr().abs()
        
        # Build edges based on correlation threshold and chronological order
        # Chronological order tiers: Features -> Treatment -> Target
        for i in range(len(all_vars)):
            for j in range(i + 1, len(all_vars)):
                v1, v2 = all_vars[i], all_vars[j]
                if corr_matrix.loc[v1, v2] > 0.15:
                    # Decide direction
                    if v1 == target or v2 == target:
                        # Orient towards target
                        src = v1 if v2 == target else v2
                        dest = target
                    elif treatment and (v1 == treatment or v2 == treatment):
                        # Orient towards treatment
                        src = v1 if v2 == treatment else v2
                        dest = treatment
                    else:
                        # Arbitrary direction based on name order to be deterministic
                        src, dest = (v1, v2) if v1 < v2 else (v2, v1)
                        
                    edge = (src, dest)
                    if edge not in discovered_edges and edge not in blacklist:
                        discovered_edges.append(edge)

    def _run_lingam_discovery(self, data: pd.DataFrame, variables: List[str]) -> List[Tuple[str, str]]:
        """
        DirectLiNGAM algorithm implementation in pure Python/NumPy.
        Finds the causal ordering by identifying exogenous variables recursively.
        """
        from sklearn.linear_model import LinearRegression
        import numpy as np
        
        X = data[variables].to_numpy()
        n_features = X.shape[1]
        
        # Track the active features and their indices
        active_idx = list(range(n_features))
        causal_order = []
        
        # Step 1: Find causal ordering
        X_residual = X.copy().astype(float)
        for step in range(n_features - 1):
            # Calculate independence score for each candidate exogenous variable
            best_candidate = None
            best_score = float('inf')
            
            for i in active_idx:
                score = 0.0
                for j in active_idx:
                    if i == j:
                        continue
                    # Fit linear regression of X_j on X_i to get residuals
                    x_i = X_residual[:, [i]]
                    x_j = X_residual[:, j]
                    reg = LinearRegression().fit(x_i, x_j)
                    residual = x_j - reg.predict(x_i)
                    # Compute independence statistic: correlation coefficient (simple proxy)
                    corr = np.abs(np.corrcoef(X_residual[:, i], residual)[0, 1])
                    if np.isnan(corr):
                        corr = 0.0
                    score += corr
                    
                if score < best_score:
                    best_score = score
                    best_candidate = i
                    
            # Append exogenous variable to order
            causal_order.append(best_candidate)
            active_idx.remove(best_candidate)
            
            # Project remaining variables onto orthogonal complement of the selected variable
            x_exog = X_residual[:, [best_candidate]]
            for j in active_idx:
                reg = LinearRegression().fit(x_exog, X_residual[:, j])
                X_residual[:, j] = X_residual[:, j] - reg.predict(x_exog)
                
        # Add the last remaining variable
        causal_order.append(active_idx[0])
        
        # Step 2: Construct DAG using linear regression on preceding variables in order
        edges = []
        for idx_in_order, child_idx in enumerate(causal_order):
            if idx_in_order == 0:
                continue
            # Predecessors in causal order
            parents_indices = causal_order[:idx_in_order]
            X_parents = X[:, parents_indices]
            y_child = X[:, child_idx]
            
            # Fit regression of child on all preceding variables in the causal order
            reg = LinearRegression().fit(X_parents, y_child)
            
            # Standardized coefficient threshold pruning
            std_y = np.std(y_child)
            for parent_pos, parent_idx in enumerate(parents_indices):
                coef = reg.coef_[parent_pos]
                std_x = np.std(X[:, parent_idx])
                std_coef = coef * std_x / (std_y + 1e-9)
                if np.abs(std_coef) > 0.15:
                    edges.append((variables[parent_idx], variables[child_idx]))
                        
        return edges
