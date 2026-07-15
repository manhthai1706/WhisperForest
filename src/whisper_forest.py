import pandas as pd
from typing import List, Optional, Union, Dict

from .causal import CausalBranch
from .predictive import PredictiveBranch
from .rca import RCABranch
from .policy import PolicyBranch
from .optimization import OptimizationBranch

class WhisperForest:
    def __init__(
        self,
        data: pd.DataFrame,
        target: str,
        treatment: Optional[str] = None,
        features: Optional[List[str]] = None,
    ):
        """
        WhisperForest: A Causal Decision Engine based on RandomForest.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset.
        target : str
            The outcome/target variable (Y).
        treatment : Optional[str]
            The treatment/intervention variable (T).
        features : Optional[List[str]]
            The list of features (X). If None, all columns except target and treatment are used.
        """
        self.data = data.copy()
        self.target = target
        self.treatment = treatment
        
        if features is None:
            exclude = {target}
            if treatment:
                exclude.add(treatment)
            self.features = [col for col in data.columns if col not in exclude]
        else:
            self.features = features
            
        # Initialize branches
        self.causal = CausalBranch(self)
        self.predictive = PredictiveBranch(self)
        self.rca = RCABranch(self)
        self.trace = self.rca # WhisperTrace alias for Causal RCA
        self.policy = PolicyBranch(self)
        self.optimization = OptimizationBranch(self)

    def save_model(self, file_path: str):
        """
        Serializes and exports the trained models, DAG structure, and metadata to a file.
        Uses HDF5 (.h5) if h5py is installed, otherwise falls back to a standard pickle file.
        """
        import pickle
        import numpy as np
        
        # Build state dict
        state = {
            "predictive_model": self.predictive.modeling.model if hasattr(self.predictive, "modeling") else None,
            "scm_models": self.rca.engine.models if hasattr(self.rca.engine, "models") else None,
            "scm_parents": self.rca.engine.parents_map if hasattr(self.rca.engine, "parents_map") else None,
            "dag_edges": self.causal.get_dag(),
            "metadata": {
                "features": self.features,
                "target": self.target,
                "treatment": self.treatment
            }
        }
        
        try:
            import h5py
            has_h5py = True
        except ImportError:
            has_h5py = False
            
        if has_h5py:
            print("Exporting model pipeline to H5 format using h5py...")
            with h5py.File(file_path, "w") as f:
                for k, v in state.items():
                    if v is not None:
                        f.create_dataset(k, data=np.void(pickle.dumps(v)))
            print(f"WhisperForest model pipeline successfully saved (H5) to: {file_path}")
        else:
            print("h5py is not installed. Falling back to standard pickle serialization...")
            with open(file_path, "wb") as f:
                pickle.dump(state, f)
            print(f"WhisperForest model pipeline successfully saved (Pickle) to: {file_path}")

    def load_model(self, file_path: str):
        """
        Loads the serialized models, DAG structure, and metadata from a file.
        Detects if the file is an HDF5 file or a pickle file.
        """
        import pickle
        import os
        
        is_h5 = False
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                header = f.read(8)
                # HDF5 file signature is \x89HDF\r\n\x1a\n
                if header == b"\x89HDF\r\n\x1a\n":
                    is_h5 = True
                    
        if is_h5:
            try:
                import h5py
            except ImportError:
                raise ImportError("The model file is in H5 format, but the h5py library is not installed in this environment. Please run 'pip install h5py' or delete the file to re-train.")
                
            print("Loading model pipeline from H5 format...")
            with h5py.File(file_path, "r") as f:
                if "predictive_model" in f:
                    self.predictive.modeling.model = pickle.loads(f["predictive_model"][()].tobytes())
                    print("- Loaded predictive model successfully.")
                if "scm_models" in f:
                    self.rca.engine.models = pickle.loads(f["scm_models"][()].tobytes())
                    self.rca.engine.is_fitted = True
                    print("- Loaded SCM equations successfully.")
                if "scm_parents" in f:
                    self.rca.engine.parents_map = pickle.loads(f["scm_parents"][()].tobytes())
                    print("- Loaded SCM parents successfully.")
                if "dag_edges" in f:
                    self.causal.set_dag(pickle.loads(f["dag_edges"][()].tobytes()))
                    print("- Loaded causal DAG successfully.")
                if "metadata" in f:
                    metadata = pickle.loads(f["metadata"][()].tobytes())
                    self.features = metadata["features"]
                    self.target = metadata["target"]
                    self.treatment = metadata["treatment"]
                    print("- Loaded metadata successfully.")
        else:
            print("Loading model pipeline from standard pickle format...")
            with open(file_path, "rb") as f:
                state = pickle.load(f)
            
            if "predictive_model" in state:
                self.predictive.modeling.model = state["predictive_model"]
                print("- Loaded predictive model successfully.")
            if "scm_models" in state:
                self.rca.engine.models = state["scm_models"]
                if self.rca.engine.models is not None:
                    self.rca.engine.is_fitted = True
                print("- Loaded SCM equations successfully.")
            if "scm_parents" in state:
                self.rca.engine.parents_map = state["scm_parents"]
                print("- Loaded SCM parents successfully.")
            if "dag_edges" in state:
                self.causal.set_dag(state["dag_edges"])
                print("- Loaded causal DAG successfully.")
            if "metadata" in state:
                metadata = state["metadata"]
                self.features = metadata["features"]
                self.target = metadata["target"]
                self.treatment = metadata["treatment"]
                print("- Loaded metadata successfully.")
                
        # Reconstruct parents_map from dag_edges if it is missing (backward compatibility)
        if not self.rca.engine.parents_map:
            dag = self.causal.get_dag()
            if dag:
                parents_map = {}
                for u, v in dag:
                    if v not in parents_map:
                        parents_map[v] = []
                    parents_map[v].append(u)
                self.rca.engine.parents_map = parents_map
                print("- Reconstructed SCM parents map from DAG edges.")
                
        print(f"WhisperForest model pipeline successfully loaded from: {file_path}")

    def simulate(self, patient: pd.DataFrame, interventions: Dict[str, float]) -> pd.DataFrame:
        """
        Simulates the counterfactual state of the patient(s) under the given interventions.
        Propagates effects topologically through the Structural Causal Model (SCM).
        """
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        
        if not self.rca.engine.is_fitted:
            self.rca.fit_scm()
            
        noise_df = self.rca.engine.compute_noise(patient)
        topo_order = self.rca.attributor._get_topological_order(self.causal.get_dag())
        
        simulated_df = patient.copy()
        
        for node in topo_order:
            if node in interventions:
                # Intervened value
                simulated_df[node] = interventions[node]
            else:
                parents = self.rca.engine.parents_map.get(node, [])
                if len(parents) > 0:
                    model = self.rca.engine.models.get(node)
                    if model is not None:
                        if isinstance(model, RandomForestClassifier):
                            prob = model.predict_proba(simulated_df[parents])
                            if prob.shape[1] == 2:
                                pred = prob[:, 1]
                            else:
                                pred = np.zeros(len(simulated_df)) if model.classes_[0] == 0 else np.ones(len(simulated_df))
                            
                            if node == self.target:
                                # Outcome node uses continuous probability representation
                                simulated_df[node] = pred + noise_df[node]
                            else:
                                # Intermediate nodes use thresholding to remain strictly binary
                                simulated_df[node] = (pred >= noise_df[node]).astype(int)
                        else:
                            pred = model.predict(simulated_df[parents])
                            simulated_df[node] = pred + noise_df[node]
                    else:
                        simulated_df[node] = noise_df[node]
                else:
                    simulated_df[node] = noise_df[node]
                    
        return simulated_df

    def do(self, patient: pd.DataFrame, interventions: Dict[str, float]) -> pd.DataFrame:
        """
        Alias for simulate(), executing a do-intervention under Pearl's Causal Framework.
        """
        return self.simulate(patient, interventions)

    def counterfactual(self, patient: pd.DataFrame, interventions: Dict[str, float]) -> Union[float, pd.Series]:
        """
        Computes the counterfactual risk difference (Y_simulated - Y_original) for the given patient(s).
        """
        simulated_df = self.simulate(patient, interventions)
        target = self.target
        
        if target not in patient.columns:
            # Estimate baseline risk using SCM propagation without intervention
            baseline_df = self.simulate(patient, {})
            y_original = baseline_df[target]
        else:
            y_original = patient[target]
            
        diff = simulated_df[target] - y_original
        if len(diff) == 1:
            return float(diff.iloc[0])
        return diff

    def evaluate_causal_consistency(self, patient: pd.DataFrame, interventions: Dict[str, float]) -> Dict[str, Union[float, str]]:
        """
        Evaluates the consistency of treatment effect estimates across the three layers:
        1. Predictive Layer (association-based static risk change)
        2. SCM Structural Layer (counterfactual simulation risk change)
        3. Causal DML Layer (CATE treatment effect estimation)
        
        Returns a dictionary containing the estimated effects, a consistency score, and a diagnostic status.
        """
        import numpy as np
        
        treatment = self.treatment
        if treatment not in interventions:
            raise ValueError(f"Interventions dict must contain the active treatment variable: '{treatment}'")
            
        val_treated = interventions[treatment]
        val_untreated = 1 - val_treated if val_treated in [0, 1] else 0.0
        
        # 1. Predictive Effect
        patient_untreated = patient.copy()
        patient_untreated[treatment] = val_untreated
        patient_treated = patient.copy()
        patient_treated[treatment] = val_treated
        
        pred_prob_untreated = self.predictive.predict_proba(patient_untreated)[0, 1]
        pred_prob_treated = self.predictive.predict_proba(patient_treated)[0, 1]
        e_pred = pred_prob_treated - pred_prob_untreated
        
        # 2. SCM Effect
        scm_untreated = self.simulate(patient, {treatment: val_untreated})
        scm_treated = self.simulate(patient, {treatment: val_treated})
        e_scm = scm_treated[self.target].iloc[0] - scm_untreated[self.target].iloc[0]
        
        # 3. DML Effect (CATE)
        cate_df = self.causal.estimate_cate(patient)
        cate_val = cate_df["CATE"].iloc[0]
        e_dml = cate_val if val_treated == 1 else -cate_val
        
        # Calculate consistency score (0 to 1)
        effects = [e_pred, e_scm, e_dml]
        mean_abs = np.mean([abs(x) for x in effects])
        
        if mean_abs < 1e-3:
            consistency_score = 1.0
        else:
            std_dev = np.std(effects)
            # Higher standard deviation relative to mean magnitude reduces consistency
            consistency_score = max(0.0, 1.0 - (std_dev / (mean_abs + 1e-5)))
            
        # Determine diagnostic status
        if consistency_score >= 0.75:
            status = "HIGH CONSISTENCY (All reasoning layers agree on direction and magnitude)"
        elif consistency_score >= 0.4:
            status = "MODERATE CONSISTENCY (Some layers show weak or partial agreement)"
        else:
            # Check for Confounded Treatment Signal (Simpson's Paradox / Confounding by Indication)
            # If DML shows a significant treatment effect but Predictive/SCM layers show close to zero or positive
            is_dml_effective = (e_dml < -0.02)
            is_pred_null = (abs(e_pred) < 0.01)
            is_scm_null = (abs(e_scm) < 0.01)
            
            if is_dml_effective and is_pred_null and is_scm_null:
                status = (
                    "CONFOUNDED TREATMENT SIGNAL DETECTED: "
                    "Observed association (Predictive/SCM) disagrees with adjusted causal estimate (DML). "
                    "Potential selection bias or strong confounding by indication (Simpson's Paradox) is masking the true treatment effect in standard models."
                )
            else:
                status = "LOW CONSISTENCY (Treatment effect estimate is NOT supported by SCM structural simulation or Predictive association)"
            
        return {
            "Predictive_Effect": float(e_pred),
            "SCM_Effect": float(e_scm),
            "DML_Effect": float(e_dml),
            "Consistency_Score": float(consistency_score),
            "Status": status
        }






