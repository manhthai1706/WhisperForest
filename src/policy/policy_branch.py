import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from typing import Union, List, Optional, Dict, Tuple

from ..memory import CausalTrace, InterventionRecord, KnowledgeMemory


class PolicyBranch:
    def __init__(self, parent):
        self.parent = parent
        self.experiment_log: List[Dict] = []
        self.memory: Optional[KnowledgeMemory] = None
        self._policy_model = None

    def log_experiment(self, trial_dict: Dict):
        self.experiment_log.append(trial_dict)

    def set_memory(self, memory: KnowledgeMemory):
        self.memory = memory

    def learn_from_traces(
        self, traces: List[CausalTrace], max_depth: int = 4
    ) -> Optional[DecisionTreeClassifier]:
        if not traces:
            return None

        self.memory = self.memory or KnowledgeMemory()
        for t in traces:
            self.memory.store(t)

        rows = []
        for trace in traces:
            for rec in trace.interventions_tried:
                row = trace.features.to_dict()
                for feat, val in rec.plan.items():
                    row[f"_interv_{feat}"] = val
                row["_success"] = int(rec.success)
                row["_delta"] = rec.delta
                rows.append(row)

        if not rows:
            return None

        df = pd.DataFrame(rows)
        feature_cols = [c for c in df.columns if not c.startswith("_")]
        X = df[feature_cols].fillna(0)
        y = df["_success"]

        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42, class_weight="balanced")
        tree.fit(X, y)
        self._policy_model = tree
        return tree

    def recommend_from_trace(
        self, patient_trace: CausalTrace, k_similar: int = 5
    ) -> List[Tuple[Dict[str, float], float]]:
        if self.memory is None or len(self.memory) == 0:
            return []

        similar = self.memory.find_similar_by_causal_profile(patient_trace, k=k_similar)

        intervention_scores: Dict[str, Dict] = {}
        for neighbor, _ in similar:
            for rec in neighbor.interventions_tried:
                key = str(sorted(rec.plan.items()))
                if key not in intervention_scores:
                    intervention_scores[key] = {
                        "plan": rec.plan,
                        "deltas": [],
                        "successes": [],
                    }
                intervention_scores[key]["deltas"].append(rec.delta)
                intervention_scores[key]["successes"].append(int(rec.success))

        scored = []
        for key, data in intervention_scores.items():
            avg_delta = np.mean(data["deltas"])
            success_rate = np.mean(data["successes"])
            score = -avg_delta * 0.4 + success_rate * 0.6
            scored.append((data["plan"], score, avg_delta, success_rate))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [(s[0], s[1]) for s in scored]

    @staticmethod
    def format_recommendations(
        recommendations: List[Tuple[Dict[str, float], float]],
        top_k: int = 5,
    ) -> str:
        if not recommendations:
            return "  No recommendations available."

        lines = []
        lines.append(f"  {'Rank':<6} {'Intervention Plan':<50} {'Score':>8}")
        lines.append("  " + "-" * 66)
        for i, (plan, score) in enumerate(recommendations[:top_k], 1):
            plan_str = ", ".join(f"{k}={v:.1f}" for k, v in plan.items())
            lines.append(f"  #{i:<5} {plan_str:<50} {score:>8.3f}")
        return "\n".join(lines)

    def distill_policy_from_experiments(self, max_depth: int = 3) -> Optional[DecisionTreeClassifier]:
        if len(self.experiment_log) == 0:
            print("No trials logged in the policy experiment buffer yet.")
            return None

        df_log = pd.DataFrame(self.experiment_log)

        train_rows = []
        for _, trial in df_log.iterrows():
            pat_idx = int(trial["patient_idx"])
            if 0 <= pat_idx < len(self.parent.data):
                patient_row = self.parent.data.iloc[pat_idx].copy()
            else:
                patient_row = pd.Series(0, index=self.parent.features + [self.parent.target]).copy()
                patient_row = patient_row.drop(self.parent.target, errors="ignore")

            orig = trial.get("original_outcome")
            sim = trial.get("simulated_outcome")
            tgt = trial.get("outcome_target")
            success = 0
            if orig is not None and sim is not None and tgt is not None:
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
        n_samples = len(cate_series)
        n_treat = int(np.floor(budget_fraction * n_samples))

        actions = pd.Series(0, index=cate_series.index, name="constrained_action")

        if n_treat <= 0:
            return actions

        if minimize_outcome:
            top_indices = cate_series.nsmallest(n_treat).index
        else:
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
        features = self.parent.features
        X_input = X[features]

        optimal_actions = self.recommend_actions(cate_series, cost=cost, minimize_outcome=minimize_outcome).values

        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        tree.fit(X_input, optimal_actions)

        print(f"Policy tree trained successfully (depth={max_depth}).")
        return tree
