import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from .causal_trace import CausalTrace, InterventionRecord


class KnowledgeMemory:
    def __init__(self, max_size: int = 10000):
        self.traces: List[CausalTrace] = []
        self.max_size = max_size

    def store(self, trace: CausalTrace):
        if len(self.traces) >= self.max_size:
            self.traces.pop(0)
        self.traces.append(trace)

    def __len__(self) -> int:
        return len(self.traces)

    def find_similar_by_causal_profile(
        self, trace: CausalTrace, k: int = 5
    ) -> List[Tuple[CausalTrace, float]]:
        if not self.traces:
            return []

        query_profile = trace.causal_profile.reshape(1, -1)
        profiles = np.array([t.causal_profile for t in self.traces])

        if query_profile.shape[1] != profiles.shape[1]:
            return []

        dists = np.linalg.norm(profiles - query_profile, axis=1)
        closest_idx = np.argsort(dists)[:k]

        return [(self.traces[i], dists[i]) for i in closest_idx]

    def find_similar_by_features(
        self, trace: CausalTrace, k: int = 5
    ) -> List[Tuple[CausalTrace, float]]:
        if not self.traces:
            return []

        query_vec = trace.features.values.astype(float).reshape(1, -1)
        profiles = np.array([t.features.values.astype(float) for t in self.traces])

        dists = np.linalg.norm(profiles - query_vec, axis=1)
        closest_idx = np.argsort(dists)[:k]

        return [(self.traces[i], dists[i]) for i in closest_idx]

    def get_success_rate(
        self, intervention_pattern: Optional[Dict[str, float]] = None
    ) -> float:
        if not self.traces:
            return 0.0

        relevant = self.traces
        if intervention_pattern:
            relevant = [
                t for t in self.traces
                if any(
                    set(r.plan.keys()) == set(intervention_pattern.keys())
                    for r in t.interventions_tried
                )
            ]

        if not relevant:
            return 0.0

        total_interventions = sum(len(t.interventions_tried) for t in relevant)
        successful = sum(
            sum(1 for r in t.interventions_tried if r.success)
            for t in relevant
        )
        return successful / total_interventions if total_interventions > 0 else 0.0

    def get_most_effective_interventions(self, top_k: int = 5) -> pd.DataFrame:
        records = []
        for t in self.traces:
            for r in t.interventions_tried:
                records.append({
                    "features_hash": str(sorted(r.plan.items())),
                    "plan": r.plan,
                    "delta": r.delta,
                    "success": r.success,
                })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        grouped = df.groupby("features_hash").agg(
            avg_delta=("delta", "mean"),
            success_rate=("success", "mean"),
            count=("delta", "count"),
            example_plan=("plan", "first"),
        ).sort_values("avg_delta", ascending=False)

        return grouped.head(top_k)

    def summarize(self) -> Dict:
        if not self.traces:
            return {"n_traces": 0}

        all_interventions = [
            r for t in self.traces for r in t.interventions_tried
        ]
        n_success = sum(1 for r in all_interventions if r.success)

        return {
            "n_traces": len(self.traces),
            "n_interventions": len(all_interventions),
            "n_successful": n_success,
            "success_rate": n_success / len(all_interventions) if all_interventions else 0.0,
            "avg_delta": np.mean([r.delta for r in all_interventions]) if all_interventions else 0.0,
        }
