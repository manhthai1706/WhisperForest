from dataclasses import dataclass, field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np


_MODIFIABLE_DEFAULTS = {"age", "sex"}


@dataclass
class InterventionRecord:
    plan: Dict[str, float]
    simulated_outcome: float
    delta: float
    success: bool
    hypothesis_source: str = ""


@dataclass
class CausalTrace:
    features: pd.Series
    target: str
    target_value: float
    rca_scores: pd.Series
    cate_estimates: pd.Series
    interventions_tried: List[InterventionRecord] = field(default_factory=list)
    dag_edges: List[tuple] = field(default_factory=list)
    patient_id: Optional[str] = None
    modifiable_features: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        return {
            "patient_id": self.patient_id,
            "features": self.features.to_dict(),
            "target": self.target,
            "target_value": self.target_value,
            "rca_scores": self.rca_scores.to_dict(),
            "cate_estimates": self.cate_estimates.to_dict(),
            "interventions_tried": [
                {"plan": r.plan, "simulated_outcome": r.simulated_outcome,
                 "delta": r.delta, "success": r.success,
                 "hypothesis_source": r.hypothesis_source}
                for r in self.interventions_tried
            ],
            "dag_edges": self.dag_edges,
            "modifiable_features": self.modifiable_features,
        }

    def top_causes(self, k: int = 5) -> List[str]:
        return self.rca_scores.abs().sort_values(ascending=False).head(k).index.tolist()

    def get_modifiable(self) -> List[str]:
        if self.modifiable_features is not None:
            return self.modifiable_features
        return [f for f in self.top_causes(k=10) if f not in _MODIFIABLE_DEFAULTS]

    @property
    def best_intervention(self) -> Optional[InterventionRecord]:
        if not self.interventions_tried:
            return None
        return max(self.interventions_tried, key=lambda r: r.delta)

    @property
    def success_rate(self) -> float:
        if not self.interventions_tried:
            return 0.0
        return sum(r.success for r in self.interventions_tried) / len(self.interventions_tried)

    @property
    def causal_profile(self) -> np.ndarray:
        rca = self.rca_scores.reindex(self.rca_scores.index, fill_value=0.0)
        cate = self.cate_estimates.reindex(rca.index, fill_value=0.0)
        profile = pd.DataFrame({"rca": rca, "cate": cate}).fillna(0.0)
        return profile.values.flatten()
