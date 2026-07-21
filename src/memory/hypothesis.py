from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from itertools import combinations
import pandas as pd
import numpy as np

from .causal_trace import CausalTrace
from .knowledge_memory import KnowledgeMemory


@dataclass
class Hypothesis:
    plan: Dict[str, float]
    rationale: str
    source: str
    features_involved: List[str] = field(default_factory=list)

    def summary(self) -> str:
        plan_str = ", ".join(f"{k}={v:.1f}" for k, v in self.plan.items())
        return f"[{self.source}] {plan_str} | {self.rationale}"


class HypothesisGenerator:
    def __init__(self, memory: Optional[KnowledgeMemory] = None):
        self.memory = memory

    @staticmethod
    def _default_modifiable() -> set:
        return {"age", "sex"}

    def get_modifiable_features(
        self, trace: CausalTrace, top_k: int = 10
    ) -> List[str]:
        non_modifiable = self._default_modifiable()
        return [f for f in trace.top_causes(k=top_k) if f not in non_modifiable]

    @staticmethod
    def _has_meaningful_cate(trace: CausalTrace) -> bool:
        return (
            trace.cate_estimates is not None
            and not trace.cate_estimates.empty
            and trace.cate_estimates.abs().max() > 0.01
        )

    def _compute_effect(self, trace: CausalTrace, feat: str) -> float:
        if self._has_meaningful_cate(trace):
            cate = trace.cate_estimates.get(feat, 0.0)
            return cate if not pd.isna(cate) else 0.0

        rca = trace.rca_scores.get(feat, 0.0)
        return rca * 0.5 if not pd.isna(rca) else 0.0

    def _single_feature_hypotheses(
        self, trace: CausalTrace, modifiable: List[str]
    ) -> List[Hypothesis]:
        hypotheses = []
        for feat in modifiable[:5]:
            current = trace.features.get(feat, 0.0)
            if current == 0.0:
                continue

            effect = self._compute_effect(trace, feat)
            if abs(effect) < 0.001:
                continue

            if effect > 0:
                target = current * 0.7
            else:
                target = current * 1.3
            target = max(0.0, target)

            if abs(target - current) / (abs(current) + 1e-8) <= 0.05:
                continue

            rca_val = trace.rca_scores.get(feat, 0.0)
            hypotheses.append(Hypothesis(
                plan={feat: target},
                rationale=f"{feat} là nguyên nhân chính (RCA={rca_val:+.3f}), giảm từ {current:.1f} xuống {target:.1f} để hạ target",
                source="RCA-driven",
                features_involved=[feat],
            ))
        return hypotheses

    def _combined_hypotheses(
        self, trace: CausalTrace, modifiable: List[str], single_hyp: List[Hypothesis]
    ) -> List[Hypothesis]:
        top_features = modifiable[:5]
        hypotheses = []

        for r in range(2, min(4, len(top_features) + 1)):
            for combo in combinations(top_features, r):
                plan = {}
                features_involved = []
                for feat in combo:
                    matching = [h for h in single_hyp if h.plan.get(feat) is not None]
                    if matching:
                        plan[feat] = matching[0].plan[feat]
                        features_involved.append(feat)
                    else:
                        current = trace.features.get(feat, 0.0)
                        effect = self._compute_effect(trace, feat)
                        if abs(effect) > 0.001 and current != 0.0:
                            target = current * (0.7 if effect > 0 else 1.3)
                            plan[feat] = max(0.0, target)
                            features_involved.append(feat)

                if len(plan) < 2:
                    continue

                feat_str = ", ".join(features_involved)
                hypotheses.append(Hypothesis(
                    plan=plan,
                    rationale=f"Kết hợp {feat_str}: tác động cộng hưởng từ nhiều nguyên nhân",
                    source="Combined",
                    features_involved=features_involved,
                ))
        return hypotheses

    def _knowledge_guided_hypotheses(
        self, trace: CausalTrace, n: int = 3
    ) -> List[Hypothesis]:
        if self.memory is None or len(self.memory) == 0:
            return []

        similar = self.memory.find_similar_by_causal_profile(trace, k=n)
        hypotheses = []
        seen = set()

        for neighbor, dist in similar:
            for rec in neighbor.interventions_tried:
                if not rec.success:
                    continue

                key = str(sorted(rec.plan.items()))
                if key in seen:
                    continue
                seen.add(key)

                feat_str = ", ".join(rec.plan.keys())
                hypotheses.append(Hypothesis(
                    plan=rec.plan,
                    rationale=f"Kinh nghiệm từ ca tương tự (khoảng cách causal={dist:.3f}): {feat_str} thành công với delta={rec.delta:+.3f}",
                    source="Knowledge-guided",
                    features_involved=list(rec.plan.keys()),
                ))

        return hypotheses

    def generate(
        self,
        trace: CausalTrace,
        n_plans: int = 8,
    ) -> List[Hypothesis]:
        modifiable = self.get_modifiable_features(trace)
        if not modifiable:
            return []

        single = self._single_feature_hypotheses(trace, modifiable)
        combined = self._combined_hypotheses(trace, modifiable, single)
        knowledge = self._knowledge_guided_hypotheses(trace)

        all_h = single + combined + knowledge
        seen = set()
        unique = []
        for h in all_h:
            key = str(sorted(h.plan.items()))
            if key not in seen:
                seen.add(key)
                unique.append(h)
            if len(unique) >= n_plans:
                break

        return unique
