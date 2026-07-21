import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple

from .causal_trace import CausalTrace, InterventionRecord
from .knowledge_memory import KnowledgeMemory
from .hypothesis import Hypothesis, HypothesisGenerator


class SCMReasoner:
    def __init__(self, parent, knowledge_memory: Optional[KnowledgeMemory] = None):
        self.parent = parent
        self.memory = knowledge_memory or KnowledgeMemory()
        self.hypothesis_generator = HypothesisGenerator(memory=self.memory)

    def retrieve_similar_cases(
        self, trace: CausalTrace, k: int = 5
    ) -> List[Tuple[CausalTrace, float]]:
        if len(self.memory) == 0:
            return []
        return self.memory.find_similar_by_causal_profile(trace, k=k)

    def generate_hypotheses(
        self, trace: CausalTrace, n_plans: int = 8
    ) -> List[Hypothesis]:
        return self.hypothesis_generator.generate(trace, n_plans=n_plans)

    def simulate_hypothesis(
        self, patient_df: pd.DataFrame, hypothesis: Hypothesis
    ) -> InterventionRecord:
        target = self.parent.target
        original_outcome = patient_df[target].iloc[0] if target in patient_df.columns else None

        if not self.parent.rca.engine.is_fitted:
            self.parent.rca.fit_scm()

        simulated = self.parent.simulate(patient_df, hypothesis.plan)
        simulated_outcome = float(simulated[target].iloc[0]) if target in simulated.columns else 0.0
        delta = simulated_outcome - original_outcome if original_outcome is not None else 0.0
        success = delta < 0

        return InterventionRecord(
            plan=hypothesis.plan,
            simulated_outcome=simulated_outcome,
            delta=delta,
            success=success,
            hypothesis_source=hypothesis.source,
        )

    def simulate_hypotheses(
        self, patient_df: pd.DataFrame, hypotheses: List[Hypothesis]
    ) -> List[Tuple[Hypothesis, InterventionRecord]]:
        results = []
        for h in hypotheses:
            record = self.simulate_hypothesis(patient_df, h)
            results.append((h, record))
        return results

    def rank_hypotheses(
        self, results: List[Tuple[Hypothesis, InterventionRecord]]
    ) -> List[Tuple[Hypothesis, InterventionRecord]]:
        return sorted(results, key=lambda x: x[1].delta)

    def reason(
        self, trace: CausalTrace, n_plans: int = 8
    ) -> Tuple[List[Tuple[Hypothesis, InterventionRecord]], CausalTrace]:
        similar_cases = self.retrieve_similar_cases(trace)

        hypotheses = self.generate_hypotheses(trace, n_plans=n_plans)

        patient_df = pd.DataFrame([trace.features])
        results = self.simulate_hypotheses(patient_df, hypotheses)
        ranked = self.rank_hypotheses(results)

        trace.interventions_tried = [r for _, r in ranked]
        self.memory.store(trace)

        return ranked, trace

    def reason_from_patient(
        self,
        patient_df: pd.DataFrame,
        rca_report: pd.DataFrame,
        cate_series: pd.Series,
        dag_edges: List[tuple],
        n_plans: int = 8,
    ) -> Tuple[List[Tuple[Hypothesis, InterventionRecord]], CausalTrace]:
        target = self.parent.target
        trace = CausalTrace(
            features=patient_df.iloc[0],
            target=target,
            target_value=float(patient_df[target].iloc[0]) if target in patient_df else 0.0,
            rca_scores=rca_report["Attribution_Mean"] if "Attribution_Mean" in rca_report.columns else rca_report.iloc[:, 0],
            cate_estimates=cate_series,
            dag_edges=dag_edges,
        )
        return self.reason(trace, n_plans=n_plans)
