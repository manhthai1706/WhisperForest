import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from sklearn.ensemble import RandomForestClassifier

class CausalAuditor:
    def __init__(self, branch):
        self.branch = branch
        self.parent = branch.parent

    def audit(self, patient: pd.DataFrame, interventions: Dict[str, float] = None, threshold_conflict: float = 0.01) -> Dict:
        """
        Audits the causal estimation across the reasoning layers.

        - If a treatment variable is defined and included in interventions:
            Compares Predictive, SCM, and DML effects → full 3-layer audit.
        - If no treatment variable (pure observational data):
            Compares Predictive vs SCM only → 2-layer audit.
        """
        treatment = self.parent.treatment
        target = self.parent.target
        interventions = interventions or {}

        has_treatment = treatment and treatment in interventions

        # ─── 1. Predictive Layer ─────────────────────────────────────────────
        is_classifier = isinstance(self.parent.predictive.modeling.model, RandomForestClassifier)

        if has_treatment:
            val_treated = interventions[treatment]
            val_untreated = 1 - val_treated if val_treated in [0, 1] else 0.0

            patient_untreated = patient.copy()
            patient_untreated[treatment] = val_untreated
            patient_treated = patient.copy()
            patient_treated[treatment] = val_treated

            if is_classifier:
                pred_untreated = self.parent.predictive.predict_proba(patient_untreated)[0, 1]
                pred_treated = self.parent.predictive.predict_proba(patient_treated)[0, 1]
            else:
                pred_untreated = self.parent.predictive.predict(patient_untreated)[0]
                pred_treated = self.parent.predictive.predict(patient_treated)[0]

            e_pred = pred_treated - pred_untreated
        else:
            # No treatment: use predictive model's raw output as baseline risk
            if is_classifier:
                e_pred = float(self.parent.predictive.predict_proba(patient)[0, 1])
            else:
                e_pred = float(self.parent.predictive.predict(patient)[0])

        # ─── 2. SCM Structural Layer ─────────────────────────────────────────
        if has_treatment:
            scm_untreated = self.parent.simulate(patient, {treatment: val_untreated})
            scm_treated = self.parent.simulate(patient, {treatment: val_treated})
            e_scm = float(scm_treated[target].iloc[0] - scm_untreated[target].iloc[0])
        else:
            # No treatment: simulate with no interventions (identity check)
            sim = self.parent.simulate(patient, {})
            e_scm = float(sim[target].iloc[0])

        # 2.1 SCM Mixture Layer Effect (optional)
        e_scm_mix = None
        if hasattr(self.parent.rca, "mixture_engine") and self.parent.rca.mixture_engine.is_fitted:
            if has_treatment:
                mix_untreated = self.parent.simulate_mixture(patient, {treatment: val_untreated})
                mix_treated = self.parent.simulate_mixture(patient, {treatment: val_treated})
                e_scm_mix = float(mix_treated[target].iloc[0] - mix_untreated[target].iloc[0])
            else:
                mix_sim = self.parent.simulate_mixture(patient, {})
                e_scm_mix = float(mix_sim[target].iloc[0])

        # ─── 3. DML Layer (only when treatment is available) ─────────────────
        e_dml = None
        if has_treatment:
            cate_df = self.parent.causal.estimate_cate(patient)
            cate_val = cate_df["CATE"].iloc[0]
            e_dml = float(cate_val if val_treated == 1 else -cate_val)

        # ─── 4. Consistency Score ────────────────────────────────────────────
        effects = [e_pred, e_scm]
        if e_dml is not None:
            effects.append(e_dml)

        mean_abs = np.mean([abs(x) for x in effects])
        if mean_abs < 1e-3:
            consistency_score = 1.0
        else:
            std_dev = np.std(effects)
            consistency_score = max(0.0, 1.0 - (std_dev / (mean_abs + 1e-5)))

        # ─── 5. Diagnosis ─────────────────────────────────────────────────────
        warnings = []
        diagnoses = []

        if e_dml is not None:
            # Simpson's Paradox check (only when DML is available)
            if e_dml < -threshold_conflict and e_pred >= -1e-3:
                warnings.append("SIMPSON'S PARADOX DETECTED")
                diagnoses.append(
                    "Observational data (Predictive/SCM) shows neutral/harmful association "
                    "due to confounders, but DML reveals a true beneficial treatment effect."
                )
            # Directional conflict: SCM vs DML
            if abs(e_dml) > threshold_conflict and abs(e_scm) > threshold_conflict:
                if np.sign(e_dml) != np.sign(e_scm):
                    warnings.append("DIRECTIONAL CONFLICT")
                    diagnoses.append(
                        f"SCM predicts {e_scm:+.4f} but DML predicts {e_dml:+.4f}. "
                        "DAG may have incorrect edge orientations or missing confounders."
                    )
        else:
            # No treatment: check Predictive vs SCM agreement
            if abs(e_pred - e_scm) > 0.15:
                warnings.append("PREDICTIVE / SCM DIVERGENCE")
                diagnoses.append(
                    f"Predictive model outputs {e_pred:.4f} but SCM structural simulation outputs {e_scm:.4f}. "
                    "The causal DAG may not fully explain the predictive model's learned associations."
                )

        # ─── 6. Safety Status ─────────────────────────────────────────────────
        if consistency_score >= 0.75 and len(warnings) == 0:
            status = "SAFE (All reasoning layers agree on direction and magnitude)"
        elif consistency_score >= 0.40 and len(warnings) == 0:
            status = "CAUTION (Moderate consistency, minor variance in effect magnitudes)"
        else:
            status = "UNSAFE / HIGH BIAS (Significant conflicts or selection bias detected across layers)"

        report = {
            "Predictive_Effect": float(e_pred),
            "SCM_Effect": float(e_scm),
            "SCM_Mixture_Effect": e_scm_mix,
            "DML_Effect": e_dml,
            "Has_Treatment": has_treatment,
            "Consistency_Score": float(consistency_score),
            "Safety_Status": status,
            "Warnings": warnings,
            "Diagnoses": diagnoses
        }
        return report

    def format_report(self, report: Dict) -> str:
        """
        Formats the audit report dict into a readable console output.
        """
        lines = []
        lines.append("=" * 65)
        lines.append(f"{'WhisperForest Causal Auditor Report':^65}")
        lines.append("=" * 65)

        has_treatment = report.get("Has_Treatment", True)
        if has_treatment:
            lines.append(f"Predictive Layer Effect:       {report['Predictive_Effect'] * 100:+.2f}%")
            lines.append(f"SCM Structural Layer Effect:   {report['SCM_Effect'] * 100:+.2f}%")
            if report["SCM_Mixture_Effect"] is not None:
                lines.append(f"MoSCM Expert Layer Effect:     {report['SCM_Mixture_Effect'] * 100:+.2f}%")
            if report["DML_Effect"] is not None:
                lines.append(f"Causal DML Layer Effect:       {report['DML_Effect'] * 100:+.2f}%")
        else:
            lines.append(f"Predictive Risk Score:         {report['Predictive_Effect']:.4f}")
            lines.append(f"SCM Structural Risk Score:     {report['SCM_Effect']:.4f}")
            if report["SCM_Mixture_Effect"] is not None:
                lines.append(f"MoSCM Expert Risk Score:       {report['SCM_Mixture_Effect']:.4f}")
            lines.append("(No treatment variable — DML layer skipped)")

        lines.append("-" * 65)
        lines.append(f"Consistency Score:             {report['Consistency_Score']:.4f}")
        lines.append(f"Safety Status:                 {report['Safety_Status']}")
        lines.append("-" * 65)

        if len(report["Warnings"]) > 0:
            lines.append("WARNINGS:")
            for w in report["Warnings"]:
                lines.append(f"  [!] {w}")
            lines.append("AUDIT DIAGNOSIS:")
            for d in report["Diagnoses"]:
                lines.append(f"  - {d}")
        else:
            lines.append("AUDIT DIAGNOSIS:")
            lines.append("  - No conflicts detected. Reasoning layers are consistent.")
        lines.append("=" * 65)
        return "\n".join(lines)
