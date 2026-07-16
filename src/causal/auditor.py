import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.ensemble import RandomForestClassifier

class CausalAuditor:
    def __init__(self, branch):
        self.branch = branch
        self.parent = branch.parent

    def audit(self, patient: pd.DataFrame, interventions: Dict[str, float], threshold_conflict: float = 0.01) -> Dict:
        """
        Audits the causal estimation by comparing predictions, simulations, and DML effects.
        Identifies Simpson's paradox, directional conflicts, and assigns a safety status.
        """
        treatment = self.parent.treatment
        target = self.parent.target
        
        if not treatment:
            raise ValueError("Treatment variable must be specified for Causal Auditing.")
        if treatment not in interventions:
            raise ValueError(f"Interventions dict must contain the active treatment variable: '{treatment}'")
            
        val_treated = interventions[treatment]
        val_untreated = 1 - val_treated if val_treated in [0, 1] else 0.0
        
        # 1. Predictive Layer Effect
        patient_untreated = patient.copy()
        patient_untreated[treatment] = val_untreated
        patient_treated = patient.copy()
        patient_treated[treatment] = val_treated
        
        is_classifier = isinstance(self.parent.predictive.modeling.model, RandomForestClassifier)
        if is_classifier:
            pred_untreated = self.parent.predictive.predict_proba(patient_untreated)[0, 1]
            pred_treated = self.parent.predictive.predict_proba(patient_treated)[0, 1]
        else:
            pred_untreated = self.parent.predictive.predict(patient_untreated)[0]
            pred_treated = self.parent.predictive.predict(patient_treated)[0]
            
        e_pred = pred_treated - pred_untreated
        
        # 2. SCM Structural Layer Effect
        scm_untreated = self.parent.simulate(patient, {treatment: val_untreated})
        scm_treated = self.parent.simulate(patient, {treatment: val_treated})
        e_scm = scm_treated[target].iloc[0] - scm_untreated[target].iloc[0]
        
        # 2.1 SCM Mixture Layer Effect (optional)
        e_scm_mix = None
        if hasattr(self.parent.rca, "mixture_engine") and self.parent.rca.mixture_engine.is_fitted:
            mix_untreated = self.parent.simulate_mixture(patient, {treatment: val_untreated})
            mix_treated = self.parent.simulate_mixture(patient, {treatment: val_treated})
            e_scm_mix = mix_treated[target].iloc[0] - mix_untreated[target].iloc[0]
            
        # 3. Causal DML Layer Effect
        cate_df = self.parent.causal.estimate_cate(patient)
        cate_val = cate_df["CATE"].iloc[0]
        e_dml = cate_val if val_treated == 1 else -cate_val
        
        # Calculate consistency score (0 to 1)
        effects = [e_pred, e_scm, e_dml]
        mean_abs = np.mean([abs(x) for x in effects])
        
        if mean_abs < 1e-3:
            consistency_score = 1.0
        else:
            std_dev = np.std(effects)
            consistency_score = max(0.0, 1.0 - (std_dev / (mean_abs + 1e-5)))
            
        # Diagnosis
        warnings = []
        diagnoses = []
        
        # Check for Simpson's Paradox (e.g. DML is beneficial, but SCM or Pred are harmful/null)
        # Note: If target is heart disease or price decrease, "beneficial" treatment means reducing target (negative effect)
        # We can detect if the DML effect is negative (beneficial reduction) but SCM or Pred are positive (harmful increase)
        is_dml_beneficial = e_dml < -threshold_conflict
        is_pred_harmful_or_null = e_pred >= -1e-3
        is_scm_harmful_or_null = e_scm >= -1e-3
        
        if is_dml_beneficial and (is_pred_harmful_or_null or is_scm_harmful_or_null):
            warnings.append("SIMPSON'S PARADOX DETECTED")
            diagnoses.append(
                "Potential Selection Bias / Confounding by Indication: "
                "Observational data shows neutral/harmful association (Predictive/SCM) due to confounders, "
                "but the adjusted causal estimator (DML) reveals a true beneficial treatment effect."
            )
            
        # Check for Directional Conflict
        # If DML and SCM disagree on the sign of the effect
        if abs(e_dml) > threshold_conflict and abs(e_scm) > threshold_conflict:
            if np.sign(e_dml) != np.sign(e_scm):
                warnings.append("DIRECTIONAL CONFLICT")
                diagnoses.append(
                    f"SCM structural simulation disagrees with DML: SCM predicts {e_scm:+.4f} "
                    f"while DML predicts {e_dml:+.4f}. Your SCM DAG may have missing confounders or incorrect edge orientations."
                )
                
        # Safety Status Assignment
        if consistency_score >= 0.75 and len(warnings) == 0:
            status = "SAFE (All reasoning layers agree on direction and magnitude)"
        elif consistency_score >= 0.40 and len(warnings) == 0:
            status = "CAUTION (Moderate consistency, minor variance in effect magnitudes)"
        else:
            status = "UNSAFE / HIGH BIAS (Significant conflicts or selection bias detected across layers)"
            
        report = {
            "Predictive_Effect": float(e_pred),
            "SCM_Effect": float(e_scm),
            "SCM_Mixture_Effect": float(e_scm_mix) if e_scm_mix is not None else None,
            "DML_Effect": float(e_dml),
            "Consistency_Score": float(consistency_score),
            "Safety_Status": status,
            "Warnings": warnings,
            "Diagnoses": diagnoses
        }
        return report

    def format_report(self, report: Dict) -> str:
        """
        Formats the audit report dict into a beautiful, readable string.
        """
        lines = []
        lines.append("="*65)
        lines.append(f"{'WhisperForest Causal Auditor Report':^65}")
        lines.append("="*65)
        lines.append(f"Predictive Layer Effect:       {report['Predictive_Effect']*100:+.2f}%")
        lines.append(f"SCM Structural Layer Effect:   {report['SCM_Effect']*100:+.2f}%")
        if report['SCM_Mixture_Effect'] is not None:
            lines.append(f"MoSCM Expert Layer Effect:     {report['SCM_Mixture_Effect']*100:+.2f}%")
        lines.append(f"Causal DML Layer Effect:       {report['DML_Effect']*100:+.2f}%")
        lines.append("-"*65)
        lines.append(f"Consistency Score:             {report['Consistency_Score']:.4f}")
        lines.append(f"Safety Status:                 {report['Safety_Status']}")
        lines.append("-"*65)
        
        if len(report['Warnings']) > 0:
            lines.append("WARNINGS:")
            for w in report['Warnings']:
                lines.append(f"  [!] {w}")
            lines.append("AUDIT DIAGNOSIS:")
            for d in report['Diagnoses']:
                lines.append(f"  - {d}")
        else:
            lines.append("AUDIT DIAGNOSIS:")
            lines.append("  - No conflicts or high selection biases detected. The intervention estimate is consistent.")
        lines.append("="*65)
        return "\n".join(lines)
