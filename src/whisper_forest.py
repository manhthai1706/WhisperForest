"""
WhisperForest: Causal Decision Intelligence Engine
===================================================

Kiến trúc đa tầng (Multi-tier Reasoning):

                    Data
                      |
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
  Predictive Branch          Causal Discovery
  (RandomForest)              DAG Learning
        │                           │
        └─────────────┬─────────────┘
                      ▼
               WhisperTrace RCA
                      |
         Per-patient Causal Trace
                      |
                      ▼
          CATE + Intervention Effects
                      |
                      ▼
      Knowledge / Experience Memory
                      |
                      ▼
      SCM Reasoning Engine (multi-hypothesis)
                      |
          ┌───────────┴────────────┐
          ▼                        ▼
 Counterfactual Simulation    Policy Generator
          │                        │
          └───────────┬────────────┘
                      ▼
             Causal Auditor
          (Consistency Voting)
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict

from .causal import CausalBranch
from .predictive import PredictiveBranch
from .rca import RCABranch
from .policy import PolicyBranch
from .optimization import OptimizationBranch
from .memory import KnowledgeMemory, CausalTrace, InterventionRecord
from .memory.reasoner import SCMReasoner


class WhisperForest:
    """
    Causal Decision Intelligence Engine — Multi-tier Reasoning.

    Tích hợp 6 tầng suy luận nhân quả:
      1. Predictive    — học dự đoán / phân loại
      2. Causal+RCA    — khám phá DAG + truy vết nguyên nhân
      3. CATE          — ước lượng hiệu quả can thiệp từng biến
      4. Memory        — Knowledge / Experience Memory (lưu causal traces)
      5. SCM Reasoner  — sinh đa giả thuyết can thiệp, simulate, vote
      6. Policy        — đúc kết chính sách từ full reasoning chain
      7. Auditor       — kiểm chứng nhất quán & biểu quyết

    Sử dụng cơ bản:
    ---------------
        wf = WhisperForest(data=df, target="target")

        # Nhánh 1: Học dự đoán
        wf.predictive.fit()
        risk = wf.predictive.predict_proba(patient)

        # Nhánh 2: Khám phá DAG + truy vết nguyên nhân
        wf.causal.get_dag(constraints=...)
        rca = wf.rca.analyze_anomaly(patient, baseline)

        # Nhánh 3: Mô phỏng nhân quả
        wf.rca.fit_scm()
        cf = wf.counterfactual(patient, outcome=0)

        # Nhánh 3 mở rộng: Multi-hypothesis reasoning
        plans, trace = wf.reason(rca_report=rca, patient=patient, cate_series=...)

        # Nhánh 4: Kiểm chứng nhất quán
        report = wf.audit_consistency(patient)
        print(wf.causal.auditor.format_report(report))
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target: str,
        treatment: Optional[str] = None,
        features: Optional[List[str]] = None,
    ):
        """
        Khởi tạo WhisperForest.

        Parameters
        ----------
        data      : Tập dữ liệu đầu vào (DataFrame).
        target    : Tên biến kết quả / mục tiêu (Y).
        treatment : Tên biến can thiệp (T), tuỳ chọn.
        features  : Danh sách đặc trưng (X). Nếu None, tự động lấy
                    toàn bộ cột trừ target và treatment.
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

        # ── Khởi tạo nhánh ────────────────────────────────────────────────
        self.causal = CausalBranch(self)        # Discovery + Estimation
        self.predictive = PredictiveBranch(self) # Predictive
        self.rca = RCABranch(self)              # RCA + SCM/MoSCM
        self.trace = self.rca                   # alias: wf.trace = wf.rca
        self.optimization = OptimizationBranch(self)  # Counterfactual

        # ── Memory & Reasoning ────────────────────────────────────────────
        self.memory = KnowledgeMemory()
        self.reasoner = SCMReasoner(self, self.memory)

        # ── Policy (kết nối với memory) ───────────────────────────────────
        self.policy = PolicyBranch(self)
        self.policy.set_memory(self.memory)

    # ══════════════════════════════════════════════════════════════════════
    #  NHÁNH 3 — SCM Propagation (core simulation engine)
    # ══════════════════════════════════════════════════════════════════════

    def simulate(self, patient: pd.DataFrame, interventions: Dict[str, float]) -> pd.DataFrame:
        """
        [Nhánh 3 — SCM] Mô phỏng trạng thái phản thực tế theo can thiệp do(T=t).

        Thực hiện 3 bước Pearl:
          1. Abduction : tính sai số ngoại sinh U_i = X_i - f_i(Pa_i)
          2. Action    : đè giá trị can thiệp lên biến được chỉ định
          3. Prediction: lan truyền topo qua DAG với U_i gốc

        Trả về DataFrame với toàn bộ biến sau can thiệp.
        """
        from sklearn.ensemble import RandomForestClassifier

        if not self.rca.engine.is_fitted:
            self.rca.fit_scm()

        noise_df = self.rca.engine.compute_noise(patient)
        topo_order = self.rca.attributor._get_topological_order(self.causal.get_dag())
        simulated_df = patient.copy()

        for node in topo_order:
            if node in interventions:
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
                                pred = (np.zeros(len(simulated_df))
                                        if model.classes_[0] == 0
                                        else np.ones(len(simulated_df)))
                            if node == self.target:
                                # Biến đích: dùng xác suất liên tục
                                simulated_df[node] = pred + noise_df[node]
                            else:
                                # Biến trung gian nhị phân: dùng threshold abduction
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
        [Nhánh 3 — SCM] Alias cho simulate(). Ký hiệu Pearl: do(T=t).
        """
        return self.simulate(patient, interventions)

    def simulate_mixture(self, patient: pd.DataFrame, interventions: Dict[str, float]) -> pd.DataFrame:
        """
        [Nhánh 3 — MoSCM] Mô phỏng can thiệp qua Mixture of SCM Experts.
        Tự động huấn luyện MoSCM nếu chưa được fit.
        """
        if not self.rca.mixture_engine.is_fitted:
            self.rca.fit_mixture_scm()
        return self.rca.mixture_engine.simulate(patient, interventions)

    def do_mixture(self, patient: pd.DataFrame, interventions: Dict[str, float]) -> pd.DataFrame:
        """
        [Nhánh 3 — MoSCM] Alias cho simulate_mixture(). Ký hiệu Pearl: do(T=t) qua MoSCM.
        """
        return self.simulate_mixture(patient, interventions)

    def counterfactual_mixture(self, patient: pd.DataFrame, interventions: Dict[str, float]) -> Union[float, pd.Series]:
        """
        [Nhánh 3 — MoSCM] Tính chênh lệch kết quả (Y_simulated - Y_original) qua MoSCM.
        """
        if not self.rca.mixture_engine.is_fitted:
            self.rca.fit_mixture_scm()
        return self.rca.mixture_engine.counterfactual(patient, interventions)

    def counterfactual(
        self,
        patient_df: pd.DataFrame,
        outcome: int = 0,
        k_neighbors: int = 5,
        top_k_causes: int = 3,
        use_mixture: bool = False
    ) -> pd.DataFrame:
        """
        [Nhánh 3 — Counterfactual] Tính trạng thái phản thực tế của bệnh nhân.

        Cho một bệnh nhân đang có kết quả outcome=1 (ví dụ: mắc bệnh), hàm này:
          1. Tìm K hàng xóm gần nhất có outcome mong muốn (ví dụ: khỏe mạnh).
          2. Xác định top root-cause features có chênh lệch lớn nhất với nhóm tham chiếu.
          3. Chạy SCM simulate với các features đó được đặt về mức tham chiếu.
          4. Trả ra bảng so sánh: Original | Counterfactual | Delta | Intervened.

        Parameters
        ----------
        patient_df   : DataFrame 1 dòng của bệnh nhân.
        outcome      : Giá trị kết quả mục tiêu cần mô phỏng (mặc định 0 = không bệnh).
        k_neighbors  : Số hàng xóm tham chiếu.
        top_k_causes : Số biến nguyên nhân để can thiệp.
        use_mixture  : Dùng MoSCM thay vì SCM đơn lẻ.
        """
        return self.optimization.counterfactual(
            patient_df=patient_df,
            outcome=outcome,
            k_neighbors=k_neighbors,
            top_k_causes=top_k_causes,
            use_mixture=use_mixture
        )

    # ══════════════════════════════════════════════════════════════════════
    #  NHÁNH 3 — Multi-hypothesis Reasoning Engine
    # ══════════════════════════════════════════════════════════════════════

    def reason(
        self,
        rca_report: pd.DataFrame,
        patient: pd.DataFrame,
        cate_series: Optional[pd.Series] = None,
        n_plans: int = 8,
    ):
        """
        [Nhánh 5 — Reasoner] Sinh và đánh giá nhiều phương án can thiệp.

        Quy trình 5 bước:
          1. Case Retrieval — tra Knowledge Memory tìm ca có causal profile tương tự.
          2. Hypothesis Generation — sinh đa giả thuyết (RCA-driven, Combined, Knowledge-guided).
          3. Counterfactual Simulation — mô phỏng từng giả thuyết qua SCM.
          4. Plan Ranking — xếp hạng theo delta outcome.
          5. Policy Recommendation — đề xuất từ kinh nghiệm lịch sử.

        Parameters
        ----------
        rca_report  : DataFrame với cột Attribution_Mean từ wf.rca.analyze_anomaly().
        patient     : DataFrame 1 dòng của bệnh nhân.
        cate_series : Series CATE cho từng feature (tuỳ chọn).
        n_plans     : Số candidate plans tối đa.

        Returns
        -------
        Tuple[ranked_results, trace, recommendations]
            ranked_results : List[Tuple[Hypothesis, InterventionRecord]] — giả thuyết + kết quả.
            trace          : CausalTrace — hồ sơ causal của patient.
            recommendations: List[Tuple[plan, score]] — policy đề xuất.
        """
        if cate_series is None:
            cate_series = pd.Series(0.0, index=self.features)

        ranked_results, trace = self.reasoner.reason_from_patient(
            patient_df=patient,
            rca_report=rca_report,
            cate_series=cate_series,
            dag_edges=self.causal.get_dag(),
            n_plans=n_plans,
        )

        recommendations = self.policy.recommend_from_trace(trace)

        return ranked_results, trace, recommendations

    def counterfactual_trace(
        self,
        patient_df: pd.DataFrame,
        outcome: int = 0,
        k_neighbors: int = 5,
        top_k_causes: int = 3,
    ) -> CausalTrace:
        """
        [Nhánh 3 — Trace] Như counterfactual() nhưng trả về CausalTrace.

        Lưu trace vào Knowledge Memory để dùng cho reasoning sau này.
        """
        cf_table = self.counterfactual(
            patient_df=patient_df,
            outcome=outcome,
            k_neighbors=k_neighbors,
            top_k_causes=top_k_causes,
        )

        rca_report = self.rca.analyze_anomaly(
            anomaly_data=patient_df,
            baseline_data=self.data[self.data[self.target] == outcome],
            causal_graph=self.causal.get_dag(),
            method="intervention",
            k_neighbors=k_neighbors,
        )

        trace = CausalTrace(
            features=patient_df.iloc[0],
            target=self.target,
            target_value=float(patient_df[self.target].iloc[0]) if self.target in patient_df else 0.0,
            rca_scores=rca_report["Attribution_Mean"] if "Attribution_Mean" in rca_report.columns else rca_report.iloc[:, 0],
            cate_estimates=pd.Series(0.0, index=self.features),
            dag_edges=self.causal.get_dag(),
        )

        intervened_features = cf_table.index[cf_table["Intervened"]].tolist()
        interventions = {
            feat: float(cf_table.loc[feat, "Counterfactual"])
            for feat in intervened_features
        }
        simulated_outcome = float(cf_table.loc[self.target, "Counterfactual"]) if self.target in cf_table.index else 0.0
        original_outcome = float(cf_table.loc[self.target, "Original"]) if self.target in cf_table.index else 0.0
        delta = simulated_outcome - original_outcome
        success = delta < 0

        trace.interventions_tried.append(InterventionRecord(
            plan=interventions,
            simulated_outcome=simulated_outcome,
            delta=delta,
            success=success,
        ))

        self.memory.store(trace)
        return trace

    # ══════════════════════════════════════════════════════════════════════
    #  NHÁNH 4 — Causal Auditor
    # ══════════════════════════════════════════════════════════════════════

    def audit_consistency(
        self,
        patient: pd.DataFrame,
        interventions: Dict[str, float] = None,
        threshold_conflict: float = 0.01,
        rca_report: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        [Nhánh 4 — Auditor] Kiểm chứng nhất quán giữa các tầng suy luận.

        Voting layers: Predictive, SCM, DML, RCA.

        Parameters
        ----------
        patient       : DataFrame bệnh nhân.
        interventions : Dict biến can thiệp {feature: value}.
        rca_report    : DataFrame RCA từ wf.rca.analyze_anomaly() — tham gia voting.
        threshold_conflict : ngưỡng chênh lệch để kích hoạt cảnh báo.
        """
        return self.causal.audit_consistency(
            patient,
            interventions or {},
            threshold_conflict=threshold_conflict,
            rca_report=rca_report,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  Serialization (Lưu & Tải mô hình)
    # ══════════════════════════════════════════════════════════════════════

    def save_model(self, file_path: str):
        """
        Xuất toàn bộ pipeline (DAG, Predictive, SCM, Metadata) ra file.
        Ưu tiên HDF5 (.h5) nếu h5py được cài đặt, ngược lại dùng Pickle.
        """
        import pickle

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
            print("Exporting model pipeline to H5 format...")
            with h5py.File(file_path, "w") as f:
                for k, v in state.items():
                    if v is not None:
                        f.create_dataset(k, data=np.void(pickle.dumps(v)))
            print(f"Saved (H5) → {file_path}")
        else:
            print("h5py not found. Using pickle serialization...")
            with open(file_path, "wb") as f:
                pickle.dump(state, f)
            print(f"Saved (Pickle) → {file_path}")

    def load_model(self, file_path: str):
        """
        Tải pipeline (DAG, Predictive, SCM, Metadata) từ file H5 hoặc Pickle.
        """
        import pickle
        import os

        is_h5 = False
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                header = f.read(8)
                if header == b"\x89HDF\r\n\x1a\n":
                    is_h5 = True

        if is_h5:
            try:
                import h5py
            except ImportError:
                raise ImportError("Model file is H5 format but h5py is not installed.")

            print("Loading model pipeline from H5 format...")
            with h5py.File(file_path, "r") as f:
                if "predictive_model" in f:
                    self.predictive.modeling.model = pickle.loads(f["predictive_model"][()].tobytes())
                    print("- Loaded predictive model.")
                if "scm_models" in f:
                    self.rca.engine.models = pickle.loads(f["scm_models"][()].tobytes())
                    self.rca.engine.is_fitted = True
                    print("- Loaded SCM equations.")
                if "scm_parents" in f:
                    self.rca.engine.parents_map = pickle.loads(f["scm_parents"][()].tobytes())
                    print("- Loaded SCM parents map.")
                if "dag_edges" in f:
                    self.causal.set_dag(pickle.loads(f["dag_edges"][()].tobytes()))
                    print("- Loaded causal DAG.")
                if "metadata" in f:
                    metadata = pickle.loads(f["metadata"][()].tobytes())
                    self.features = metadata["features"]
                    self.target = metadata["target"]
                    self.treatment = metadata["treatment"]
                    print("- Loaded metadata.")
        else:
            print("Loading model pipeline from pickle format...")
            with open(file_path, "rb") as f:
                state = pickle.load(f)

            if "predictive_model" in state:
                self.predictive.modeling.model = state["predictive_model"]
                print("- Loaded predictive model.")
            if "scm_models" in state:
                self.rca.engine.models = state["scm_models"]
                if self.rca.engine.models is not None:
                    self.rca.engine.is_fitted = True
                print("- Loaded SCM equations.")
            if "scm_parents" in state:
                self.rca.engine.parents_map = state["scm_parents"]
                print("- Loaded SCM parents map.")
            if "dag_edges" in state:
                self.causal.set_dag(state["dag_edges"])
                print("- Loaded causal DAG.")
            if "metadata" in state:
                metadata = state["metadata"]
                self.features = metadata["features"]
                self.target = metadata["target"]
                self.treatment = metadata["treatment"]
                print("- Loaded metadata.")

        # Backward compat: reconstruct parents_map from dag if missing
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

        print(f"WhisperForest pipeline loaded ← {file_path}")

    # ══════════════════════════════════════════════════════════════════════
    #  Legacy / backward-compatible (giữ lại để không phá vỡ test cũ)
    # ══════════════════════════════════════════════════════════════════════

    def evaluate_causal_consistency(self, patient: pd.DataFrame, interventions: Dict[str, float]) -> Dict:
        """
        [Deprecated] Dùng audit_consistency() thay thế.
        Giữ lại để tương thích với test_full_features.py cũ.
        """
        return self.audit_consistency(patient, interventions)
