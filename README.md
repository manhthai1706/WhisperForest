# 🌲 WhisperForest: Lắng Nghe Tiếng Thì Thầm Của Khu Rừng Nhân Quả

> *"Giữa đại ngàn dữ liệu mênh mông, kẻ lữ hành tầm thường chỉ nghe thấy tiếng xào xạc hỗn loạn của lá khô (Correlation). Nhưng với người biết cách lắng nghe, khu rừng luôn thì thầm những sợi dây nhân quả vô hình kết nối vạn vật (Causality)."*

**WhisperForest** là một động cơ trí tuệ nhân quả (Causal Decision Intelligence Engine) được thiết kế để lắng nghe những tiếng thì thầm nhỏ bé nhất của dữ liệu, bóc tách nhiễu loạn để tìm ra cội nguồn của sự cố (RCA) và mô phỏng các thế giới phản thực tế (Counterfactual Simulation).

---

## 🗺️ Bản Đồ Kiến Trúc Của Đại Ngàn WhisperForest

Giống như cấu trúc phân tầng của một khu rừng tự nhiên, WhisperForest kết nối các tầng suy luận để tạo nên một hệ sinh thái khép kín:

```
                  [ 🌲 Dữ Liệu Quan Sát ]
                             │
                             ▼
     ┌──────────────────────────────────────────────┐
     │ 🔍 Tầng Khám Phá (Discovery Layer)           │ ──► Lắng nghe & phác thảo bản đồ DAG
     │    ├─ PC Algorithm (Constraint)              │     thông qua Ensemble Voting ổn định
     │    └─ DirectLiNGAM (Functional)               │
     └──────────────────────────────────────────────┘
                             │
                             ▼
     ┌──────────────────────────────────────────────┐
     │ 🪵 Tầng Cấu Trúc SCM (Structural Layer)       │ ──► Thiết lập hệ phương trình cấu trúc
     │    └─ Pearl's Binary Threshold Abduction     │     để lưu giữ quy luật vận hành của rừng
     └──────────────────────────────────────────────┘
                             │
                             ▼
     ┌──────────────────────────────────────────────┐
     │ 💊 Tầng Can Thiệp & Dự Báo (Causal & Pred)   │ ──► Ước lượng hiệu ứng can thiệp CATE
     │    ├─ CausalForestDML (Double ML)            │     và chạy dự báo rủi ro tĩnh P(Y|X)
     │    └─ RandomForest Classifier/Regressor      │
     └──────────────────────────────────────────────┘
         │                                      │
         ▼                                      ▼
┌────────────────────────────────┐    ┌────────────────────────────────┐
│ 🔍 Tầng Suy Diễn Phản Thực Tế   │    │ ⚖️ Tầng Kiểm Chứng Nhất Quán    │
│    (Counterfactual & RCA)      │    │    (Consistency Layer)         │
│    Mô phỏng do(T=t) lan truyền │    │    Đo lường độ lệch pha giữa   │
│    và truy vết nguyên nhân gốc │    │    Association và Causality    │
│    rễ sự cố qua KNN Cohorts    │    │    để phát hiện Simpson's    │
└────────────────────────────────┘    └────────────────────────────────┘
```

---

## 🕯️ Nghệ Thuật Lắng Nghe - 3 Cấp Độ Suy Luận Nhân Quả

### 1. Khám Phá Bản Đồ DAG (Ensemble Discovery & Stability Selection)
Bằng cách kết hợp thuật toán **PC** (tìm kiếm sự độc lập có điều kiện) và **DirectLiNGAM** (khai phá mối quan hệ tuyến tính phi Gauss đệ quy) cùng cơ chế **Bootstrap Stability Selection**, WhisperForest gom nhặt ý kiến của hàng chục lượt lấy mẫu để vẽ nên đồ thị nhân quả DAG chính xác nhất, phá vỡ các vòng lặp luẩn quẩn (Cycle Breaking) dựa trên sức mạnh tương quan.

### 2. Mô Phỏng Phản Thực Tế Chuẩn Pearl (SCM do-calculus)
Khi ta gieo một tác nhân can thiệp $do(T = 1)$, WhisperForest không chỉ đổi nhãn một cột tĩnh. Nó thực hiện quy trình 3 bước của Pearl:
*   **Abduction (Suy diễn):** Trích xuất xác suất sai số ngoại sinh (exogenous noise) của từng bệnh nhân. Với biến nhị phân, nó dùng cơ chế **Threshold Abduction** ($U_i$) để tìm ngưỡng nhạy cảm riêng biệt.
*   **Action (Can thiệp):** Cắt bỏ mọi tác động đầu vào của biến can thiệp, ép nó nhận giá trị mong muốn.
*   **Prediction (Dự báo):** Lan truyền tác động xuôi dòng theo cấu trúc liên kết topo của DAG để tính toán xác suất rủi ro phản thực tế mới, bảo toàn tuyệt đối thuộc tính logic của các nút trung gian.

### 3. Tiếng Vọng Đồng Thanh (Causal Consistency Score)
Khu rừng dữ liệu luôn đầy rẫy những ảo ảnh của sự gây nhiễu (Confounding by indication / Simpson's Paradox). Tầng chẩn đoán nhất quán chéo tự động so sánh:
$$\text{Predictive Effect } P(Y|X) \quad \text{vs} \quad \text{SCM Simulation } P(Y|do(T)) \quad \text{vs} \quad \text{DML CATE } \tau(X)$$
Nếu có sự mâu thuẫn (DML báo có tác dụng nhưng SCM/RF báo $0\%$), động cơ sẽ ngay lập tức gióng chuông cảnh báo:
`CONFOUNDED TREATMENT SIGNAL DETECTED: Potential selection bias or strong confounding is masking the true treatment effect.`

---

## 🛠️ Hướng Dẫn Vận Hành Động Cơ

### Khởi Tạo Động Cơ và Nạp Dữ Liệu
```python
import pandas as pd
from src.whisper_forest import WhisperForest

# Đọc tập dữ liệu quan sát từ khu rừng của bạn
df = pd.read_csv("heart.csv")

# Chỉ định mục tiêu (Target), biến can thiệp (Treatment) và các thuộc tính nền (Features)
wf = WhisperForest(
    data=df,
    target="target",
    treatment="statin_treatment",
    features=["age", "sex", "chol", "thalach", "oldpeak", "ca", "thal", "cp"]
)
```

### Chạy Quy Trình Khám Phá & Huấn Luyện Khép Kín
```python
# 1. Huấn luyện tầng dự báo tĩnh
wf.predictive.fit(model_type="classifier")

# 2. Khám phá DAG nhân quả bằng phương pháp Ensemble (PC + LiNGAM voting)
discovered_dag = wf.causal.get_dag(method="ensemble", bootstrap_runs=25)

# 3. Khớp hệ phương trình cấu trúc nhân quả (SCM)
wf.rca.fit_scm()
```

### Đóng Gói Và Lưu Trữ Động Cơ (H5 & Pickle Hybrid)
WhisperForest hỗ trợ xuất toàn bộ tri thức của nó ra tệp tin nhị phân. Nếu môi trường thiếu thư viện `h5py`, nó tự động chuyển đổi sang định dạng `Pickle` tương thích ngược:
```python
# Lưu lại toàn bộ mô hình dự báo, SCM, DAG và metadata
wf.save_model("whisper_forest_model.h5")

# Nạp lại mô hình ở bất kỳ máy chủ runtime nào chỉ trong 1 dòng lệnh
new_wf = WhisperForest(data=df, target="target", treatment="statin_treatment", features=features)
new_wf.load_model("whisper_forest_model.h5")
```

### Chạy Thử Nghiệm Quyết Định Kháng Cự Phản Thực Thế (Inference)
```python
# Định nghĩa hồ sơ một bệnh nhân có rủi ro cao (chưa được điều trị)
patient = pd.DataFrame([{
    "age": 62, "sex": 1, "cp": 2, "trestbps": 140, "chol": 280,
    "thalach": 130, "exang": 1, "oldpeak": 2.2, "slope": 1, "ca": 2, "thal": 2,
    "statin_treatment": 0
}])

# Mô phỏng phản thực tế: Do(statin_treatment = 1) và lan truyền xuôi dòng
simulated_state = wf.simulate(patient, {"statin_treatment": 1})
print(simulated_state[["age", "chol", "statin_treatment", "target"]])

# Chẩn đoán nhất quán chéo đa tầng để phát hiện Simpson's Paradox
consistency = wf.evaluate_causal_consistency(patient, {"statin_treatment": 1})
print(f"Chỉ số nhất quán: {consistency['Consistency_Score']:.4f}")
print(f"Trạng thái hệ thống: {consistency['Status']}")
```

---

## 🌲 Tiêu Chuẩn Đại Ngàn
*   **Không ồn ào:** Chỉ trích xuất thông tin thực thông qua Stability Selection.
*   **Bảo toàn cấu trúc:** Mọi can thiệp phản thực tế đều phải tuân thủ nghiêm ngặt quy luật lan truyền topo nhân quả của DAG.
*   **Hoài nghi khoa học:** Luôn kiểm chứng chéo độ nhất quán trước khi đưa ra quyết định hành động.
# WhisperForest
