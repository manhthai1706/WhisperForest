# 🌲 WhisperForest: Causal Decision Intelligence Engine

**WhisperForest** là một thư viện suy luận nhân quả và mô phỏng phản thực tế (Causal AI & Decision Intelligence Engine) được thiết kế nhằm mục đích bóc tách nhiễu loạn và lắng nghe những tín hiệu nhân quả tinh tế ("tiếng thì thầm") bị che khuất bởi các biến gây nhiễu (confounders) trong dữ liệu quan sát.

Khác với các mô hình máy học truyền thống chỉ dừng lại ở việc khai phá mối quan hệ tương quan (P(Y|X)), WhisperForest cung cấp một pipeline khép kín từ khám phá đồ thị nhân quả (Causal Discovery), ước lượng tác động can thiệp (CATE), mô phỏng phản thực tế (P(Y|do(T))) cho đến truy vết nguyên nhân gốc rễ (Root Cause Analysis - RCA).

---

## 🏗️ Kiến Trúc Hệ Thống (System Architecture)

Hệ thống được thiết kế dạng phân tầng mô-đun hóa, chia sẻ chung cơ sở dữ liệu và các mô hình nền tảng:

```
[ Dữ Liệu Quan Sát / Observational Data ]
                   │
                   ▼
┌──────────────────────────────────────────────┐
│ 🔍 Tầng Khám Phá (Causal Discovery Layer)    │ ──► Phác thảo DAG nhân quả thông qua
│    ├─ PC Algorithm (Constraint-based)        │     Ensemble Voting (PC + DirectLiNGAM)
│    └─ DirectLiNGAM (Functional-based)        │
└──────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│ 🪵 Tầng Cấu Trúc SCM (Structural Layer)       │ ──► Thiết lập hệ phương trình cấu trúc lai
│    ├─ Pearl's Binary Threshold Abduction     │     để học quy luật vận hành và sai số
│    └─ Continuous Probability representation  │     ngoại sinh (exogenous noise)
└──────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│ 💊 Tầng Can Thiệp & Dự Báo (Causal & Pred)   │ ──► Tính toán rủi ro dự báo tĩnh P(Y|X)
│    ├─ CausalForestDML (Double ML)            │     và ước lượng hiệu ứng can thiệp CATE
│    └─ RandomForest Classifier/Regressor      │
└──────────────────────────────────────────────┘
        │                                 │
        ▼                                 ▼
┌────────────────────────────────┐ ┌────────────────────────────────┐
│ 🔍 Tầng Suy Diễn Phản Thực Tế  │ │ ⚖️ Tầng Kiểm Chứng Nhất Quán   │
│    (Counterfactual & RCA)      │ │    (Causal Consistency Layer)  │
│    Mô phỏng do(T=t) lan truyền │ │    Đo lường sự bất tương đồng  │
│    và truy vết nguyên nhân sự  │ │    giữa ba tầng suy luận để    │
│    cố qua KNN Cohorts (v2)     │ │    phát hiện Simpson's Paradox │
└────────────────────────────────┘ └────────────────────────────────┘
```

---

## 🔬 Các Tính Năng Cốt Lõi (Core Features)

### 1. Ensemble Causal Discovery (Khám phá DAG kết hợp)
*   **Ensemble Voting:** Tích hợp đồng thời thuật toán PC (dựa trên kiểm định độc lập có điều kiện) và DirectLiNGAM (dựa trên phân tích phi Gauss tuyến tính đệ quy).
*   **Bootstrap Stability Selection:** Chạy mô phỏng bootstrap nhiều lượt để tính toán điểm số tin cậy (Confidence Score) cho từng cạnh, loại bỏ cạnh giả và tự động bẻ gãy các chu kỳ (Cycle Breaking) dựa trên mức độ tương quan.

### 2. Pearl-Compliant SCM Simulation (Mô phỏng cấu trúc lai)
*   **Threshold Abduction cho biến nhị phân:** Sử dụng thuật toán suy diễn sai số dựa trên ngưỡng xác suất U_i của Pearl để mô phỏng phản thực tế, giúp bảo toàn tính chất logic nhị phân (0/1) của các nút trung gian trong đồ thị.
*   **Continuous Probability cho biến mục tiêu:** Cho phép nút kết quả (`target`) trả về xác suất nguy cơ liên tục (pred + noise) thay vì nhãn phân loại cứng, giúp đo lường chính xác các thay đổi rủi ro vi mô dưới tác động can thiệp do(T = t).

### 3. Double Machine Learning (DML) for CATE
*   Sử dụng **CausalForestDML** (thông qua EconML) để loại bỏ nhiễu hệ thống (Confounding by indication) và ước lượng chính xác tác động can thiệp cá thể hóa (CATE) trên từng đối tượng.

### 4. Causal Consistency Diagnostics (Chẩn đoán nhất quán đa tầng)
*   Công cụ độc nhất tự động kiểm chuẩn chéo hiệu ứng thu được giữa: **Predictive Layer vs SCM Layer vs DML Layer**.
*   Tự động phát hiện và cảnh báo trạng thái **Simpson's Paradox / Selection Bias** (`CONFOUNDED TREATMENT SIGNAL DETECTED`) khi mô hình dự báo thông thường và SCM bị che khuất hiệu ứng bởi các biến gây nhiễu mạnh, trong khi DML vẫn bóc tách được tín hiệu can thiệp thực tế.

---

## 🌲 Tiêu Chuẩn Thiết Kế Hệ Thống
*   **Tính ổn định (Stability):** Chỉ chấp nhận các mối quan hệ nhân quả vượt qua bộ lọc Stability Selection.
*   **Tính toàn vẹn cấu trúc (Structural Integrity):** Mọi can thiệp phản thực tế phải tuân thủ cơ chế lan truyền topo (Topological Propagation) của DAG để cập nhật các nút con.
*   **Tính nghiêm ngặt (Verification):** Luôn chẩn đoán nhất quán chéo để phát hiện lỗi ngoại suy và các thiên lệch lựa chọn (selection bias) trong mô hình.
