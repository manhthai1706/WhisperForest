# 🌲 WhisperForest: Causal Decision Intelligence Engine

**WhisperForest** là một thư viện suy luận nhân quả và mô phỏng phản thực tế (Causal AI & Decision Intelligence Engine) được thiết kế nhằm mục đích bóc tách nhiễu loạn và lắng nghe những tín hiệu nhân quả tinh tế ("tiếng thì thầm") bị che khuất bởi các biến gây nhiễu (confounders) trong dữ liệu quan sát.

Khác với các mô hình máy học truyền thống chỉ dừng lại ở việc khai phá mối quan hệ tương quan ($P(Y|X)$), WhisperForest cung cấp một pipeline khép kín từ khám phá đồ thị nhân quả (Causal Discovery), ước lượng tác động can thiệp (CATE), mô phỏng phản thực tế ($P(Y|do(T))$) cho đến truy vết nguyên nhân gốc rễ (Root Cause Analysis - RCA).

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
│ 🪵 Tầng Cấu Trúc MoSCM (SCM Experts Layer)    │ ──► Thiết lập hệ thống SCM chuyên biệt hóa
│    ├─ Expectation-Maximization (EM) Learning  │     bằng EM-MoSCM và bộ định tuyến động
│    └─ Pearl's Binary Threshold Abduction     │     (Gating Router) dựa trên đặc trưng
└──────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│ 💊 Tầng Can Thiệp & Dự Báo (Causal & Pred)   │ ──► Tính toán hồi quy/phân loại dự báo tĩnh
│    ├─ CausalForestDML (Double ML)            │     và ước lượng hiệu ứng nhân quả CATE
│    └─ RandomForest Classifier/Regressor      │
└──────────────────────────────────────────────┘
         │                                 │
         ▼                                 ▼
┌────────────────────────────────┐ ┌────────────────────────────────┐
│ 🔍 Tầng Suy Diễn Phản Thực Tế  │ │ ⚖️ Tầng Kiểm Toán Nhân Quả     │
│    (Counterfactual & RCA)      │ │    (Causal Auditor Layer)      │
│    Mô phỏng do(T=t) lan truyền │ │    Đánh giá độ nhất quán giữa   │
│    và truy vết nguyên nhân sự  │ │    các tầng và gán nhãn trạng   │
│    cố qua KNN Cohorts (v2)     │ │    thái an toàn (Safety Status) │
└────────────────────────────────┘ └────────────────────────────────┘
```

---

## 🔬 Các Tính Năng Cốt Lõi (Core Features)

### 1. Ensemble Causal Discovery (Khám phá DAG kết hợp)
*   **Ensemble Voting:** Tích hợp đồng thời thuật toán PC (dựa trên kiểm định độc lập có điều kiện) và DirectLiNGAM (dựa trên phân tích phi Gauss tuyến tính đệ quy).
*   **Bootstrap Stability Selection:** Chạy mô phỏng bootstrap nhiều lượt để tính toán điểm số tin cậy (Confidence Score) cho từng cạnh, loại bỏ cạnh giả và tự động bẻ gãy các chu kỳ (Cycle Breaking) dựa trên mức độ tương quan.

### 2. Pearl-Compliant SCM Simulation (Mô phỏng cấu trúc lai)
*   **Threshold Abduction cho biến nhị phân:** Sử dụng thuật toán suy diễn sai số dựa trên ngưỡng xác suất $U_i$ của Pearl để mô phỏng phản thực tế, giúp bảo toàn tính chất logic nhị phân (0/1) của các nút trung gian trong đồ thị.
*   **Continuous Probability cho biến mục tiêu:** Cho phép nút kết quả (`target`) trả về xác suất nguy cơ liên tục ($pred + noise$) thay vì nhãn phân loại cứng, giúp đo lường chính xác các thay đổi rủi ro vi mô dưới tác động can thiệp $do(T = t)$.

### 3. EM-MoSCM (Mạng lưới SCM hỗn hợp phân tầng học bằng EM)
*   **Không phụ thuộc một DAG duy nhất:** Tự động tìm kiếm các cơ chế nhân quả tiềm ẩn (latent mechanisms) bằng thuật toán phân cụm mềm (Soft Clustering) và tối ưu hóa đồng thời các SCM chuyên gia (SCM Experts).
*   **Thuật toán EM nhân quả (Causal EM Loop):**
    *   **M-Step:** Huấn luyện các phương trình của mỗi Cluster SCM Expert sử dụng trọng số mềm của mẫu dữ liệu (`sample_weight`).
    *   **E-Step:** Đánh giá độ khớp của dữ liệu bệnh nhân đối với các phương trình cấu trúc (Log-Likelihood) để phân phối lại trọng số phân nhóm.
*   **Gating Router:** Huấn luyện một bộ định tuyến học giám sát để dự đoán phân phối trọng số của các cơ chế nhân quả cho bệnh nhân mới dựa trên các đặc trưng lâm sàng $X$.

### 4. Causal Auditor (Bộ kiểm toán Nhân quả chuyên biệt)
*   Tự động so sánh chéo hiệu ứng can thiệp thu được từ 3 lớp: **Predictive Layer vs SCM/MoSCM Layer vs DML Layer**.
*   **Simpson's Paradox & Confounding Detection**: Tự động phát hiện và cảnh báo trạng thái nhiễu loạn chọn lọc khi xu hướng tương quan (Predictive/SCM) mâu thuẫn hoàn toàn với tác động nhân quả đã điều chỉnh nhiễu (DML).
*   **Safety Status Assignment**: Phân cấp mức độ tin cậy của can thiệp phản thực tế thành `SAFE` (Nhất quán cao), `CAUTION` (Nhất quán trung bình), và `UNSAFE / HIGH BIAS` (Xuất hiện xung đột hướng hoặc nhiễu chọn lọc lớn).

---

## 💻 Kịch Bản Kiểm Thử & Chạy Thử Nghiệm (Testing Scripts)

Thư viện tích hợp sẵn 3 kịch bản kiểm thử toàn diện từ lý thuyết đến thực tế:

### 1. Kiểm thử phân hệ SCM Hỗn hợp (MoSCM)
Chạy thử nghiệm giải thuật EM phân cụm cơ chế nhân quả và định tuyến động cho bệnh nhân tim mạch:
```powershell
python test_mixture_scm.py
```

### 2. Kiểm thử hợp nhất toàn bộ tính năng (Unified Integration Test)
Chạy xuyên suốt qua 10 giai đoạn từ chuẩn bị dữ liệu, khám phá DAG, học máy dự báo, ước lượng ATE/CATE đến kiểm toán nhân quả và lưu trữ mô hình (HDF5):
```powershell
python test_full_features.py
```

### 3. Case Study thực tế bất động sản Boston (Boston Housing Case Study)
Áp dụng WhisperForest vào bài toán hồi quy thực tế (`medv` - giá nhà liên tục) và can thiệp vị trí giáp sông Charles (`chas`), kiểm toán kinh tế đô thị và truy vết nguyên nhân giá cao thông qua các lộ trình nhân quả:
```powershell
python test_boston_housing.py
```

---

## 🌲 Tiêu Chuẩn Thiết Kế Hệ Thống
*   **Tính ổn định (Stability):** Chỉ chấp nhận các mối quan hệ nhân quả vượt qua bộ lọc Stability Selection.
*   **Tính toàn vẹn cấu trúc (Structural Integrity):** Mọi can thiệp phản thực tế phải tuân thủ cơ chế lan truyền topo (Topological Propagation) của DAG để cập nhật các nút con.
*   **Tính kiểm toán (Auditability):** Luôn bắt buộc chẩn đoán nhất quán chéo để phát hiện lỗi ngoại suy và các thiên lệch lựa chọn (selection bias) trong mô hình thông qua **Causal Auditor**.
