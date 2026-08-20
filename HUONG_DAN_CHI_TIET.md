# HƯỚNG DẪN CHUYÊN SÂU — XÂY DỰNG MICRA THÀNH SẢN PHẨM THẬT

Tài liệu này trả lời đúng một câu hỏi: **từ số không, làm gì, theo thứ tự nào, để có
một AI agent chạy thật.** Mỗi bước đều ghi rõ: chuẩn bị gì, làm gì, xong thì có gì,
và kiểm tra bằng cách nào.

---

## PHẦN 0 — BẢNG TỔNG QUAN CÁC BƯỚC

| # | Giai đoạn | Việc chính | Thời gian | Xong thì có gì | Cách kiểm tra |
|---|---|---|---|---|---|
| 1 | Chuẩn bị máy | Python, môi trường ảo, thư viện | 30 phút | `pip list` chạy sạch | `python -c "import sklearn, pandas"` |
| 2 | Khóa API | Tài khoản Anthropic hoặc OpenAI, nạp tiền | 20 phút | Tệp `.env` có khóa | `python -c "from src.llm import goi_llm; print(goi_llm('bạn là trợ lý','xin chào'))"` |
| 3 | Dữ liệu | Chuẩn hóa về đúng lược đồ | 1–2 giờ | `data/*.csv` | `python -c "import pandas as pd; pd.read_csv(...).shape"` |
| 4 | Tầng đặc trưng | Viết công thức tính chỉ số | 3–4 giờ | `src/features.py` | mục [1] của `kiem_thu.py` |
| 5 | Huấn luyện | So sánh mô hình, chọn bằng kiểm định chéo | 2–3 giờ | `models/*.joblib` | `python -m src.train` |
| 6 | Công cụ chấm điểm | Bọc mô hình thành hàm tất định | 1 giờ | `src/scoring.py` | mục [2] của `kiem_thu.py` |
| 7 | Lớp LLM | Trừu tượng hóa nhà cung cấp | 1 giờ | `src/llm.py` | đổi `MICRA_LLM` không phải sửa mã |
| 8 | Sáu tác tử | Mỗi tác tử một việc duy nhất | 6–8 giờ | `src/agents/` | `python run_cli.py` |
| 9 | Bộ điều phối | Nối các tác tử theo thứ tự | 1 giờ | `src/orchestrator.py` | chạy 60 hồ sơ không lỗi |
| 10 | Kiểm soát ảo giác | Đối chiếu số + rà soát ngữ nghĩa | 3–4 giờ | `a5_kiem_soat.py` | mục [3] của `kiem_thu.py` |
| 11 | Giao diện | Streamlit 4 màn hình | 4–5 giờ | `app.py` | `streamlit run app.py` |
| 12 | Kiểm thử | Bộ tự kiểm 16 mục | 2 giờ | `kiem_thu.py` | phải 0 mục hỏng |

Tổng: khoảng **30–40 giờ làm việc** cho một người có nền lập trình cơ bản.
Nếu chia cho 2 người thì 3–4 ngày.

---

## PHẦN 1 — CHUẨN BỊ MÁY

### Cần cài trước

| Phần mềm | Phiên bản | Lấy ở đâu | Vì sao cần |
|---|---|---|---|
| Python | 3.10 trở lên | python.org | Nền tảng chạy toàn bộ |
| Git | bất kỳ | git-scm.com | Quản lý phiên bản, làm việc nhóm |
| VS Code | bất kỳ | code.visualstudio.com | Soạn mã, có sẵn terminal |

### Các lệnh chạy theo đúng thứ tự

```bash
cd micra
python -m venv .venv                 # tạo môi trường riêng, không làm bẩn máy
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Nếu `pip install xgboost` hoặc `shap` lỗi:** cứ bỏ qua. Hệ thống được thiết kế để
tự chuyển sang `GradientBoostingClassifier` của scikit-learn. Bạn sẽ thấy dòng
`[Lưu ý] Chưa cài xgboost — dùng GradientBoosting thay thế`. Mọi thứ vẫn chạy.

> Đây là một quyết định thiết kế đáng nói khi phỏng vấn: hệ thống không được sập
> chỉ vì thiếu một thư viện tùy chọn.

---

## PHẦN 2 — LẤY KHÓA API

Bạn **không bắt buộc** phải có khóa API. Chế độ `mock` chạy được toàn bộ hệ thống.
Nhưng để demo trước hội đồng thì nên có, vì tờ trình do LLM thật viết đọc hay hơn nhiều.

| Nhà cung cấp | Đăng ký tại | Biến môi trường | Chi phí ước tính |
|---|---|---|---|
| Anthropic (Claude) | console.anthropic.com | `ANTHROPIC_API_KEY` | vài nghìn đồng cho một hồ sơ |
| OpenAI (GPT) | platform.openai.com | `OPENAI_API_KEY` | tương đương |

```bash
cp .env.example .env
# mở .env, sửa:
#   MICRA_LLM=anthropic
#   ANTHROPIC_API_KEY=sk-ant-...
```

**Ba điều bắt buộc về bảo mật — hỏng ở đây là mất điểm liêm chính:**

1. `.env` đã nằm trong `.gitignore`. Đừng bao giờ bỏ ra.
2. Đừng dán khóa vào mã nguồn, ảnh chụp màn hình, hay bài nộp.
3. Đặt hạn mức chi tiêu trong bảng điều khiển của nhà cung cấp trước khi chạy hàng loạt.

Kiểm tra khóa hoạt động:

```bash
python -c "from src.llm import goi_llm, dang_dung; print(dang_dung()); print(goi_llm('Bạn là trợ lý.','Trả lời đúng hai chữ: Xin chào'))"
```

---

## PHẦN 3 — DỮ LIỆU

### Lược đồ bắt buộc

| Nhóm cột | Tên cột | Kiểu | Ghi chú |
|---|---|---|---|
| Định danh | `ma_ho`, `ten_ho`, `nganh`, `tinh_thanh` | chuỗi | `nganh` phải khớp khóa trong `DIEN_CHUAN_NGANH` |
| Thâm niên | `so_nam_hoat_dong` | số thực | lấy từ giấy đăng ký kinh doanh |
| Doanh thu | `doanh_thu_hoa_don_T1` … `_T12` | số nguyên (VNĐ) | từ hóa đơn điện tử |
| Dòng tiền | `dong_tien_ngan_hang_T1` … `_T12` | số nguyên (VNĐ) | từ sao kê tài khoản kinh doanh |
| Tiện ích | `tien_dien_T1` … `_T12` | số nguyên (VNĐ) | biến đối chứng chống khai khống |
| Hoạt động | `so_giao_dich_trung_binh_ngay` | số nguyên | từ máy tính tiền |
| Tín dụng | `co_lich_su_tin_dung_CIC` | "Có"/"Không" | từ CIC |
| Nhu cầu | `so_tien_de_nghi_vay` | số nguyên (VNĐ) | từ đơn vay |
| Nhãn | `nhan_vo_no` | 0/1 | **chỉ có khi huấn luyện** |

### Vì sao có cột tiền điện

Đây là biến quan trọng nhất mà ít người nghĩ tới. Doanh thu hóa đơn có thể bị thổi lên
để được vay — đúng như **định luật Goodhart**. Nhưng tiền điện thì khó ngụy tạo, vì nó
do bên thứ ba đo. Khi doanh thu khai báo tăng mà tiền điện không tăng tương ứng, đó là
tín hiệu bất thường. Đặc trưng `dien_lech_nganh` chính là để bắt việc này.

### Khi thay bằng dữ liệu thật

Thứ tự ưu tiên đấu nối, dễ trước khó sau:

1. **Sao kê ngân hàng** — dễ nhất, khách hàng tự tải PDF/CSV gửi lên
2. **Hóa đơn điện tử** — qua nhà cung cấp dịch vụ hóa đơn (Viettel, VNPT, MISA…)
3. **Tiền điện** — cổng thanh toán tiện ích hoặc khách tự chụp hóa đơn
4. **CIC** — cần tổ chức tín dụng đứng tên truy vấn, khó nhất

Chỉ cần thay phần thân các hàm trong `src/agents/a1_thu_thap.py`. Không phải sửa gì khác.

---

## PHẦN 4 — TẦNG ĐẶC TRƯNG

Toàn bộ 11 đặc trưng nằm trong `src/features.py`. Đây là **tầng tất định** — mọi giá
trị đều tính bằng công thức, tái lập được, kiểm toán được.

| Đặc trưng | Ý nghĩa kinh tế | Công thức |
|---|---|---|
| `dt_trung_binh` | Quy mô kinh doanh | trung bình 12 tháng |
| `dt_bien_dong` | Rủi ro dòng tiền | độ lệch chuẩn / trung bình |
| `dt_xu_huong` | Đang lên hay đang xuống | TB 6 tháng cuối / 6 tháng đầu |
| `dt_thang_thap` | Sức chịu đựng mùa thấp điểm | tháng thấp nhất / trung bình |
| `nh_tren_dt` | Mức khớp giữa khai báo và thực tế | tổng dòng tiền / tổng hóa đơn |
| `nh_on_dinh` | Tính đều đặn của dòng tiền | 1 − CV của tỷ lệ theo tháng |
| `dien_lech_nganh` | **Dấu hiệu khai khống hóa đơn** | (điện/doanh thu − chuẩn ngành) / chuẩn ngành |
| `so_nam` | Khả năng sống sót đã được kiểm chứng | trực tiếp |
| `gd_ngay` | Mật độ khách thật | trực tiếp |
| `vay_tren_dt` | Mức đòn bẩy đề nghị | số tiền vay / doanh thu tháng |
| `co_cic` | Đã có lịch sử tín dụng chưa | 0/1 |

**Quy tắc khi thêm đặc trưng mới:** phải giải thích được bằng một câu tiếng Việt cho
cán bộ tín dụng hiểu. Đặc trưng nào không diễn giải được thì không đưa vào — vì lúc từ
chối khách hàng bạn sẽ không giải trình nổi.

---

## PHẦN 5 — HUẤN LUYỆN MÔ HÌNH

```bash
python -m src.train
```

### Kết quả trên bộ dữ liệu mô phỏng

```
Dữ liệu: 60 hộ, 9 hộ vỡ nợ (15%)
Mô hình                  AUC          PR-AUC           Brier
logistic         0.858 ±0.023     0.630 ±0.042     0.089 ±0.006
gradient_boosting 0.787 ±0.054    0.500 ±0.074     0.116 ±0.011

Mô hình được chọn: LOGISTIC (AUC 0.858)
```

### Bốn quyết định kỹ thuật cần bảo vệ được

**1. Vì sao báo cáo bằng kiểm định chéo, không phải trên tập huấn luyện.**
AUC trên tập huấn luyện của bộ này là 0,98 — con số đẹp và vô nghĩa, vì mô hình đã
nhìn thấy chính những dòng đó. Chỉ số duy nhất đáng tin là chỉ số trên dữ liệu mô hình
chưa từng thấy. Hệ thống lặp lại kiểm định chéo 10 lần và báo cả sai số chuẩn.

**2. Vì sao hồi quy logistic thắng XGBoost.**
Vì chỉ có 60 mẫu và 9 ca vỡ nợ. Ở quy mô đó, mô hình phức tạp học nhiễu nhiều hơn học
quy luật. Đây là kết quả trung thực và nên báo cáo đúng như vậy — đừng ép XGBoost thắng
cho "có vẻ AI hơn". Hệ thống **tự chọn mô hình bằng kiểm định chéo**, không gán cứng.
Khi có vài nghìn hồ sơ thật, cán cân sẽ đảo chiều và mã nguồn tự xử lý.

**3. Vì sao phải hiệu chỉnh xác suất (calibration).**
Mô hình gốc trả về điểm số xếp hạng tốt nhưng con số không đọc được như xác suất thật.
`CalibratedClassifierCV` biến nó thành xác suất có nghĩa: "13%" phải nghĩa là trong 100
hộ tương tự thì khoảng 13 hộ vỡ nợ. Không hiệu chỉnh thì mọi ngưỡng phê duyệt đều sai.

**4. Vì sao ngưỡng phân hạng nằm ngoài mô hình.**
Ngưỡng Thấp/Trung bình/Cao trong `src/scoring.py` là **quyết định khẩu vị rủi ro của tổ
chức cho vay**, không phải kết quả của mô hình. Tách ra để tổ chức tín dụng tự chỉnh mà
không phải huấn luyện lại.

### Khi có dữ liệu thật

- Cần tối thiểu **300–500 hồ sơ** đã biết kết quả, trong đó ít nhất 30 ca vỡ nợ
- Phải chia theo **thời gian**, không chia ngẫu nhiên: huấn luyện trên hồ sơ cũ, kiểm
  tra trên hồ sơ mới. Chia ngẫu nhiên sẽ rò rỉ thông tin tương lai và cho kết quả ảo
- Phải kiểm định công bằng theo giới tính, vùng miền, ngành nghề trước khi đưa vào dùng

---

## PHẦN 6 — SÁU TÁC TỬ

| # | Tác tử | Tệp | Có gọi LLM? | Nhiệm vụ |
|---|---|---|---|---|
| 1 | Thu thập | `a1_thu_thap.py` | Không | Nạp dữ liệu, dựng danh sách bằng chứng |
| 2 | Phỏng vấn | `a2_phong_van.py` | Có | Hỏi 6 câu qua Zalo, thu thông tin mềm |
| 3 | Phân tích | `a3_phan_tich.py` | **Không** | Tính đặc trưng, gọi công cụ chấm điểm |
| 4 | Tờ trình | `a4_to_trinh.py` | Có | Diễn đạt dữ kiện thành tờ trình |
| 5 | Kiểm soát | `a5_kiem_soat.py` | Có (lớp 2) | Bắt số bịa và khẳng định vô căn cứ |
| 6 | Giám sát | `a6_giam_sat.py` | Không | Cảnh báo sớm sau giải ngân |

### Vì sao tác tử 3 tuyệt đối không gọi LLM

Đây là câu hỏi hội đồng chắc chắn sẽ hỏi. Ba lý do:

- **Giải trình được** — luật buộc giải thích căn cứ từ chối. Mô hình thống kê chỉ ra
  được "vì doanh thu biến động 40% và mới hoạt động 8 tháng". LLM không chỉ ra được.
- **Ổn định** — cùng hồ sơ phải cho cùng kết quả mọi lần chạy. Mục [2] của `kiem_thu.py`
  kiểm tra đúng điều này, sai số phải nhỏ hơn 1e-12.
- **Kiểm soát thiên lệch** — tính công bằng theo nhóm chỉ đo được trên mô hình thống kê.

### Cách thêm một tác tử mới

1. Tạo `src/agents/a7_ten_moi.py`, viết hàm `chay(tt)` nhận và sửa đối tượng `TrangThai`
2. Thêm vào danh sách `buoc` trong `src/orchestrator.py`
3. Mọi con số mới sinh ra phải gọi `tt.them_bang_chung(...)`, nếu không tác tử 5 sẽ báo lỗi

---

## PHẦN 7 — CƠ CHẾ CHỐNG ẢO GIÁC

Đây là phần đáng khoe nhất của hệ thống. Nó gồm **hai lớp độc lập**.

### Lớp 1 — Đối chiếu số học (tất định, luôn chạy)

Bóc mọi con số trong tờ trình, đối chiếu ngược với khối dữ kiện đã đưa cho LLM.
Số nào không truy được nghĩa là LLM tự nghĩ ra.

Kết quả kiểm thử trên 6 câu bịa cố ý:

| Câu chèn vào | Kết quả |
|---|---|
| "Doanh thu đạt 412 triệu đồng" | **Bắt được** |
| "Xác suất vỡ nợ 4.7%" | **Bắt được** |
| "Hộ hoạt động 11.5 năm" | **Bắt được** |
| "Tỷ lệ dòng tiền 83%" | **Bắt được** |
| "Tài sản bảo đảm 1.800 triệu đồng" | **Bắt được** |
| "Trung bình 340 giao dịch/ngày" | **Bắt được** |
| "Cửa hàng có 27 nhân viên" | Bỏ lọt — xem bên dưới |

Đồng thời **không báo nhầm**: câu diễn đạt lại đúng dữ kiện vẫn cho qua.

**Một chi tiết kỹ thuật đã suýt làm hỏng toàn bộ.** Tiếng Việt dùng dấu chấm phân nhóm
hàng nghìn, nên "2.3" có thể hiểu là 2,3 hoặc 23. Bản đầu tiên nhận nhầm khiến bộ kiểm
soát báo sai hàng loạt. Quy tắc đúng: dấu phân nhóm luôn tách thành đúng 3 chữ số, nên
"1.14" chỉ có thể là 1,14 — không thể là 114. Sửa xong, tỷ lệ truy vết từ 54% lên 100%.

### Lớp 2 — Rà soát ngữ nghĩa (dùng LLM, cần khóa API)

Lớp 1 không phân biệt được "27%" hợp lệ với "27 nhân viên" bịa đặt, vì xét thuần con số
thì cả hai đều là 27. Lớp 2 đọc từng khẳng định và đối chiếu ý nghĩa với dữ kiện.

**Vì sao phải có cả hai:** lớp 1 rẻ, tất định, kiểm toán được và luôn có mặt; lớp 2 bắt
được lỗi ngữ nghĩa nhưng tốn phí và không tất định. Hệ thống không bao giờ phụ thuộc
riêng vào lớp 2 — nếu mất mạng hay hết hạn mức, lớp 1 vẫn bảo vệ.

> Nói được đoạn này trong phỏng vấn là bạn hơn hẳn các đội chỉ nói "chúng em dùng RAG
> để chống ảo giác".

---

## PHẦN 8 — KIỂM THỬ

```bash
python kiem_thu.py
```

Bộ kiểm thử có 16 mục, chia 6 nhóm. Kết quả hiện tại: **16 đạt, 0 hỏng**.

| Nhóm | Kiểm tra điều gì |
|---|---|
| [1] Tầng đặc trưng | Công thức tính đúng, không sinh giá trị vô hạn |
| [2] Tính tất định | Chạy 2 lần ra cùng xác suất, sai số < 1e-12 |
| [3] Chống ảo giác | Bắt được số bịa, không báo nhầm số đúng |
| [4] Chạy hàng loạt | 60 hồ sơ không lỗi, khoảng 32 ms mỗi hồ sơ |
| [5] Phân biệt rủi ro | Nhóm vỡ nợ có xác suất cao gấp 3 lần nhóm trả tốt |
| [6] Giám sát | Có phát sinh cảnh báo khi dữ liệu xấu đi |

Chạy `kiem_thu.py` trước mỗi lần nộp bài hoặc demo. Nó bắt lỗi nhanh hơn mắt người.

---

## PHẦN 9 — DEMO TRƯỚC HỘI ĐỒNG

### Kịch bản 5 phút

| Phút | Làm gì | Nói gì |
|---|---|---|
| 0–1 | Mở màn hình Danh sách hồ sơ | "60 hộ kinh doanh đang chờ thẩm định. Dữ liệu mô phỏng." |
| 1–3 | Chọn một hộ, mở Chi tiết thẩm định | Chỉ vào biểu đồ doanh thu và dòng tiền: "Đây là chỗ lộ ra bất thường." |
| 3–4 | Cuộn tới ô Kiểm soát chống ảo giác | "Mọi con số trong tờ trình đều truy được về bằng chứng. Đây là kiểm tra thật." |
| 4–5 | Mở màn hình Giám sát sau vay | "Đây là phần khác biệt với chấm điểm tín dụng thông thường." |

### Ảnh cần chụp cho proposal và poster A0

1. Màn hình Chi tiết thẩm định — có biểu đồ doanh thu, dòng tiền, điểm rủi ro
2. Biểu đồ đóng góp của từng yếu tố
3. Ô Kiểm soát chống ảo giác hiện tỷ lệ truy vết 100%
4. Màn hình Giám sát sau vay với các cảnh báo đỏ/vàng
5. Đoạn hội thoại phỏng vấn qua Zalo
6. Kết quả `kiem_thu.py` — 16 đạt, 0 hỏng (ảnh này ít đội có, rất đáng chụp)

### Chuẩn bị phòng hờ

- Chạy thử **trước** khi vào phòng, để `@st.cache_resource` đã nạp sẵn
- Chuẩn bị ảnh chụp màn hình dự phòng phòng khi máy chiếu hoặc mạng trục trặc
- Nếu không có mạng, đặt `MICRA_LLM=mock` — hệ thống vẫn chạy đủ 6 tác tử

---

## PHẦN 10 — TỪ BẢN MẪU SANG SẢN PHẨM THẬT

Bảng này để trả lời câu hỏi "còn thiếu gì mới dùng thật được".

| Hạng mục | Bản mẫu hiện tại | Cần cho sản phẩm thật |
|---|---|---|
| Dữ liệu | CSV mô phỏng | Đấu nối API hóa đơn điện tử, Open Banking, CIC |
| Lưu trữ | Trong bộ nhớ | PostgreSQL, có nhật ký kiểm toán không sửa được |
| Xác thực | Không có | Đăng nhập, phân quyền theo vai trò |
| Đồng ý dữ liệu | Giả định | Màn hình đồng ý minh thị, cho phép rút lại |
| Giám sát | Chạy tay | Chạy nền định kỳ, gửi cảnh báo tự động |
| Mô hình | 60 mẫu mô phỏng | 300–500 hồ sơ thật, chia theo thời gian |
| Kiểm định công bằng | Chưa có | Bắt buộc, kiểm định định kỳ theo nhóm |
| Hạ tầng | Máy cá nhân | Đám mây trong nước, đáp ứng yêu cầu lưu trữ dữ liệu cá nhân |
| Pháp lý | Chưa | Đăng ký cơ chế thử nghiệm có kiểm soát |

### Ba việc làm ngay nếu muốn đi tiếp sau cuộc thi

1. **Phỏng vấn một quỹ tín dụng nhân dân** và xin 200–300 hồ sơ cũ đã biết kết quả.
   Không có dữ liệu thật thì mọi thứ còn lại chỉ là bài tập.
2. **Đo chi phí thẩm định thật** của họ. Con số này là toàn bộ luận điểm kinh tế của bạn.
3. **Chạy song song 3 tháng** — agent chấm nhưng người quyết — rồi so kết quả. Đó là
   bằng chứng duy nhất thuyết phục được một tổ chức tín dụng.

---

## PHỤ LỤC — XỬ LÝ SỰ CỐ

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `FileNotFoundError: mo_hinh_rui_ro.joblib` | Chưa huấn luyện | `python -m src.train` |
| `ModuleNotFoundError: xgboost` | Chưa cài, không sao | Hệ thống tự chuyển sang scikit-learn |
| `LoiLLM: Thiếu ANTHROPIC_API_KEY` | `.env` chưa có khóa | Điền khóa, hoặc đặt `MICRA_LLM=mock` |
| Tỷ lệ truy vết dưới 100% | LLM bịa số, hoặc dữ kiện thiếu | Xem `so_khong_truy_duoc`, bổ sung `them_bang_chung` |
| Streamlit không đổi kết quả | Bộ nhớ đệm | Bấm "Rerun", hoặc xóa `@st.cache_resource` |
| Tiếng Việt lỗi font khi mở CSV | Excel không đọc UTF-8 | Tệp đã lưu UTF-8-BOM, mở bằng Excel bản mới |
