# MICRA — Trợ lý AI thẩm định tín dụng hộ kinh doanh

Hệ thống đa tác tử chạy thật, dùng cho AI Challenge 2026 (Bảng A).
Sáu tác tử nối thành một quy trình duy nhất: thu thập dữ liệu → phỏng vấn chủ hộ →
chấm điểm rủi ro → soạn tờ trình → kiểm soát chống ảo giác → giám sát sau vay.

---

## Chạy trong 5 phút

```bash
# 1. Tạo môi trường riêng
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Cài thư viện
pip install -r requirements.txt

# 3. Cấu hình (bước này có thể bỏ qua — mặc định chạy chế độ mock, không cần khóa API)
cp .env.example .env

# 4. Huấn luyện mô hình rủi ro
python -m src.train

# 5. Kiểm thử toàn hệ thống
python kiem_thu.py

# 6. Thẩm định một hồ sơ trên terminal
python run_cli.py HKD001

# 7. Mở giao diện web
streamlit run app.py
```

---

## Ba nguyên tắc thiết kế — nhớ kỹ khi phỏng vấn

**1. Mô hình ngôn ngữ không bao giờ tạo ra con số.**
Xác suất vỡ nợ do mô hình thống kê tất định tính (`src/scoring.py`). LLM chỉ diễn đạt.
Lý do: quy định ngành ngân hàng buộc phải giải trình được căn cứ từ chối khách hàng,
và cùng một hồ sơ phải cho cùng một kết quả ở mọi lần chạy.

**2. Mọi con số trong tờ trình phải truy được về bằng chứng.**
Tác tử 5 bóc từng con số trong tờ trình và đối chiếu ngược với khối dữ kiện đã đưa cho
LLM. Không truy được nghĩa là LLM tự nghĩ ra. Đây là kiểm tra thật, chạy được, có thể
biểu diễn trực tiếp trước hội đồng.

**3. Con người giữ quyền quyết định.**
Hệ thống rút thời gian cán bộ tín dụng xử lý một hồ sơ từ 3–5 giờ xuống 20–30 phút.
Nó không tự phê duyệt khoản vay.

---

## Cấu trúc mã nguồn

```
micra/
├── data/           bộ dữ liệu mô phỏng 60 hộ kinh doanh
├── models/         mô hình đã huấn luyện (sinh ra sau bước 4)
├── src/
│   ├── schema.py           cấu trúc dữ liệu dùng chung
│   ├── features.py         tầng đặc trưng — TẤT ĐỊNH
│   ├── train.py            huấn luyện, so sánh và chọn mô hình
│   ├── scoring.py          công cụ chấm điểm — TẤT ĐỊNH
│   ├── llm.py              lớp gọi LLM (mock / anthropic / openai)
│   ├── orchestrator.py     bộ điều phối 6 tác tử
│   ├── mo_phong_sau_vay.py sinh dữ liệu sau giải ngân cho demo giám sát
│   └── agents/             6 tác tử
├── app.py          giao diện web Streamlit, 4 màn hình
├── run_cli.py      chạy thử trên terminal
└── kiem_thu.py     bộ tự kiểm thử 16 mục
```

---

## Bốn chế độ mô hình ngôn ngữ

| Chế độ | Khóa API | Chi phí | Dùng khi nào |
|---|---|---|---|
| `mock` | Không cần | Miễn phí | Chạy thử, kiểm thử tự động, demo khi mất mạng |
| `gemini` | `GEMINI_API_KEY` | **Có gói miễn phí** | **Khuyến nghị** — trình diễn thật trước hội đồng |
| `anthropic` | `ANTHROPIC_API_KEY` | Trả phí | Thay thế |
| `openai` | `OPENAI_API_KEY` | Trả phí | Thay thế |

Lấy khóa Gemini miễn phí tại https://aistudio.google.com/apikey — chỉ cần tài khoản Google,
không cần thẻ tín dụng. Kiểm tra kết nối: `python -m src.llm`

Đổi chế độ bằng cách sửa `MICRA_LLM` trong `.env`. Không cần sửa dòng mã nào.

Hạn mức gói miễn phí Gemini (tháng 3/2026): Flash-Lite 15 lượt/phút và 1.000 lượt/ngày;
Flash 10 lượt/phút và 250 lượt/ngày; Pro 5 lượt/phút và 100 lượt/ngày. Hệ thống tự giãn
nhịp theo `MICRA_RPM` và tự chờ rồi thử lại khi gặp lỗi 429.

Ở chế độ `mock`, tác tử 5 chỉ chạy lớp kiểm soát số học. Lớp rà soát ngữ nghĩa cần
khóa API — hệ thống báo rõ trạng thái này thay vì lặng lẽ bỏ qua.

---

## Trung thực học thuật

Toàn bộ dữ liệu trong kho mã này là **dữ liệu mô phỏng**, không phải dữ liệu thật của
bất kỳ cá nhân hay tổ chức nào. Ba tháng dữ liệu ở màn hình giám sát cũng là mô phỏng.
Khi trình bày, phải nói rõ điều này.

Xem `HUONG_DAN_CHI_TIET.md` để biết từng bước chuẩn bị, cách huấn luyện mô hình,
và lộ trình chuyển từ bản mẫu này sang sản phẩm thật.
