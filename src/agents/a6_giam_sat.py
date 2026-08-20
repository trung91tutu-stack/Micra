"""Tác tử 6 — Giám sát sau giải ngân.

Chạy định kỳ hằng tuần trên dữ liệu mới nhất. Đây là công cụ giảm RỦI RO ĐẠO ĐỨC
— vế thứ hai của mô hình Stiglitz–Weiss — với chi phí biên gần bằng không.

Toàn bộ quy tắc đều tất định để có thể kiểm toán và giải trình.
"""
from __future__ import annotations
import numpy as np

TEN_TAC_TU = "6. Giám sát sau vay"

QUY_TAC = [
    ("DOANH_THU_SUT", "Doanh thu 3 tháng gần nhất giảm quá 30% so với 3 tháng trước đó", "Đỏ"),
    ("NGUNG_HOA_DON", "Có tháng gần đây gần như không phát hành hóa đơn", "Đỏ"),
    ("DONG_TIEN_LECH", "Tỷ lệ dòng tiền ngân hàng trên hóa đơn tụt dưới 40%", "Vàng"),
    ("DIEN_GIAM", "Tiền điện giảm quá 25% trong khi doanh thu khai báo không giảm tương ứng", "Vàng"),
    ("BIEN_DONG_TANG", "Mức biến động doanh thu tăng vọt so với giai đoạn thẩm định", "Vàng"),
]


def chay(tt, du_lieu_moi: dict | None = None) -> dict:
    h = tt.ho_so
    dt = np.asarray((du_lieu_moi or {}).get("doanh_thu", h.doanh_thu_hoa_don), float)
    nh = np.asarray((du_lieu_moi or {}).get("dong_tien", h.dong_tien_ngan_hang), float)
    dien = np.asarray((du_lieu_moi or {}).get("tien_dien", h.tien_dien), float)

    canh_bao = []
    truoc, sau = dt[-6:-3].mean(), dt[-3:].mean()
    if truoc > 0 and sau / truoc < 0.70:
        canh_bao.append(("DOANH_THU_SUT", "Đỏ",
                         f"Doanh thu 3 tháng gần nhất giảm {(1-sau/truoc):.0%} so với 3 tháng trước."))
    if (dt[-3:] < dt.mean() * 0.12).any():
        canh_bao.append(("NGUNG_HOA_DON", "Đỏ", "Có tháng gần đây gần như không phát hành hóa đơn."))
    r = nh[-3:].sum() / dt[-3:].sum() if dt[-3:].sum() > 0 else 0
    if r < 0.40:
        canh_bao.append(("DONG_TIEN_LECH", "Vàng",
                         f"Tỷ lệ dòng tiền ngân hàng trên hóa đơn chỉ còn {r:.0%}."))
    d_truoc, d_sau = dien[-6:-3].mean(), dien[-3:].mean()
    if d_truoc > 0 and d_sau / d_truoc < 0.75 and sau / max(truoc, 1e-9) > 0.9:
        canh_bao.append(("DIEN_GIAM", "Vàng",
                         "Tiền điện giảm mạnh trong khi doanh thu khai báo không giảm tương ứng."))
    cv_cu, cv_moi = dt[:6].std() / max(dt[:6].mean(), 1e-9), dt[-6:].std() / max(dt[-6:].mean(), 1e-9)
    if cv_moi > cv_cu * 1.6 and cv_moi > 0.25:
        canh_bao.append(("BIEN_DONG_TANG", "Vàng",
                         f"Biến động doanh thu tăng từ {cv_cu:.0%} lên {cv_moi:.0%}."))

    muc = "Đỏ" if any(c[1] == "Đỏ" for c in canh_bao) else "Vàng" if canh_bao else "Xanh"
    hanh_dong = {
        "Đỏ":  "Liên hệ chủ hộ trong 24 giờ; đề xuất phương án cơ cấu nợ trước khi chuyển nhóm nợ.",
        "Vàng": "Nhắn tin hỏi thăm tình hình kinh doanh; theo dõi sát trong 4 tuần tới.",
        "Xanh": "Không cần can thiệp. Tiếp tục theo dõi định kỳ hằng tuần.",
    }[muc]

    kq = {"muc_canh_bao": muc, "chi_tiet": canh_bao, "hanh_dong": hanh_dong}
    tt.du_lieu_tho["giam_sat"] = kq
    tt.ghi(TEN_TAC_TU, f"Mức cảnh báo {muc}, {len(canh_bao)} dấu hiệu. {hanh_dong}")
    return kq
