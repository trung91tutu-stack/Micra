"""Tầng đặc trưng — TẤT TẦN TẬT đều tất định (deterministic).

Nguyên tắc: mô hình ngôn ngữ KHÔNG bao giờ tạo ra con số ở đây.
Mọi giá trị đều tính bằng công thức, có thể kiểm toán và tái lập.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Tỷ lệ tiền điện / doanh thu điển hình theo ngành (dùng để phát hiện khai khống hóa đơn).
# Nguồn: hiệu chỉnh từ chính bộ dữ liệu; khi triển khai thật phải thay bằng
# số liệu khảo sát ngành hoặc trung vị của toàn bộ danh mục khách hàng.
DIEN_CHUAN_NGANH = {
    "Quán ăn": 0.0420, "Tạp hóa": 0.0200, "Thời trang": 0.0210,
    "Cà phê": 0.0620, "Vật liệu XD": 0.0110,
}

TEN_DAC_TRUNG = {
    "dt_trung_binh":   "Doanh thu trung bình tháng",
    "dt_bien_dong":    "Mức biến động doanh thu (CV)",
    "dt_xu_huong":     "Xu hướng doanh thu nửa cuối / nửa đầu năm",
    "dt_thang_thap":   "Tỷ lệ tháng thấp nhất so với trung bình",
    "nh_tren_dt":      "Dòng tiền ngân hàng / Doanh thu hóa đơn",
    "nh_on_dinh":      "Độ ổn định của tỷ lệ dòng tiền theo tháng",
    "dien_lech_nganh": "Mức lệch tiền điện so với chuẩn ngành",
    "so_nam":          "Số năm hoạt động",
    "gd_ngay":         "Số giao dịch trung bình mỗi ngày",
    "vay_tren_dt":     "Số tiền vay / Doanh thu một tháng",
    "co_cic":          "Có lịch sử tín dụng CIC",
}


def tinh_dac_trung(ho_so) -> dict:
    """Tính bộ đặc trưng từ một hồ sơ. Trả về dict tên -> giá trị."""
    dt = np.asarray(ho_so.doanh_thu_hoa_don, dtype=float)
    nh = np.asarray(ho_so.dong_tien_ngan_hang, dtype=float)
    dien = np.asarray(ho_so.tien_dien, dtype=float)

    dt_tb = float(dt.mean())
    ty_le_thang = np.divide(nh, dt, out=np.zeros_like(nh), where=dt > 0)
    dien_tren_dt = float(dien.sum() / dt.sum()) if dt.sum() > 0 else 0.0
    chuan = DIEN_CHUAN_NGANH.get(ho_so.nganh, 0.025)

    return {
        "dt_trung_binh":   dt_tb,
        "dt_bien_dong":    float(dt.std() / dt_tb) if dt_tb else 0.0,
        "dt_xu_huong":     float(dt[6:].mean() / dt[:6].mean()) if dt[:6].mean() else 1.0,
        "dt_thang_thap":   float(dt.min() / dt_tb) if dt_tb else 0.0,
        "nh_tren_dt":      float(nh.sum() / dt.sum()) if dt.sum() else 0.0,
        "nh_on_dinh":      float(1.0 - min(ty_le_thang.std() / max(ty_le_thang.mean(), 1e-9), 1.0)),
        "dien_lech_nganh": float((dien_tren_dt - chuan) / chuan),
        "so_nam":          float(ho_so.so_nam_hoat_dong),
        "gd_ngay":         float(ho_so.so_giao_dich_ngay),
        "vay_tren_dt":     float(ho_so.so_tien_de_nghi_vay / dt_tb) if dt_tb else 0.0,
        "co_cic":          1.0 if ho_so.co_cic else 0.0,
    }


COT = list(TEN_DAC_TRUNG.keys())


def bang_dac_trung(danh_sach_ho_so) -> pd.DataFrame:
    return pd.DataFrame([tinh_dac_trung(h) for h in danh_sach_ho_so])[COT]


def dien_giai(ten: str, gia_tri: float) -> str:
    """Diễn giải một đặc trưng sang câu tiếng Việt để đưa vào tờ trình."""
    if ten == "dt_bien_dong":
        m = "rất ổn định" if gia_tri < 0.15 else "ổn định vừa" if gia_tri < 0.25 else "biến động mạnh"
        return f"Doanh thu {m} (hệ số biến động {gia_tri:.0%})."
    if ten == "nh_tren_dt":
        m = "khớp tốt" if gia_tri > 0.65 else "khớp ở mức trung bình" if gia_tri > 0.5 else "lệch đáng kể"
        return f"Dòng tiền ngân hàng {m} với doanh thu hóa đơn (tỷ lệ {gia_tri:.0%})."
    if ten == "dt_xu_huong":
        m = "tăng" if gia_tri > 1.05 else "đi ngang" if gia_tri > 0.95 else "giảm"
        return f"Doanh thu nửa cuối năm {m} so với nửa đầu ({gia_tri:.2f} lần)."
    if ten == "so_nam":
        return f"Hộ hoạt động được {gia_tri:.1f} năm."
    if ten == "vay_tren_dt":
        m = "hợp lý" if gia_tri < 1.0 else "khá cao" if gia_tri < 2.0 else "rất cao"
        return f"Số tiền đề nghị vay bằng {gia_tri:.2f} lần doanh thu một tháng — mức {m}."
    if ten == "dien_lech_nganh":
        if gia_tri < -0.25:
            return (f"Tiền điện thấp hơn chuẩn ngành {abs(gia_tri):.0%} — dấu hiệu doanh thu "
                    f"hóa đơn có thể cao hơn hoạt động thực tế.")
        return f"Tiền điện lệch {gia_tri:+.0%} so với chuẩn ngành — trong ngưỡng bình thường."
    if ten == "co_cic":
        return "Có lịch sử tín dụng trên CIC." if gia_tri else "Chưa có lịch sử tín dụng trên CIC."
    if ten == "gd_ngay":
        return f"Trung bình {gia_tri:.0f} giao dịch mỗi ngày."
    if ten == "nh_on_dinh":
        return f"Độ ổn định của tỷ lệ dòng tiền theo tháng đạt {gia_tri:.0%}."
    if ten == "dt_thang_thap":
        return f"Tháng thấp điểm chỉ đạt {gia_tri:.0%} doanh thu trung bình."
    if ten == "dt_trung_binh":
        return f"Doanh thu trung bình {gia_tri/1e6:,.0f} triệu đồng/tháng."
    return f"{TEN_DAC_TRUNG.get(ten, ten)}: {gia_tri:.3f}"
