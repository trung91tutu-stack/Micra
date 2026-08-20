"""Tác tử 1 — Thu thập dữ liệu.

Trong bản mẫu này, dữ liệu được nạp từ tệp CSV mô phỏng. Khi triển khai thật,
chỉ cần thay phần thân của các hàm nap_* bằng lời gọi API tương ứng
(hóa đơn điện tử, Open Banking, sàn TMĐT, CIC) — phần còn lại giữ nguyên.
"""
from __future__ import annotations
import numpy as np


TEN_TAC_TU = "1. Thu thập dữ liệu"


def chay(tt) -> None:
    h = tt.ho_so
    dt = np.asarray(h.doanh_thu_hoa_don, float)
    nh = np.asarray(h.dong_tien_ngan_hang, float)
    dien = np.asarray(h.tien_dien, float)

    tt.du_lieu_tho = {
        "hoa_don_dien_tu": {"nguon": "Hệ thống hóa đơn điện tử (mô phỏng)",
                            "so_thang": 12, "tong": float(dt.sum())},
        "dong_tien_ngan_hang": {"nguon": "Sao kê tài khoản kinh doanh (mô phỏng)",
                                "so_thang": 12, "tong": float(nh.sum())},
        "tien_dien": {"nguon": "Hóa đơn tiện ích (mô phỏng)",
                      "so_thang": 12, "tong": float(dien.sum())},
        "cic": {"nguon": "CIC (mô phỏng)", "co_lich_su": h.co_cic},
    }

    tt.them_bang_chung("Tổng doanh thu hóa đơn 12 tháng", float(dt.sum()),
                       "Hệ thống hóa đơn điện tử")
    tt.them_bang_chung("Doanh thu trung bình tháng", float(dt.mean()),
                       "Hệ thống hóa đơn điện tử")
    tt.them_bang_chung("Tổng dòng tiền vào tài khoản 12 tháng", float(nh.sum()),
                       "Sao kê ngân hàng")
    tt.them_bang_chung("Tổng tiền điện 12 tháng", float(dien.sum()),
                       "Hóa đơn tiện ích")
    tt.them_bang_chung("Số năm hoạt động", h.so_nam_hoat_dong,
                       "Giấy chứng nhận đăng ký hộ kinh doanh")
    tt.them_bang_chung("Số tiền đề nghị vay", h.so_tien_de_nghi_vay,
                       "Đơn đề nghị vay vốn")

    thieu = [k for k, v in tt.du_lieu_tho.items() if not v]
    tt.ghi(TEN_TAC_TU, f"Đã thu thập 4 nguồn dữ liệu, {len(tt.bang_chung)} mẩu bằng chứng."
                       + (f" Thiếu: {thieu}" if thieu else ""))
