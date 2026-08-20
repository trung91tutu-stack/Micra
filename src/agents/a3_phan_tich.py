"""Tác tử 3 — Phân tích.

QUY TẮC VÀNG: tác tử này KHÔNG gọi mô hình ngôn ngữ. Nó chỉ tính đặc trưng
rồi gọi công cụ chấm điểm tất định. Điểm số không bao giờ do LLM sinh ra.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import tinh_dac_trung, dien_giai
from scoring import cham_diem, han_muc_de_xuat

TEN_TAC_TU = "3. Phân tích"


def chay(tt) -> None:
    tt.dac_trung = tinh_dac_trung(tt.ho_so)
    kq = cham_diem(tt.dac_trung)

    tt.xac_suat_vo_no = kq["xac_suat_vo_no"]
    tt.hang_rui_ro = kq["hang_rui_ro"]
    tt.dong_gop_shap = kq["dong_gop"]
    tt.khuyen_nghi = han_muc_de_xuat(tt.dac_trung, tt.xac_suat_vo_no)

    tt.them_bang_chung("Xác suất vỡ nợ 12 tháng (mô hình XGBoost đã hiệu chỉnh)",
                       round(tt.xac_suat_vo_no, 4), "Mô hình rủi ro nội bộ")
    tt.them_bang_chung("Hạng rủi ro", tt.hang_rui_ro, "Ngưỡng phân hạng theo chính sách tín dụng")
    tt.them_bang_chung("Hạn mức đề xuất duyệt", tt.khuyen_nghi["de_xuat_duyet"],
                       "Quy tắc hạn mức theo chính sách tín dụng")
    for ten, gt in tt.dac_trung.items():
        tt.them_bang_chung(dien_giai(ten, gt), round(float(gt), 4), "Tính từ dữ liệu gốc")

    tt.ghi(TEN_TAC_TU,
           f"Xác suất vỡ nợ {tt.xac_suat_vo_no:.1%}, hạng {tt.hang_rui_ro}. "
           f"Ba yếu tố ảnh hưởng mạnh nhất: "
           + ", ".join(t for t, _ in tt.dong_gop_shap[:3]))
