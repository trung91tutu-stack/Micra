"""Công cụ chấm điểm rủi ro — TẤT ĐỊNH.

Đây là ranh giới cứng của hệ thống: mô hình ngôn ngữ chỉ được GỌI hàm này
và đọc kết quả, tuyệt đối không được tự sinh ra điểm số.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

GOC = Path(__file__).resolve().parent.parent
DUONG_DAN = GOC / "models" / "mo_hinh_rui_ro.joblib"

# Ngưỡng phân hạng. Đặt theo khẩu vị rủi ro của tổ chức cho vay,
# KHÔNG phải do mô hình quyết định.
NGUONG = [(0.10, "Thấp"), (0.25, "Trung bình"), (0.45, "Cao")]

_goi = None


def _huan_luyen_lai():
    """Tự huấn luyện lại từ dữ liệu gốc.

    Cần cho môi trường máy chủ: tệp .joblib có thể chưa được tải lên, hoặc
    được tạo bởi một phiên bản scikit-learn khác nên không đọc được. Huấn
    luyện lại trên cùng bộ dữ liệu cho ra đúng mô hình đó, mất vài giây.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train import huan_luyen
    huan_luyen(im_lang=True)
    return joblib.load(DUONG_DAN)


def _nap():
    global _goi
    if _goi is None:
        if not DUONG_DAN.exists():
            _goi = _huan_luyen_lai()
        else:
            try:
                _goi = joblib.load(DUONG_DAN)
            except Exception:
                # tệp hỏng hoặc lệch phiên bản thư viện — dựng lại từ dữ liệu
                _goi = _huan_luyen_lai()
    return _goi


def nap_lai():
    """Xóa mô hình đang giữ trong bộ nhớ để lần chấm điểm sau đọc lại từ đĩa.

    Gọi sau khi huấn luyện lại, nếu không hệ thống vẫn dùng mô hình cũ.
    """
    global _goi
    _goi = None


def phan_hang(p: float) -> str:
    for nguong, ten in NGUONG:
        if p < nguong:
            return ten
    return "Rất cao"


def _dong_gop(g, X):
    """Đóng góp của từng yếu tố vào điểm rủi ro.

    Ưu tiên SHAP (chuẩn ngành, có nền tảng lý thuyết trò chơi). Nếu máy chưa cài
    thư viện shap, dùng phương pháp thay thế tất định: đo mức thay đổi log-odds
    khi thay từng biến bằng trung vị của danh mục. Cùng ý nghĩa diễn giải,
    chỉ khác cách tính.
    """
    cot = g["cot"]

    # Mô hình tuyến tính: đóng góp tính chính xác bằng hệ số × độ lệch chuẩn hóa.
    if g.get("loai") == "logistic":
        try:
            sc, lr = g["tho"].named_steps["standardscaler"], g["tho"].named_steps["logisticregression"]
            z = (X[cot].values[0] - sc.mean_) / sc.scale_
            return sorted([(c, float(w * zi)) for c, w, zi in zip(cot, lr.coef_[0], z)],
                          key=lambda t: abs(t[1]), reverse=True)
        except Exception:
            pass
    try:
        import shap
        sv = shap.TreeExplainer(g["tho"]).shap_values(X)[0]
        return sorted([(c, float(v)) for c, v in zip(cot, sv)],
                      key=lambda t: abs(t[1]), reverse=True)
    except Exception:
        nen = g["X_nen"].median()
        mo = g["mo_hinh"]
        goc = float(mo.predict_proba(X)[0, 1])
        lo = lambda p: np.log(max(p, 1e-6) / max(1 - p, 1e-6))
        ket = []
        for c in cot:
            X2 = X.copy(); X2[c] = nen[c]
            ket.append((c, float(lo(goc) - lo(float(mo.predict_proba(X2)[0, 1])))))
        return sorted(ket, key=lambda t: abs(t[1]), reverse=True)


def cham_diem(dac_trung: dict) -> dict:
    """Trả về xác suất vỡ nợ, hạng rủi ro và đóng góp SHAP của từng yếu tố."""
    g = _nap()
    X = pd.DataFrame([dac_trung])[g["cot"]]

    p = float(g["mo_hinh"].predict_proba(X)[0, 1])

    dong_gop = _dong_gop(g, X)

    return {
        "xac_suat_vo_no": p,
        "hang_rui_ro": phan_hang(p),
        "dong_gop": dong_gop,
        "ten_dac_trung": g["ten_dac_trung"],
    }


def han_muc_de_xuat(dac_trung: dict, p: float) -> dict:
    """Quy tắc hạn mức — cũng tất định, do chính sách tín dụng quy định."""
    dt_thang = dac_trung["dt_trung_binh"]
    de_nghi = dac_trung["vay_tren_dt"] * dt_thang
    he_so = {"Thấp": 1.5, "Trung bình": 1.0, "Cao": 0.5, "Rất cao": 0.0}[phan_hang(p)]
    tran = dt_thang * he_so
    duyet = min(de_nghi, tran)
    duyet = float(np.floor(duyet / 5e6) * 5e6)
    return {
        "de_nghi": float(de_nghi),
        "tran_theo_hang": float(tran),
        "de_xuat_duyet": duyet,
        "quyet_dinh": ("Từ chối" if duyet <= 0
                       else "Đề nghị duyệt" if duyet >= de_nghi * 0.95
                       else "Đề nghị duyệt có điều kiện (giảm hạn mức)"),
    }
