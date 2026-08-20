"""Mô phỏng dữ liệu 3 tháng sau giải ngân, phục vụ demo tác tử giám sát.

Lưu ý quan trọng khi thuyết trình: tác tử giám sát vận hành trên DỮ LIỆU MỚI
phát sinh sau khi giải ngân. Trong bản demo chưa có dữ liệu tương lai thật,
nên ta mô phỏng — và phải nói rõ điều đó, không được trình bày như dữ liệu thật.
"""
from __future__ import annotations
import zlib
import numpy as np


def ba_thang_tiep(ho_so, xac_suat_vo_no: float, seed: int | None = None) -> dict:
    """Sinh 3 tháng tiếp theo. Hộ rủi ro càng cao thì càng dễ suy giảm."""
    # hash() của Python thay đổi giữa các lần chạy -> dùng crc32 để tái lập được
    rng = np.random.default_rng(seed if seed is not None else zlib.crc32(ho_so.ma_ho.encode()))
    dt = np.asarray(ho_so.doanh_thu_hoa_don, float)
    nh = np.asarray(ho_so.dong_tien_ngan_hang, float)
    dien = np.asarray(ho_so.tien_dien, float)

    xau = rng.random() < xac_suat_vo_no          # hộ này có thực sự xấu đi không
    he_so = rng.uniform(0.45, 0.70) if xau else rng.uniform(0.94, 1.10)
    ty_le_nh = (nh.sum() / dt.sum()) * (rng.uniform(0.55, 0.80) if xau else rng.uniform(0.95, 1.05))

    dt_moi = dt[-3:] * he_so * rng.normal(1, 0.08, 3).clip(0.7, 1.3)
    if xau and rng.random() < 0.35:
        dt_moi[-1] *= 0.05                       # tháng gần nhất gần như ngừng xuất hóa đơn
    nh_moi = dt_moi * ty_le_nh
    dien_moi = dien[-3:] * (rng.uniform(0.60, 0.80) if xau else rng.uniform(0.95, 1.06))

    return {
        "doanh_thu": np.concatenate([dt[3:], dt_moi]).tolist(),
        "dong_tien": np.concatenate([nh[3:], nh_moi]).tolist(),
        "tien_dien": np.concatenate([dien[3:], dien_moi]).tolist(),
        "thuc_su_xau_di": bool(xau),
    }
