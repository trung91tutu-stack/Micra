#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chạy và kiểm tra toàn bộ hệ thống MICRA bằng MỘT lệnh.

    python chay_tat_ca.py

Lần lượt: môi trường -> thư viện -> dữ liệu -> đặc trưng -> mô hình -> AUC
-> tác tử -> chống ảo giác -> giám sát -> kết nối LLM. Cuối cùng in bảng tổng kết.
"""
from __future__ import annotations
import sys, time, subprocess
from pathlib import Path

GOC = Path(__file__).resolve().parent
sys.path.insert(0, str(GOC / "src"))
KQ = []


def muc(ten):
    print("\n" + "=" * 68); print("  " + ten); print("=" * 68)


def ghi(ten, ok, chi_tiet=""):
    KQ.append((ten, ok, chi_tiet))
    print(f"  {'✓' if ok else '✗'} {ten}" + (f"   {chi_tiet}" if chi_tiet else ""))


# ---------------------------------------------------------------- 1
muc("1. MÔI TRƯỜNG")
in_venv = sys.base_prefix != sys.prefix
print(f"  Python      : {sys.version.split()[0]}")
print(f"  Chạy từ     : {sys.executable}")
print(f"  Thư mục     : {Path.cwd()}")
ghi("Đang trong môi trường ảo", in_venv,
    "" if in_venv else "-> chạy: .\\.venv\\Scripts\\Activate.ps1")

muc("2. THƯ VIỆN")
for ten, goi, bat_buoc in [("pandas", "pandas", True), ("numpy", "numpy", True),
                           ("scikit-learn", "sklearn", True), ("joblib", "joblib", True),
                           ("python-dotenv", "dotenv", True), ("streamlit", "streamlit", True),
                           ("xgboost", "xgboost", False), ("shap", "shap", False),
                           ("matplotlib", "matplotlib", False), ("google-genai", "google.genai", False)]:
    try:
        __import__(goi); ghi(ten, True, "đã cài")
    except ImportError:
        ghi(ten + (" (bắt buộc)" if bat_buoc else " (tuỳ chọn)"), not bat_buoc,
            "chưa cài" + ("" if bat_buoc else " — hệ thống vẫn chạy"))

# ---------------------------------------------------------------- 3
muc("3. DỮ LIỆU")
import pandas as pd, numpy as np
f = GOC / "data" / "du_lieu_mo_phong_60_ho_kinh_doanh.csv"
if not f.exists():
    ghi("Tìm thấy tệp dữ liệu", False, str(f)); print("\nDỪNG: thiếu dữ liệu."); sys.exit(1)
df = pd.read_csv(f)
ghi("Tìm thấy tệp dữ liệu", True, f"{len(df)} dòng × {len(df.columns)} cột")
ghi("Đủ 60 hộ kinh doanh", len(df) == 60)
ghi("Không có ô trống", df.isna().sum().sum() == 0)
ghi("Tỷ lệ vỡ nợ hợp lý", 0.10 <= df.nhan_vo_no.mean() <= 0.20,
    f"{df.nhan_vo_no.mean():.0%} ({int(df.nhan_vo_no.sum())} hộ)")

# ---------------------------------------------------------------- 4
muc("4. TẦNG ĐẶC TRƯNG (tất định)")
from schema import HoSo
from features import tinh_dac_trung, COT
h = HoSo.tu_dong_csv(df.iloc[0].to_dict())
d1, d2 = tinh_dac_trung(h), tinh_dac_trung(h)
dt = np.array(h.doanh_thu_hoa_don)
ghi("Đủ 11 đặc trưng", len(d1) == 11, f"có {len(d1)}")
ghi("Mọi giá trị hữu hạn", all(np.isfinite(v) for v in d1.values()))
ghi("Tính 2 lần ra cùng kết quả", d1 == d2)
ghi("Doanh thu TB đúng công thức", abs(d1["dt_trung_binh"] - dt.mean()) < 1)
ghi("Hệ số biến động đúng công thức", abs(d1["dt_bien_dong"] - dt.std() / dt.mean()) < 1e-9)

# ---------------------------------------------------------------- 5
muc("5. MÔ HÌNH RỦI RO")
t0 = time.time()
from train import huan_luyen
mo_hinh, chi_so = huan_luyen(luu=True, im_lang=True)
print(f"  Huấn luyện xong sau {time.time()-t0:.1f}s\n")
print(f"  {'Mô hình':<22}{'AUC (kiểm định chéo)':>26}{'Brier':>10}")
for ten, m in chi_so.items():
    print(f"  {ten:<22}{m['auc'][0]:>16.3f} ±{m['auc'][1]:.3f}{m['brier'][0]:>10.3f}")
tot = max(chi_so, key=lambda k: chi_so[k]["auc"][0])
auc = chi_so[tot]["auc"][0]
print()
ghi("Mô hình tốt hơn đoán ngẫu nhiên", auc > 0.5, f"AUC {auc:.3f} so với 0.500")
ghi("AUC đạt mức chấp nhận được (>0.75)", auc > 0.75, f"{tot}")
ghi("Xác suất đã hiệu chỉnh (Brier < 0.15)", chi_so[tot]["brier"][0] < 0.15,
    f"{chi_so[tot]['brier'][0]:.3f}")

# ---------------------------------------------------------------- 6
muc("6. CHẤM ĐIỂM — TÍNH TẤT ĐỊNH")
from scoring import cham_diem, han_muc_de_xuat
k1, k2 = cham_diem(d1), cham_diem(d1)
p = k1["xac_suat_vo_no"]
ghi("Chạy 2 lần ra cùng xác suất", abs(p - k2["xac_suat_vo_no"]) < 1e-12, f"{p:.6f}")
ghi("Xác suất nằm trong [0,1]", 0 <= p <= 1)
hm = han_muc_de_xuat(d1, p)
ghi("Hạn mức không vượt đề nghị", hm["de_xuat_duyet"] <= hm["de_nghi"] + 1,
    f"{hm['de_xuat_duyet']/1e6:.0f} / {hm['de_nghi']/1e6:.0f} triệu")
ghi("Có phân rã đóng góp từng yếu tố", len(k1["dong_gop"]) == 11)

# ---------------------------------------------------------------- 7
muc("7. SÁU TÁC TỬ — CHẠY TOÀN BỘ 60 HỒ SƠ")
from orchestrator import tham_dinh
t0 = time.time(); rows = []
for r in df.to_dict("records"):
    t = tham_dinh(HoSo.tu_dong_csv(r))
    rows.append({"p": t.xac_suat_vo_no, "that": r["nhan_vo_no"],
                 "truy_vet": t.ket_qua_kiem_soat["ty_le_truy_vet"],
                 "canh_bao": t.du_lieu_tho["giam_sat"]["muc_canh_bao"], "tt": t})
giay = time.time() - t0
o = pd.DataFrame([{k: v for k, v in r.items() if k != "tt"} for r in rows])
ghi("60 hồ sơ chạy không lỗi", len(o) == 60)
ghi("Tốc độ dưới 1 giây mỗi hồ sơ", giay / 60 < 1, f"{giay/60*1000:.0f} ms/hồ sơ")
ghi("Mọi tờ trình truy vết 100%", (o.truy_vet == 1.0).all(), f"thấp nhất {o.truy_vet.min():.0%}")
from sklearn.metrics import roc_auc_score
tb0, tb1 = o[o["that"] == 0].p.mean(), o[o["that"] == 1].p.mean()
ghi("Nhóm vỡ nợ có xác suất cao hơn hẳn", tb1 > tb0 * 2, f"{tb1:.1%} so với {tb0:.1%}")

# ---------------------------------------------------------------- 8
muc("8. CHỐNG ẢO GIÁC")
from agents import a5_kiem_soat
tt = rows[0]["tt"]; goc = tt.to_trinh
bia = ["Doanh thu đạt 412 triệu đồng.", "Xác suất vỡ nợ 4.7%.", "Hộ hoạt động 11.5 năm.",
       "Tỷ lệ dòng tiền 83%.", "Tài sản bảo đảm 1.800 triệu đồng.", "Trung bình 340 giao dịch/ngày."]
bat = 0
for c in bia:
    tt.to_trinh = goc + "\n" + c; a5_kiem_soat.chay(tt)
    bat += not tt.ket_qua_kiem_soat["dat"]
ghi("Bắt được câu bịa số", bat >= len(bia) - 1, f"{bat}/{len(bia)}")
tt.to_trinh = goc; a5_kiem_soat.chay(tt)
ghi("Không báo nhầm tờ trình đúng", tt.ket_qua_kiem_soat["dat"])
ghi("Lớp 2 ngữ nghĩa", tt.ket_qua_kiem_soat["lop_2"].get("kich_hoat", False),
    "đang bật" if tt.ket_qua_kiem_soat["lop_2"].get("kich_hoat") else "cần khóa API mới bật")

# ---------------------------------------------------------------- 9
muc("9. GIÁM SÁT SAU VAY")
from agents import a6_giam_sat
from mo_phong_sau_vay import ba_thang_tiep
do_ = xanh = 0
for r in rows[:30]:
    g = a6_giam_sat.chay(r["tt"], ba_thang_tiep(r["tt"].ho_so, r["tt"].xac_suat_vo_no))
    do_ += g["muc_canh_bao"] == "Đỏ"; xanh += g["muc_canh_bao"] == "Xanh"
ghi("Phát sinh cảnh báo khi dữ liệu xấu đi", do_ > 0, f"{do_} đỏ / {xanh} xanh trên 30 hộ")

# ---------------------------------------------------------------- 10
muc("10. KẾT NỐI MÔ HÌNH NGÔN NGỮ")
import llm
che_do = llm.dang_dung()
print(f"  Chế độ hiện tại: {che_do}")
ok, tb = llm.kiem_tra_ket_noi()
ghi(f"Gọi được mô hình ({che_do})", ok, tb[:90].replace("\n", " "))
if che_do == "mock":
    print("  Lưu ý: đang chạy mock. Đặt MICRA_LLM=gemini trong .env để dùng LLM thật.")

# ---------------------------------------------------------------- TỔNG KẾT
dat = sum(1 for _, o_, _ in KQ if o_)
print("\n" + "=" * 68)
print(f"  TỔNG KẾT: {dat}/{len(KQ)} mục đạt")
hong = [t for t, o_, c in KQ if not o_]
if hong:
    print("\n  Chưa đạt:")
    for t in hong:
        print(f"    - {t}")
else:
    print("  Toàn bộ hệ thống hoạt động bình thường.")
print("=" * 68)
print(f"\n  Câu dùng trong proposal:")
print(f'  "Mô hình đạt AUC {auc:.3f} đo bằng kiểm định chéo phân tầng trên bộ')
print(f'   dữ liệu mô phỏng 60 hộ kinh doanh."')
