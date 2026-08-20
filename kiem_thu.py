#!/usr/bin/env python3
"""Bộ tự kiểm thử MICRA — chạy: python kiem_thu.py

Kiểm tra những thứ dễ hỏng nhất và những thứ giám khảo sẽ hỏi:
  1. Đặc trưng tính đúng công thức
  2. Điểm số tất định (chạy 2 lần ra cùng kết quả)
  3. Bộ kiểm soát BẮT ĐƯỢC số bịa và KHÔNG báo nhầm số đúng
  4. Toàn bộ 60 hồ sơ chạy trót lọt
  5. Xếp hạng rủi ro có phân biệt được nhóm vỡ nợ thật
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parent
sys.path.insert(0, str(GOC / "src"))
from schema import HoSo
from features import tinh_dac_trung
from orchestrator import tham_dinh
from agents import a5_kiem_soat, a6_giam_sat
from mo_phong_sau_vay import ba_thang_tiep

DF = pd.read_csv(GOC / "data" / "du_lieu_mo_phong_60_ho_kinh_doanh.csv")
dat, hong = 0, 0


def kt(ten: str, dieu_kien: bool, ghi_chu: str = ""):
    global dat, hong
    if dieu_kien:
        dat += 1; print(f"  ✓ {ten}" + (f"  ({ghi_chu})" if ghi_chu else ""))
    else:
        hong += 1; print(f"  ✗ {ten}  << HỎNG >>  {ghi_chu}")


print("=" * 70); print("KIỂM THỬ MICRA"); print("=" * 70)

print("\n[1] Tầng đặc trưng")
h = HoSo.tu_dong_csv(DF.iloc[0].to_dict())
d = tinh_dac_trung(h)
dt = np.array(h.doanh_thu_hoa_don)
kt("doanh thu trung bình đúng", abs(d["dt_trung_binh"] - dt.mean()) < 1)
kt("hệ số biến động đúng", abs(d["dt_bien_dong"] - dt.std()/dt.mean()) < 1e-9)
kt("mọi đặc trưng đều hữu hạn", all(np.isfinite(v) for v in d.values()))
kt("đủ 11 đặc trưng", len(d) == 11, f"có {len(d)}")

print("\n[2] Tính tất định của điểm số")
t1 = tham_dinh(h); t2 = tham_dinh(h)
kt("chạy 2 lần ra cùng xác suất", abs(t1.xac_suat_vo_no - t2.xac_suat_vo_no) < 1e-12,
   f"{t1.xac_suat_vo_no:.6f}")
kt("xác suất nằm trong [0,1]", 0 <= t1.xac_suat_vo_no <= 1)
kt("hạn mức không vượt đề nghị",
   t1.khuyen_nghi["de_xuat_duyet"] <= t1.khuyen_nghi["de_nghi"] + 1)

print("\n[3] Bộ kiểm soát chống ảo giác")
goc = t1.to_trinh
kt("tờ trình gốc đạt 100%", t1.ket_qua_kiem_soat["dat"],
   f"{t1.ket_qua_kiem_soat['ty_le_truy_vet']:.0%}")
bia = ["Doanh thu đạt 412 triệu đồng.", "Xác suất vỡ nợ 4.7%.",
       "Hộ hoạt động 11.5 năm.", "Tỷ lệ dòng tiền 83%.",
       "Tài sản bảo đảm 1.800 triệu đồng.", "Trung bình 340 giao dịch/ngày."]
bat = 0
for c in bia:
    t1.to_trinh = goc + "\n" + c
    a5_kiem_soat.chay(t1)
    bat += not t1.ket_qua_kiem_soat["dat"]
kt("bắt được số bịa", bat >= len(bia) - 1, f"{bat}/{len(bia)}")
t1.to_trinh = goc + f"\nDoanh thu {d['dt_trung_binh']/1e6:,.0f} triệu đồng, hoạt động {d['so_nam']:.1f} năm."
a5_kiem_soat.chay(t1)
kt("không báo nhầm số đúng", t1.ket_qua_kiem_soat["dat"],
   str(t1.ket_qua_kiem_soat["so_khong_truy_duoc"]))

print("\n[4] Chạy toàn bộ 60 hồ sơ")
t0 = time.time(); kq = []
for r in DF.to_dict("records"):
    t = tham_dinh(HoSo.tu_dong_csv(r))
    kq.append({"p": t.xac_suat_vo_no, "that": r["nhan_vo_no"],
               "truy_vet": t.ket_qua_kiem_soat["ty_le_truy_vet"],
               "hang": t.hang_rui_ro, "tt": t})
giay = time.time() - t0
o = pd.DataFrame([{k: v for k, v in r.items() if k != "tt"} for r in kq])
kt("không hồ sơ nào lỗi", len(o) == 60)
kt("tốc độ dưới 1 giây mỗi hồ sơ", giay/60 < 1.0, f"{giay/60*1000:.0f} ms/hồ sơ")
kt("mọi tờ trình truy vết 100%", (o.truy_vet == 1.0).all(),
   f"thấp nhất {o.truy_vet.min():.0%}")

print("\n[5] Năng lực phân biệt của mô hình")
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(o["that"], o["p"])
tb0, tb1 = o[o["that"] == 0].p.mean(), o[o["that"] == 1].p.mean()
kt("nhóm vỡ nợ có xác suất cao hơn hẳn", tb1 > tb0 * 2, f"{tb1:.1%} so với {tb0:.1%}")
kt("AUC in-sample hợp lý", 0.85 <= auc <= 1.0, f"{auc:.3f}")

print("\n[6] Tác tử giám sát trên dữ liệu 3 tháng mới")
do, xanh = 0, 0
for i, r in enumerate(kq[:30]):
    t = r["tt"]
    moi = ba_thang_tiep(t.ho_so, t.xac_suat_vo_no)
    g = a6_giam_sat.chay(t, moi)
    do += g["muc_canh_bao"] == "Đỏ"; xanh += g["muc_canh_bao"] == "Xanh"
kt("có phát sinh cảnh báo khi dữ liệu xấu đi", do > 0, f"{do} đỏ / {xanh} xanh trên 30 hộ")

tong = dat + hong
print("\n" + "=" * 70)
print(f"KẾT QUẢ: {dat}/{tong} MỤC ĐẠT  ·  {hong} hỏng")
print(f"Thời điểm chạy: {time.strftime('%d/%m/%Y %H:%M:%S')}")
print("=" * 70)
sys.exit(1 if hong else 0)
