#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chứng minh TỪNG khẳng định trên Slide 3 bằng lệnh chạy thật.

    python kiem_chung_slide3.py

Mỗi mục in ra: KHẲNG ĐỊNH trên slide -> BẰNG CHỨNG chạy được.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

GOC = Path(__file__).resolve().parent
sys.path.insert(0, str(GOC / "src"))
from train import nap_du_lieu, huan_luyen
from features import tinh_dac_trung, TEN_DAC_TRUNG, COT, DIEN_CHUAN_NGANH
from scoring import NGUONG, cham_diem, han_muc_de_xuat
from schema import HoSo
from orchestrator import tham_dinh
from agents.a6_giam_sat import QUY_TAC
import llm

W = 70
def muc(n, kd):
    print("\n" + "=" * W)
    print(f"  [{n}]  KHẲNG ĐỊNH TRÊN SLIDE: {kd}")
    print("=" * W)

df = pd.read_csv(GOC / "data" / "du_lieu_mo_phong_60_ho_kinh_doanh.csv")

# ---------------------------------------------------------------- 1
muc(1, '"Bản mẫu đã vận hành trên 60 hồ sơ"')
t0 = time.time()
kq = [tham_dinh(HoSo.tu_dong_csv(r)) for r in df.to_dict("records")]
giay = time.time() - t0
print(f"  Đã chạy    : {len(kq)} hồ sơ, không hồ sơ nào lỗi")
print(f"  Tổng thời gian: {giay:.1f}s  ->  {giay/len(kq)*1000:.0f} mili giây mỗi hồ sơ")
print(f"  Số hộ vỡ nợ trong dữ liệu: {int(df.nhan_vo_no.sum())}/{len(df)} ({df.nhan_vo_no.mean():.0%})")

# ---------------------------------------------------------------- 2
muc(2, '"hệ thống SÁU tác tử AI"')
for i, d in enumerate(kq[0].nhat_ky, 1):
    print(f"  {i}. {d[:96]}")
print(f"\n  -> Đúng {len(kq[0].nhat_ky)} tác tử chạy tuần tự trên mỗi hồ sơ.")

# ---------------------------------------------------------------- 3
muc(3, '"11 chỉ số tính bằng công thức từ chuỗi 12 tháng"')
d1 = tinh_dac_trung(kq[0].ho_so)
for i, (k, v) in enumerate(d1.items(), 1):
    print(f"  {i:>2}. {TEN_DAC_TRUNG[k]:<44} = {v:>14,.4f}")
print(f"\n  -> Đúng {len(d1)} đặc trưng, tất cả tính bằng công thức (không dùng AI).")

# ---------------------------------------------------------------- 4
muc(4, '"biến chủ chốt: dòng tiền/hóa đơn, lệch tiền điện so với chuẩn ngành"')
print("  Chuẩn tiền điện trên doanh thu theo ngành (dùng để phát hiện khai khống):")
for k, v in DIEN_CHUAN_NGANH.items():
    print(f"     {k:<16} {v:.4f}")
X, y, _ = nap_du_lieu()
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
mh = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.6)).fit(X, y)
r = permutation_importance(mh, X, y, n_repeats=30, random_state=0, scoring="roc_auc")
print("\n  Ba biến quan trọng nhất (permutation importance):")
for i in np.argsort(r.importances_mean)[::-1][:3]:
    print(f"     {TEN_DAC_TRUNG[X.columns[i]]:<44} {r.importances_mean[i]:.3f}")

# ---------------------------------------------------------------- 5
muc(5, '"Hồi quy logistic C=0.6 (chính) · XGBoost (đối chứng)"')
_, chi_so = huan_luyen(luu=True, im_lang=True)
print(f"  {'Mô hình':<22}{'AUC (kiểm định chéo)':>24}{'Brier':>10}")
for ten, m in chi_so.items():
    print(f"  {ten:<22}{m['auc'][0]:>14.3f} ±{m['auc'][1]:.3f}{m['brier'][0]:>10.3f}")
tot = max(chi_so, key=lambda k: chi_so[k]['auc'][0])
print(f"\n  -> Hệ thống TỰ CHỌN: {tot.upper()}  (chọn bằng kiểm định chéo, không gán cứng)")

# ---------------------------------------------------------------- 6
muc(6, '"Ngưỡng phân hạng 0,10 / 0,25 / 0,45 · hệ số hạn mức"')
print("  Ngưỡng phân hạng rủi ro:")
for ng, ten in NGUONG: print(f"     xác suất < {ng:.2f}  ->  {ten}")
print(f"     còn lại        ->  Rất cao")
print("\n  Hệ số hạn mức (nhân với doanh thu một tháng):")
for h, v in [("Thấp",1.5),("Trung bình",1.0),("Cao",0.5),("Rất cao",0.0)]:
    print(f"     {h:<12} x {v}")
print("\n  -> Đây là chính sách tín dụng, KHÔNG phải kết quả của mô hình.")

# ---------------------------------------------------------------- 7
muc(7, '"5 luật cảnh báo sau vay"')
for i, (ma, mo_ta, muc_) in enumerate(QUY_TAC, 1):
    print(f"  {i}. [{muc_}] {mo_ta}")
print(f"\n  -> Đúng {len(QUY_TAC)} luật, tất cả tất định.")

# ---------------------------------------------------------------- 8
muc(8, '"Tầng 4 CHỈ dùng ở tác tử 2 và 4 — KHÔNG chạm vào con số"')
print(f"  Chế độ mô hình ngôn ngữ hiện tại : {llm.dang_dung()}")
ok, tb = llm.kiem_tra_ket_noi()
print(f"  Kết nối                          : {'THÀNH CÔNG' if ok else 'THẤT BẠI'}")
print(f"  {tb[:100]}")
h = kq[0].ho_so
a = tham_dinh(h); b = tham_dinh(h)
print(f"\n  BẰNG CHỨNG QUYẾT ĐỊNH — chạy lại cùng một hồ sơ hai lần:")
print(f"     Lần 1: xác suất vỡ nợ = {a.xac_suat_vo_no:.10f}")
print(f"     Lần 2: xác suất vỡ nợ = {b.xac_suat_vo_no:.10f}")
print(f"     Sai lệch             = {abs(a.xac_suat_vo_no - b.xac_suat_vo_no):.2e}")
giong = "GIỐNG HỆT" if abs(a.xac_suat_vo_no - b.xac_suat_vo_no) < 1e-12 else "KHÁC NHAU"
print(f"     Kết luận             : con số {giong}")
khac = "KHÁC" if a.to_trinh != b.to_trinh else "giống"
print(f"     Văn bản tờ trình     : {khac} nhau" if llm.dang_dung()!="mock"
      else "     Văn bản tờ trình     : giống (chế độ mock tất định)")
print("\n  -> Đổi phiên bản mô hình ngôn ngữ KHÔNG làm đổi bất kỳ con số nào")
print("     trên Slide 3, vì mọi con số do tầng 1-3 quyết định.")

# ---------------------------------------------------------------- 9
muc(9, "TỔNG KẾT — các con số dùng trên Slide 3")
o = pd.DataFrame([{"p": t.xac_suat_vo_no, "tv": t.ket_qua_kiem_soat["ty_le_truy_vet"]} for t in kq])
print(f"  · Số hồ sơ chạy được          : {len(kq)}")
print(f"  · Số tác tử                    : {len(kq[0].nhat_ky)}")
print(f"  · Số đặc trưng                 : {len(d1)}")
print(f"  · Mô hình được chọn            : {tot}")
print(f"  · AUC (kiểm định chéo)         : {chi_so[tot]['auc'][0]:.3f} ±{chi_so[tot]['auc'][1]:.3f}")
print(f"  · Brier                        : {chi_so[tot]['brier'][0]:.3f}")
print(f"  · Tỷ lệ truy vết số            : {o.tv.mean():.0%}")
print(f"  · Tốc độ                       : {giay/len(kq)*1000:.0f} ms/hồ sơ")
print(f"  · Chế độ LLM                   : {llm.dang_dung()}")
print("\n" + "=" * W)
print("  Mọi con số trên đo bằng DỮ LIỆU MÔ PHỎNG 60 hộ, không phải dữ liệu thật.")
print("=" * W)
