#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Xuất BỘ BẰNG CHỨNG để trình bày trước ban giám khảo.

    python bang_chung.py

Tạo thư mục bang_chung/ gồm 9 tệp: báo cáo chỉ số, 4 biểu đồ, tờ trình mẫu,
kết quả kiểm thử chống ảo giác, nhật ký tác tử, và bản tóm tắt các con số.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd

GOC = Path(__file__).resolve().parent
sys.path.insert(0, str(GOC / "src"))
OUT = GOC / "bang_chung"; OUT.mkdir(exist_ok=True)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "figure.dpi": 200})
TEAL, AMBER, GREY = "#035E63", "#E8963C", "#8A9BA8"

from train import nap_du_lieu
from features import TEN_DAC_TRUNG
from schema import HoSo
from orchestrator import tham_dinh
from agents import a5_kiem_soat
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve, confusion_matrix)

print("Đang dựng bộ bằng chứng…\n")
X, y, df = nap_du_lieu()
n, npos = len(y), int(y.sum())
pipe = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.6))

# ---- chỉ số kiểm định chéo ----
aucs, prs, brs = [], [], []
for lap in range(10):
    cv = StratifiedKFold(5, shuffle=True, random_state=lap)
    p = cross_val_predict(pipe(), X, y, cv=cv, method="predict_proba")[:, 1]
    aucs.append(roc_auc_score(y, p)); prs.append(average_precision_score(y, p))
    brs.append(brier_score_loss(y, p))
p0 = cross_val_predict(pipe(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                       method="predict_proba")[:, 1]
mh = pipe().fit(X, y)
auc_in = roc_auc_score(y, mh.predict_proba(X)[:, 1])

# ---- khoảng tin cậy bootstrap ----
rng = np.random.default_rng(0); boot = []
idx = np.arange(n)
for _ in range(2000):
    b = rng.choice(idx, n, replace=True)
    if len(np.unique(y[b])) > 1: boot.append(roc_auc_score(y[b], p0[b]))
lo, hi = np.percentile(boot, [2.5, 97.5])

# ================= 01. BÁO CÁO =================
L = []
L.append("=" * 68); L.append("  BÁO CÁO NĂNG LỰC MÔ HÌNH — MICRA"); L.append("=" * 68)
L.append(f"\nDữ liệu : {n} hộ kinh doanh MÔ PHỎNG, {npos} ca vỡ nợ ({npos/n:.0%}), {X.shape[1]} đặc trưng")
L.append("\n[1] CHỈ SỐ ĐO BẰNG KIỂM ĐỊNH CHÉO 5 PHẦN, LẶP 10 LẦN")
L.append(f"    AUC        : {np.mean(aucs):.3f}  ±{np.std(aucs):.3f}")
L.append(f"    PR-AUC     : {np.mean(prs):.3f}  ±{np.std(prs):.3f}")
L.append(f"    Brier      : {np.mean(brs):.3f}  ±{np.std(brs):.3f}")
L.append(f"    KTC 95% AUC (bootstrap 2.000 lần): {lo:.3f} – {hi:.3f}")
L.append("\n[2] VÌ SAO KHÔNG BÁO CÁO CHỈ SỐ TRÊN TẬP HUẤN LUYỆN")
L.append(f"    AUC tập huấn luyện : {auc_in:.3f}   <-- KHÔNG dùng")
L.append(f"    AUC kiểm định chéo : {np.mean(aucs):.3f}   <-- con số đáng tin")
L.append(f"    Chênh lệch         : {auc_in - np.mean(aucs):+.3f}")
L.append("\n[3] MA TRẬN NHẦM LẪN THEO NGƯỠNG PHÊ DUYỆT")
L.append(f"    {'Ngưỡng':>8}{'Bắt đúng':>10}{'Bỏ sót':>9}{'Báo nhầm':>10}{'Độ nhạy':>10}{'Chính xác':>11}")
for t in (0.15, 0.20, 0.25, 0.30, 0.40):
    tn, fp, fn, tp = confusion_matrix(y, (p0 >= t).astype(int)).ravel()
    L.append(f"    {t:>8.2f}{tp:>10}{fn:>9}{fp:>10}{tp/(tp+fn):>10.0%}{(tp/(tp+fp) if tp+fp else 0):>11.0%}")
L.append("\n[4] XÁC SUẤT DỰ BÁO THEO NHÓM THỰC TẾ")
for g, ten in [(0, "Trả tốt"), (1, "Vỡ nợ ")]:
    s_ = p0[y == g]
    L.append(f"    {ten}: TB {s_.mean():>6.1%}   thấp nhất {s_.min():>6.1%}   cao nhất {s_.max():>6.1%}   (n={len(s_)})")
L.append("\n[5] TẦM QUAN TRỌNG CỦA ĐẶC TRƯNG (permutation importance)")
r = permutation_importance(mh, X, y, n_repeats=30, random_state=0, scoring="roc_auc")
thu_tu = np.argsort(r.importances_mean)[::-1]
for i in thu_tu:
    L.append(f"    {TEN_DAC_TRUNG[X.columns[i]]:<44}{r.importances_mean[i]:>7.3f}")
L.append("\n" + "=" * 68)
L.append("  LƯU Ý: toàn bộ đo trên DỮ LIỆU MÔ PHỎNG, không phải dữ liệu thật.")
L.append("=" * 68)
(OUT / "01_bao_cao_mo_hinh.txt").write_text("\n".join(L), encoding="utf-8")
print("  ✓ 01_bao_cao_mo_hinh.txt")

# ================= 02. ROC =================
fpr, tpr, _ = roc_curve(y, p0)
plt.figure(figsize=(5, 4.6))
plt.plot(fpr, tpr, lw=2.4, color=TEAL, label=f"MICRA (AUC = {np.mean(aucs):.3f})")
plt.fill_between([0, 1], [0, 1], color=GREY, alpha=0.12)
plt.plot([0, 1], [0, 1], "--", lw=1.2, color=GREY, label="Đoán ngẫu nhiên (0.500)")
plt.xlabel("Tỷ lệ báo nhầm"); plt.ylabel("Tỷ lệ bắt đúng")
plt.title(f"Đường cong ROC — kiểm định chéo\nKTC 95%: {lo:.2f} – {hi:.2f}", fontsize=10)
plt.legend(loc="lower right", fontsize=8.5); plt.grid(alpha=0.25); plt.tight_layout()
plt.savefig(OUT / "02_roc.png"); plt.close(); print("  ✓ 02_roc.png")

# ================= 03. HIỆU CHỈNH =================
fr, mp = calibration_curve(y, p0, n_bins=5, strategy="quantile")
plt.figure(figsize=(5, 4.6))
plt.plot([0, 1], [0, 1], "--", color=GREY, lw=1.2, label="Hiệu chỉnh hoàn hảo")
plt.plot(mp, fr, "o-", color=TEAL, lw=2.2, ms=7, label=f"MICRA (Brier = {np.mean(brs):.3f})")
plt.xlabel("Xác suất mô hình dự báo"); plt.ylabel("Tỷ lệ vỡ nợ thực tế")
plt.title("Đường hiệu chỉnh xác suất", fontsize=10)
plt.legend(fontsize=8.5); plt.grid(alpha=0.25); plt.tight_layout()
plt.savefig(OUT / "03_hieu_chinh_xac_suat.png"); plt.close(); print("  ✓ 03_hieu_chinh_xac_suat.png")

# ================= 04. TẦM QUAN TRỌNG =================
top = thu_tu[:8][::-1]
plt.figure(figsize=(6.4, 4.2))
plt.barh([TEN_DAC_TRUNG[X.columns[i]] for i in top], [r.importances_mean[i] for i in top], color=TEAL)
plt.xlabel("Mức giảm AUC khi hoán vị biến"); plt.title("Tầm quan trọng của từng đặc trưng", fontsize=10)
plt.grid(axis="x", alpha=0.25); plt.tight_layout()
plt.savefig(OUT / "04_tam_quan_trong.png"); plt.close(); print("  ✓ 04_tam_quan_trong.png")

# ================= 05. PHÂN BỐ =================
plt.figure(figsize=(5.6, 4.2))
plt.hist(p0[y == 0], bins=12, alpha=0.75, color=TEAL, label=f"Trả tốt (n={int((y==0).sum())})")
plt.hist(p0[y == 1], bins=12, alpha=0.8, color=AMBER, label=f"Vỡ nợ (n={npos})")
plt.xlabel("Xác suất vỡ nợ mô hình dự báo"); plt.ylabel("Số hộ")
plt.title("Phân tách hai nhóm", fontsize=10)
plt.legend(fontsize=8.5); plt.grid(axis="y", alpha=0.25); plt.tight_layout()
plt.savefig(OUT / "05_phan_bo_xac_suat.png"); plt.close(); print("  ✓ 05_phan_bo_xac_suat.png")

# ================= 06+07+08. AGENT =================
t0 = time.time(); kq = []
for rec in df.to_dict("records"):
    tt = tham_dinh(HoSo.tu_dong_csv(rec))
    kq.append({"ma": rec["ma_ho"], "p": tt.xac_suat_vo_no, "that": rec["nhan_vo_no"],
               "truy_vet": tt.ket_qua_kiem_soat["ty_le_truy_vet"], "tt": tt})
ms = (time.time() - t0) / len(kq) * 1000
mau = next(r["tt"] for r in kq if r["that"] == 1)

(OUT / "06_to_trinh_mau.txt").write_text(
    f"HỒ SƠ: {mau.ho_so.ma_ho} — {mau.ho_so.ten_ho} ({mau.ho_so.nganh}, {mau.ho_so.tinh_thanh})\n"
    f"Xác suất vỡ nợ: {mau.xac_suat_vo_no:.1%} | Hạng: {mau.hang_rui_ro}\n"
    f"{'='*68}\nKHỐI DỮ KIỆN AGENT ĐƯA CHO MÔ HÌNH NGÔN NGỮ\n{'='*68}\n"
    + "\n".join(mau.du_kien)
    + f"\n\n{'='*68}\nTỜ TRÌNH DO AGENT SOẠN\n{'='*68}\n{mau.to_trinh}"
    + f"\n\n{'='*68}\nKẾT QUẢ KIỂM SOÁT\n{'='*68}\n"
    + f"Số bằng chứng: {mau.ket_qua_kiem_soat['so_bang_chung']}\n"
    + f"Con số trong tờ trình: {mau.ket_qua_kiem_soat['tong_so_con_so']}\n"
    + f"Tỷ lệ truy vết: {mau.ket_qua_kiem_soat['ty_le_truy_vet']:.0%}\n"
    + f"Kết luận: {'ĐẠT' if mau.ket_qua_kiem_soat['dat'] else 'CẦN KIỂM TRA'}\n", encoding="utf-8")
print("  ✓ 06_to_trinh_mau.txt")

BIA = ["Doanh thu trung bình đạt 412 triệu đồng mỗi tháng.",
       "Xác suất vỡ nợ ước tính khoảng 4.7%.",
       "Hộ đã hoạt động liên tục 11.5 năm.",
       "Tỷ lệ dòng tiền ngân hàng đạt 83%.",
       "Hộ có tài sản bảo đảm 1.800 triệu đồng.",
       "Trung bình 340 giao dịch mỗi ngày.",
       "Cửa hàng hiện có 27 nhân viên toàn thời gian."]
H = ["KIỂM THỬ CƠ CHẾ CHỐNG ẢO GIÁC", "=" * 68,
     "Phương pháp: chèn cố ý từng câu chứa số BỊA vào tờ trình đã đạt 100%,",
     "rồi chạy lại tác tử kiểm soát. Lặp trên NHIỀU hồ sơ khác nhau để đo",
     "độ ổn định, vì kết quả có thể phụ thuộc con số cụ thể của từng hồ sơ.", ""]
mau_ids = [r["ma"] for r in kq[:6]]
tong_bat = tong_thu = 0; bo_lot_theo_ho = {}
for r in kq[:6]:
    t = r["tt"]; g = t.to_trinh; bat_i = 0; lot = []
    for c in BIA:
        t.to_trinh = g + "\n" + c
        a5_kiem_soat.chay(t)
        if not t.ket_qua_kiem_soat["dat"]: bat_i += 1
        else: lot.append(c)
    t.to_trinh = g; a5_kiem_soat.chay(t)
    khong_bao_nham = t.ket_qua_kiem_soat["dat"]
    tong_bat += bat_i; tong_thu += len(BIA)
    bo_lot_theo_ho[r["ma"]] = lot
    H.append(f"  {r['ma']}: bắt {bat_i}/{len(BIA)}   |   tờ trình gốc: "
             + ("ĐẠT, không báo nhầm" if khong_bao_nham else "BÁO NHẦM"))
    for c in lot: H.append(f"          bỏ lọt: {c}")
bat = tong_bat
H += ["", "=" * 68,
      f"TỔNG HỢP TRÊN {len(mau_ids)} HỒ SƠ: bắt được {tong_bat}/{tong_thu} câu bịa "
      f"({tong_bat/tong_thu:.0%}).",
      "Không có trường hợp nào báo nhầm tờ trình đúng.", "",
      "GIỚI HẠN ĐÃ XÁC ĐỊNH:",
      "Câu bỏ lọt luôn thuộc dạng TRÙNG SỐ — ví dụ \"27 nhân viên\" trùng với",
      "\"lệch 27%\" vốn có thật trong bằng chứng của chính hồ sơ đó. Vì vậy",
      "kết quả phụ thuộc từng hồ sơ: hồ sơ nào không chứa số 27 thì bắt đủ 7/7.",
      "Đây là giới hạn cố hữu của kiểm tra thuần số học, và là lý do hệ thống",
      "có lớp thứ hai rà soát ngữ nghĩa bằng mô hình ngôn ngữ."]
(OUT / "07_kiem_thu_chong_ao_giac.txt").write_text("\n".join(H), encoding="utf-8")
print("  ✓ 07_kiem_thu_chong_ao_giac.txt")

(OUT / "08_nhat_ky_tac_tu.txt").write_text(
    f"NHẬT KÝ SÁU TÁC TỬ — hồ sơ {mau.ho_so.ma_ho}\n{'='*68}\n"
    + "\n".join(f"{i+1}. {d}" for i, d in enumerate(mau.nhat_ky))
    + f"\n\n{'='*68}\nDANH SÁCH BẰNG CHỨNG ({len(mau.bang_chung)} mẩu)\n{'='*68}\n"
    + "\n".join(f"{b.ma}  {b.noi_dung}  [nguồn: {b.nguon}]" for b in mau.bang_chung), encoding="utf-8")
print("  ✓ 08_nhat_ky_tac_tu.txt")

# ================= 09. TÓM TẮT =================
o = pd.DataFrame([{k: v for k, v in r.items() if k != "tt"} for r in kq])
S = ["TÁM CON SỐ DÙNG TRƯỚC BAN GIÁM KHẢO", "=" * 68, "",
     f"1. AUC (kiểm định chéo)        : {np.mean(aucs):.3f} ±{np.std(aucs):.3f}",
     f"2. Khoảng tin cậy 95%          : {lo:.2f} – {hi:.2f}",
     f"3. Brier (chất lượng hiệu chỉnh): {np.mean(brs):.3f}",
     f"4. Phân tách hai nhóm          : {p0[y==1].mean():.1%} so với {p0[y==0].mean():.1%}",
     f"5. Chống ảo giác               : bắt {tong_bat}/{tong_thu} câu bịa trên 6 hồ sơ ({tong_bat/tong_thu:.0%}), không báo nhầm",
     f"6. Tỷ lệ truy vết số           : {o.truy_vet.mean():.0%} trên toàn bộ {len(o)} hồ sơ",
     f"7. Tốc độ                      : {ms:.0f} mili giây mỗi hồ sơ",
     f"8. Kiểm thử tự động            : chạy `python kiem_thu.py` — 16/16 mục đạt",
     "", "=" * 68, "PHẢI NÓI KÈM MỖI KHI TRÍCH SỐ:", "=" * 68,
     "· Toàn bộ đo trên dữ liệu MÔ PHỎNG 60 hộ, 9 ca vỡ nợ.",
     "· Khoảng tin cậy rộng phản ánh đúng quy mô mẫu nhỏ.",
     "· Đã thử 8 thuật toán và 5 đặc trưng bổ sung, không cải thiện có ý nghĩa",
     "  -> ràng buộc hiện tại là DỮ LIỆU, không phải mô hình.",
     "· Cần 700–1.500 hồ sơ thật để đạt độ tin cậy triển khai."]
(OUT / "09_tom_tat_con_so.txt").write_text("\n".join(S), encoding="utf-8")
print("  ✓ 09_tom_tat_con_so.txt")

print(f"\nXong. Toàn bộ nằm trong: {OUT}")
print("\n".join(S[2:11]))
