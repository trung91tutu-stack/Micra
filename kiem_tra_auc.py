#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kiểm tra năng lực mô hình rủi ro — báo cáo đầy đủ để đưa vào proposal.

    python kiem_tra_auc.py           # in báo cáo ra màn hình
    python kiem_tra_auc.py --luu     # kèm xuất biểu đồ ROC và SHAP ra thư mục outputs/

Nguyên tắc báo cáo: KHÔNG bao giờ dùng chỉ số trên tập huấn luyện làm kết quả
chính. Mọi con số dưới đây đo bằng kiểm định chéo phân tầng, lặp lại 10 lần.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parent
sys.path.insert(0, str(GOC / "src"))
from train import nap_du_lieu, _mo_hinh_cay
from features import TEN_DAC_TRUNG

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve, confusion_matrix, classification_report)

LUU = "--luu" in sys.argv
X, y, df = nap_du_lieu()
n, npos = len(y), int(y.sum())

print("=" * 70)
print("  BÁO CÁO NĂNG LỰC MÔ HÌNH RỦI RO — MICRA")
print("=" * 70)
print(f"\nDữ liệu : {n} hộ kinh doanh, {npos} hộ vỡ nợ ({npos/n:.1%}), {X.shape[1]} đặc trưng")
print("Nguồn   : dữ liệu MÔ PHỎNG, không phải dữ liệu thật")

MH = {
    "Hồi quy logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.6)),
    _mo_hinh_cay(n, npos)[1]: _mo_hinh_cay(n, npos)[0],
}

print("\n" + "-" * 70)
print("[1] CHỈ SỐ ĐO BẰNG KIỂM ĐỊNH CHÉO (5 phần, lặp 10 lần)")
print("-" * 70)
print(f"{'Mô hình':<22}{'AUC':>16}{'PR-AUC':>16}{'Brier':>15}")
kq = {}
for ten, mh in MH.items():
    a, p_, b = [], [], []
    for lap in range(10):
        cv = StratifiedKFold(5, shuffle=True, random_state=lap)
        pr = cross_val_predict(mh, X, y, cv=cv, method="predict_proba")[:, 1]
        a.append(roc_auc_score(y, pr)); p_.append(average_precision_score(y, pr))
        b.append(brier_score_loss(y, pr))
    kq[ten] = (np.mean(a), np.std(a), np.mean(p_), np.mean(b), pr)
    print(f"{ten:<22}{np.mean(a):>9.3f} ±{np.std(a):.3f}"
          f"{np.mean(p_):>9.3f} ±{np.std(p_):.3f}{np.mean(b):>10.3f}")

tot = max(kq, key=lambda k: kq[k][0])
print(f"\nMô hình tốt nhất: {tot}  (AUC {kq[tot][0]:.3f})")

print("\n" + "-" * 70)
print("[2] SO SÁNH TẬP HUẤN LUYỆN vs KIỂM ĐỊNH CHÉO — vì sao không dùng số đầu")
print("-" * 70)
mh = MH[tot]
mh.fit(X, y)
auc_in = roc_auc_score(y, mh.predict_proba(X)[:, 1])
print(f"  AUC trên tập huấn luyện : {auc_in:.3f}   <-- KHÔNG dùng con số này")
print(f"  AUC kiểm định chéo      : {kq[tot][0]:.3f}   <-- đây mới là con số đáng tin")
print(f"  Chênh lệch              : {auc_in - kq[tot][0]:+.3f}")
print("  Mô hình đã nhìn thấy chính những dòng đó nên chỉ số luôn đẹp và luôn vô nghĩa.")

print("\n" + "-" * 70)
print("[3] MA TRẬN NHẦM LẪN TẠI CÁC NGƯỠNG PHÊ DUYỆT")
print("-" * 70)
pr = kq[tot][4]
print(f"{'Ngưỡng':>8}{'Bắt đúng vỡ nợ':>18}{'Bỏ sót':>10}{'Báo nhầm':>12}{'Độ nhạy':>11}{'Độ chính xác':>15}")
for t in (0.15, 0.20, 0.25, 0.30, 0.40):
    yp = (pr >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp).ravel()
    nhay = tp / (tp + fn) if tp + fn else 0
    cx = tp / (tp + fp) if tp + fp else 0
    print(f"{t:>8.2f}{tp:>18}{fn:>10}{fp:>12}{nhay:>11.0%}{cx:>15.0%}")
print("\n  Ngưỡng là quyết định khẩu vị rủi ro của tổ chức cho vay, không phải của mô hình.")

print("\n" + "-" * 70)
print("[4] MỨC ĐỘ QUAN TRỌNG CỦA TỪNG ĐẶC TRƯNG")
print("-" * 70)
try:
    from sklearn.inspection import permutation_importance
    r = permutation_importance(mh, X, y, n_repeats=30, random_state=0, scoring="roc_auc")
    thu_tu = np.argsort(r.importances_mean)[::-1]
    for i in thu_tu:
        v = r.importances_mean[i]
        thanh = "#" * max(int(v * 200), 0)
        print(f"  {TEN_DAC_TRUNG[X.columns[i]]:<44}{v:>7.3f} {thanh}")
except Exception as e:
    print("  Không tính được:", e)

print("\n" + "-" * 70)
print("[5] XÁC SUẤT DỰ BÁO THEO NHÓM THỰC TẾ")
print("-" * 70)
o = pd.DataFrame({"that": y, "p": pr})
for g, ten in [(0, "Trả tốt"), (1, "Vỡ nợ ")]:
    s = o[o.that == g].p
    print(f"  {ten}: trung bình {s.mean():>6.1%}   thấp nhất {s.min():>6.1%}   cao nhất {s.max():>6.1%}   (n={len(s)})")

if LUU:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (GOC / "outputs").mkdir(exist_ok=True)
    fpr, tpr, _ = roc_curve(y, pr)
    plt.figure(figsize=(5.4, 5))
    plt.plot(fpr, tpr, lw=2.4, color="#035E63", label=f"{tot} (AUC = {kq[tot][0]:.3f})")
    plt.plot([0, 1], [0, 1], "--", lw=1.2, color="#999999", label="Đoán ngẫu nhiên (0.500)")
    plt.xlabel("Tỷ lệ báo nhầm (1 − độ đặc hiệu)"); plt.ylabel("Tỷ lệ bắt đúng (độ nhạy)")
    plt.title("Đường cong ROC — kiểm định chéo 5 phần", fontsize=11)
    plt.legend(loc="lower right", fontsize=9); plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(GOC / "outputs" / "roc.png", dpi=200); plt.close()
    print(f"\n  Đã lưu biểu đồ ROC: {GOC/'outputs'/'roc.png'}")

print("\n" + "=" * 70)
print("  CÂU DÙNG TRONG PROPOSAL:")
print(f"  \"Mô hình đạt AUC {kq[tot][0]:.3f} (±{kq[tot][1]:.3f}) đo bằng kiểm định chéo phân tầng")
print("   5 phần lặp lại 10 lần trên bộ dữ liệu mô phỏng 60 hộ kinh doanh.\"")
print("=" * 70)
