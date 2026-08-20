"""Huấn luyện mô hình rủi ro tín dụng.

Hai mô hình được huấn luyện và so sánh:
  1. Hồi quy logistic  — đơn giản, hệ số đọc được, dùng làm mốc đối chiếu
  2. XGBoost           — bắt được quan hệ phi tuyến, dùng làm mô hình chính

Vì bộ dữ liệu chỉ có 60 dòng, mọi chỉ số hiệu năng đều báo cáo bằng
kiểm định chéo phân tầng lặp lại. KHÔNG bao giờ báo cáo AUC trên tập
huấn luyện — con số đó luôn đẹp và luôn vô nghĩa.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import HoSo
from features import bang_dac_trung, COT, TEN_DAC_TRUNG

GOC = Path(__file__).resolve().parent.parent
DATA = GOC / "data" / "du_lieu_mo_phong_60_ho_kinh_doanh.csv"
MODELS = GOC / "models"


def nap_du_lieu(duong_dan: Path = DATA):
    df = pd.read_csv(duong_dan)
    ho_so = [HoSo.tu_dong_csv(r) for r in df.to_dict("records")]
    X = bang_dac_trung(ho_so)
    y = df["nhan_vo_no"].astype(int).values
    return X, y, df


def _mo_hinh_cay(n, n_pos):
    """Ưu tiên XGBoost. Nếu máy chưa cài, tự chuyển sang GradientBoosting của
    scikit-learn để hệ thống vẫn chạy được — kết quả tương đương ở quy mô này."""
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=180, max_depth=2, learning_rate=0.06,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
            reg_lambda=3.0, gamma=0.2, scale_pos_weight=(n - n_pos) / n_pos,
            eval_metric="logloss", random_state=42), "xgboost"
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        print("[Lưu ý] Chưa cài xgboost — dùng GradientBoosting của scikit-learn thay thế.")
        return GradientBoostingClassifier(
            n_estimators=160, max_depth=2, learning_rate=0.06,
            subsample=0.85, min_samples_leaf=4, random_state=42), "gradient_boosting"


def _danh_gia(mo_hinh, X, y, lan_lap=10):
    """AUC / PR-AUC / Brier bằng kiểm định chéo lặp lại — có sai số chuẩn."""
    aucs, prs, briers = [], [], []
    for lap in range(lan_lap):
        cv = StratifiedKFold(5, shuffle=True, random_state=lap)
        p = cross_val_predict(mo_hinh, X, y, cv=cv, method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, p))
        prs.append(average_precision_score(y, p))
        briers.append(brier_score_loss(y, p))
    f = lambda v: (float(np.mean(v)), float(np.std(v)))
    return {"auc": f(aucs), "pr_auc": f(prs), "brier": f(briers)}


def nap_tu_bang(df):
    """Nạp dữ liệu từ một DataFrame có sẵn thay vì đọc tệp CSV.

    Dùng cho màn hình 'Nạp dữ liệu mới': người dùng tải lên bộ hồ sơ của mình,
    hệ thống huấn luyện lại ngay mà không phải ghi tệp trung gian.
    """
    ho_so = [HoSo.tu_dong_csv(r) for r in df.to_dict("records")]
    X = bang_dac_trung(ho_so)
    y = df["nhan_vo_no"].astype(int).values
    return X, y, df


def huan_luyen(luu: bool = True, im_lang: bool = False, bang=None,
               duong_dan_luu=None):
    """Huấn luyện mô hình.

    bang          : DataFrame để huấn luyện. Bỏ trống thì đọc bộ dữ liệu mặc định.
    duong_dan_luu : nơi ghi tệp mô hình. Bỏ trống thì ghi đè mô hình đang dùng.
    """
    X, y, df = nap_tu_bang(bang) if bang is not None else nap_du_lieu()
    n_pos = int(y.sum())

    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.6))

    cay, ten_cay = _mo_hinh_cay(len(y), n_pos)
    kq = {"logistic": _danh_gia(logit, X, y), ten_cay: _danh_gia(cay, X, y)}

    # Chọn mô hình bằng kiểm định chéo, KHÔNG chọn theo cảm tính.
    # Ở quy mô vài chục mẫu, mô hình đơn giản thường thắng — đó là kết quả
    # trung thực và nên báo cáo đúng như vậy.
    ten_chon = max(kq, key=lambda k: kq[k]["auc"][0])
    goc = logit if ten_chon == "logistic" else cay

    if not im_lang:
        print(f"Dữ liệu: {len(y)} hộ, {n_pos} hộ vỡ nợ ({n_pos/len(y):.0%})")
        print(f"{'Mô hình':<12}{'AUC':>16}{'PR-AUC':>16}{'Brier':>16}")
        for ten, m in kq.items():
            print(f"{ten:<12}"
                  f"{m['auc'][0]:>10.3f} ±{m['auc'][1]:.3f}"
                  f"{m['pr_auc'][0]:>10.3f} ±{m['pr_auc'][1]:.3f}"
                  f"{m['brier'][0]:>10.3f} ±{m['brier'][1]:.3f}")

    # mô hình chính = XGBoost, hiệu chỉnh xác suất để con số đọc được như xác suất thật
    chinh = CalibratedClassifierCV(goc, method="sigmoid",
                                   cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0))
    chinh.fit(X, y)
    tho = goc.fit(X, y)              # bản chưa hiệu chỉnh, dùng để giải thích

    if luu:
        MODELS.mkdir(exist_ok=True)
        dich = Path(duong_dan_luu) if duong_dan_luu else MODELS / "mo_hinh_rui_ro.joblib"
        joblib.dump({"mo_hinh": chinh, "tho": tho, "cot": COT, "loai": ten_chon,
                     "ten_dac_trung": TEN_DAC_TRUNG, "X_nen": X}, dich)
        (MODELS / "chi_so.json").write_text(
            json.dumps({"so_mau": len(y), "so_vo_no": n_pos,
                        "mo_hinh_chon": ten_chon, "ket_qua": kq},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        if not im_lang:
            print(f"\nMô hình được chọn: {ten_chon.upper()} "
                  f"(AUC {kq[ten_chon]['auc'][0]:.3f})")
            print(f"Đã lưu: {dich}")
    kq["_chon"] = ten_chon
    kq["_so_mau"] = int(len(y))
    kq["_so_vo_no"] = n_pos
    return chinh, kq


if __name__ == "__main__":
    huan_luyen()
