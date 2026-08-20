# -*- coding: utf-8 -*-
"""Bóc TỪNG BƯỚC TÍNH của một hồ sơ, từ dữ liệu thô đến hạn mức."""
import sys, json; sys.path.insert(0,'src')
import numpy as np, pandas as pd, joblib
from schema import HoSo
from features import tinh_dac_trung, TEN_DAC_TRUNG, COT, DIEN_CHUAN_NGANH
from scoring import phan_hang, han_muc_de_xuat

df = pd.read_csv('data/du_lieu_mo_phong_60_ho_kinh_doanh.csv')
rec = df[df.ma_ho=='HKD023'].iloc[0].to_dict()
h = HoSo.tu_dong_csv(rec)
dt = np.array(h.doanh_thu_hoa_don); nh = np.array(h.dong_tien_ngan_hang); dien = np.array(h.tien_dien)

out = {}
out['ho'] = dict(ma=h.ma_ho, ten=h.ten_ho, nganh=h.nganh, tinh=h.tinh_thanh,
                 nam=h.so_nam_hoat_dong, gd=h.so_giao_dich_ngay, cic=h.co_cic, vay=h.so_tien_de_nghi_vay)
out['dt'] = (dt/1e6).round(1).tolist()
out['nh'] = (nh/1e6).round(1).tolist()
out['dien'] = (dien/1e6).round(2).tolist()

d = tinh_dac_trung(h)
out['dac_trung'] = {k: round(float(v),4) for k,v in d.items()}
# công thức từng đặc trưng
out['ct'] = {
 'dt_trung_binh': f"trung bình 12 tháng = {dt.mean()/1e6:.1f} triệu",
 'dt_bien_dong' : f"độ lệch chuẩn {dt.std()/1e6:.1f} / trung bình {dt.mean()/1e6:.1f} = {d['dt_bien_dong']:.4f}",
 'dt_xu_huong'  : f"TB 6 tháng cuối {dt[6:].mean()/1e6:.1f} / TB 6 tháng đầu {dt[:6].mean()/1e6:.1f} = {d['dt_xu_huong']:.4f}",
 'dt_thang_thap': f"tháng thấp nhất {dt.min()/1e6:.1f} / trung bình {dt.mean()/1e6:.1f} = {d['dt_thang_thap']:.4f}",
 'nh_tren_dt'   : f"tổng dòng tiền {nh.sum()/1e6:.0f} / tổng hóa đơn {dt.sum()/1e6:.0f} = {d['nh_tren_dt']:.4f}",
 'dien_lech_nganh': f"(điện/doanh thu {dien.sum()/dt.sum():.4f} − chuẩn ngành {DIEN_CHUAN_NGANH[h.nganh]:.4f}) / {DIEN_CHUAN_NGANH[h.nganh]:.4f} = {d['dien_lech_nganh']:.4f}",
 'vay_tren_dt'  : f"vay {h.so_tien_de_nghi_vay/1e6:.0f} / doanh thu tháng {dt.mean()/1e6:.1f} = {d['vay_tren_dt']:.4f}",
}

g = joblib.load('models/mo_hinh_rui_ro.joblib')
tho = g['tho']; sc = tho.named_steps['standardscaler']; lr = tho.named_steps['logisticregression']
X = pd.DataFrame([d])[COT]
z = (X.values[0] - sc.mean_) / sc.scale_
w = lr.coef_[0]; b0 = float(lr.intercept_[0])
gop = w*z
out['buoc'] = []
for i,c in enumerate(COT):
    out['buoc'].append(dict(ten=TEN_DAC_TRUNG[c], goc=round(float(X.values[0][i]),4),
        tb=round(float(sc.mean_[i]),4), sd=round(float(sc.scale_[i]),4),
        z=round(float(z[i]),3), w=round(float(w[i]),3), gop=round(float(gop[i]),3)))
logit = float(b0 + gop.sum())
p_tho = 1/(1+np.exp(-logit))
p_hc = float(g['mo_hinh'].predict_proba(X)[0,1])
out['tong'] = dict(b0=round(b0,3), tong_gop=round(float(gop.sum()),3), logit=round(logit,3),
                   p_tho=round(p_tho,4), p_hc=round(p_hc,4), hang=phan_hang(p_hc))
out['hm'] = {k: (round(v/1e6,0) if isinstance(v,float) and v>1000 else v)
             for k,v in han_muc_de_xuat(d,p_hc).items()}
print(json.dumps(out, ensure_ascii=False, indent=1))
