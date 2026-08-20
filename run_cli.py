#!/usr/bin/env python3
"""Chạy thử MICRA trên terminal.

    python run_cli.py                 # thẩm định hộ đầu tiên trong bộ dữ liệu
    python run_cli.py HKD007          # thẩm định một hộ cụ thể
    python run_cli.py HKD007 --hoi    # tự trả lời phỏng vấn thay chủ hộ
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

GOC = Path(__file__).resolve().parent
sys.path.insert(0, str(GOC / "src"))
from schema import HoSo
from orchestrator import tham_dinh
from llm import dang_dung

DATA = GOC / "data" / "du_lieu_mo_phong_60_ho_kinh_doanh.csv"


def tien(x):
    return f"{x/1e6:,.0f} triệu đ"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    hoi_that = "--hoi" in sys.argv

    df = pd.read_csv(DATA)
    ma = args[0] if args else df.iloc[0]["ma_ho"]
    dong = df[df.ma_ho == ma]
    if dong.empty:
        sys.exit(f"Không tìm thấy {ma}. Ví dụ hợp lệ: {', '.join(df.ma_ho[:5])}")
    ho_so = HoSo.tu_dong_csv(dong.iloc[0].to_dict())

    print("=" * 74)
    print(f"MICRA — THẨM ĐỊNH TÍN DỤNG HỘ KINH DOANH   (LLM: {dang_dung()})")
    print("=" * 74)
    print(f"{ho_so.ma_ho} · {ho_so.ten_ho} · {ho_so.nganh} · {ho_so.tinh_thanh}")
    print(f"Đề nghị vay: {tien(ho_so.so_tien_de_nghi_vay)}\n")

    tt = tham_dinh(ho_so, tu_dong_phong_van=not hoi_that, in_tien_trinh=True)

    print("\n" + "-" * 74)
    print("KẾT QUẢ CHẤM ĐIỂM")
    print("-" * 74)
    print(f"Xác suất vỡ nợ 12 tháng : {tt.xac_suat_vo_no:.1%}")
    print(f"Hạng rủi ro             : {tt.hang_rui_ro}")
    print(f"Đề nghị vay             : {tien(tt.khuyen_nghi['de_nghi'])}")
    print(f"Hạn mức đề xuất duyệt   : {tien(tt.khuyen_nghi['de_xuat_duyet'])}")
    print(f"Khuyến nghị             : {tt.khuyen_nghi['quyet_dinh']}")

    print("\nCÁC YẾU TỐ ẢNH HƯỞNG MẠNH NHẤT (SHAP)")
    from features import TEN_DAC_TRUNG
    for ten, gt in tt.dong_gop_shap[:5]:
        thanh = "█" * min(int(abs(gt) * 28), 28)
        print(f"  {TEN_DAC_TRUNG.get(ten, ten):<42}{gt:+7.3f} {thanh}")

    print("\n" + "-" * 74)
    print("TỜ TRÌNH DO AGENT SOẠN")
    print("-" * 74)
    print(tt.to_trinh)

    k = tt.ket_qua_kiem_soat
    print("\n" + "-" * 74)
    print("KIỂM SOÁT CHỐNG ẢO GIÁC")
    print("-" * 74)
    print(f"Số bằng chứng thu thập  : {k['so_bang_chung']}")
    print(f"Con số trong tờ trình   : {k['tong_so_con_so']}")
    print(f"Tỷ lệ truy vết được     : {k['ty_le_truy_vet']:.0%}")
    print(f"Kết luận                : {'ĐẠT' if k['dat'] else 'CẦN KIỂM TRA THỦ CÔNG'}")
    if k["so_khong_truy_duoc"]:
        print(f"Con số thiếu nguồn      : {', '.join(k['so_khong_truy_duoc'][:10])}")

    gs = tt.du_lieu_tho["giam_sat"]
    print("\n" + "-" * 74)
    print(f"GIÁM SÁT SAU VAY — MỨC {gs['muc_canh_bao'].upper()}")
    print("-" * 74)
    for _, muc, mo_ta in gs["chi_tiet"]:
        print(f"  [{muc}] {mo_ta}")
    print(f"  → {gs['hanh_dong']}")

    print("\n" + "-" * 74)
    print("NHẬT KÝ TÁC TỬ (phục vụ kiểm toán)")
    print("-" * 74)
    for d in tt.nhat_ky:
        print(" ", d)
    print("\n>>> Quyết định phê duyệt cuối cùng thuộc về cán bộ tín dụng. <<<")


if __name__ == "__main__":
    main()
