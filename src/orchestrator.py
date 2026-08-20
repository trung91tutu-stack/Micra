"""Bộ điều phối — nơi 6 tác tử được nối lại thành một quy trình duy nhất.

Luồng:
    [1] Thu thập  →  [2] Phỏng vấn  →  [3] Phân tích  →  [4] Tờ trình
                                  →  [5] Kiểm soát   →  [6] Giám sát
                                  →  CÁN BỘ TÍN DỤNG PHÊ DUYỆT

Điểm cần nhớ khi bảo vệ trước hội đồng: bộ điều phối không "thông minh".
Nó chỉ bảo đảm mỗi tác tử chạy đúng thứ tự, nhận đủ đầu vào, và mọi thứ
đều được ghi vào nhật ký để kiểm toán.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import HoSo, TrangThai
from agents import a1_thu_thap, a2_phong_van, a3_phan_tich, a4_to_trinh, a5_kiem_soat, a6_giam_sat


def tham_dinh(ho_so: HoSo, tu_dong_phong_van: bool = True,
              ham_hoi=None, in_tien_trinh: bool = False,
              dung_llm: bool = True, bao_tien_trinh=None) -> TrangThai:
    """Chạy đủ sáu tác tử trên một hồ sơ.

    dung_llm=False  : tác tử 4 soạn tờ trình bằng quy tắc tất định thay vì gọi
                      API. Dùng khi chạy hàng loạt — nhanh hơn hàng trăm lần và
                      không tốn hạn mức. Mọi con số không đổi.
    bao_tien_trinh  : hàm nhận (chi_so, tong, ten_buoc), gọi sau mỗi tác tử.
                      Dùng để vẽ thanh tiến trình trên giao diện.
    """
    tt = TrangThai(ho_so=ho_so)
    buoc = [
        ("Thu thập dữ liệu",   lambda: a1_thu_thap.chay(tt)),
        ("Phỏng vấn chủ hộ",   lambda: a2_phong_van.chay(tt, tu_dong=tu_dong_phong_van, ham_hoi=ham_hoi)),
        ("Phân tích & chấm điểm", lambda: a3_phan_tich.chay(tt)),
        ("Soạn tờ trình",      lambda: a4_to_trinh.chay(tt, dung_llm=dung_llm)),
        ("Kiểm soát chống ảo giác", lambda: a5_kiem_soat.chay(tt)),
        ("Giám sát sau vay",   lambda: a6_giam_sat.chay(tt)),
    ]
    for i, (ten, ham) in enumerate(buoc, 1):
        t0 = time.time()
        ham()
        if in_tien_trinh:
            print(f"  ✓ {ten:<28} {time.time()-t0:5.2f}s")
        if bao_tien_trinh is not None:
            bao_tien_trinh(i, len(buoc), ten)
    return tt


def tham_dinh_hang_loat(danh_sach: list[HoSo]) -> list[TrangThai]:
    return [tham_dinh(h) for h in danh_sach]
