"""Tác tử 4 — Soạn tờ trình tín dụng.

Mô hình ngôn ngữ chỉ được phép DIỄN ĐẠT các dữ kiện đưa vào. Mọi con số
phải lấy từ danh sách bằng chứng, không được tự tính, không được làm tròn khác đi.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm import goi_llm
from features import dien_giai, TEN_DAC_TRUNG

TEN_TAC_TU = "4. Soạn tờ trình"

HE_THONG = (
    "Bạn là chuyên viên phân tích tín dụng. Nhiệm vụ: viết tờ trình thẩm định ngắn gọn, "
    "khách quan, bằng tiếng Việt, cho cán bộ tín dụng đọc trong 5 phút.\n"
    "RÀNG BUỘC BẮT BUỘC:\n"
    "1. Chỉ được dùng những con số xuất hiện trong phần DỮ KIỆN. Không tự tính toán thêm.\n"
    "2. Không suy đoán thông tin không có trong dữ kiện.\n"
    "3. Không đưa ra quyết định thay con người — chỉ nêu khuyến nghị.\n"
    "4. Nêu rõ cả điểm mạnh lẫn điểm cần lưu ý."
)


def _tien(x) -> str:
    try:
        return f"{float(x)/1e6:,.0f} triệu đồng"
    except (TypeError, ValueError):
        return str(x)


def _to_trinh_mau(tt, du_kien: list[str]) -> str:
    """Tờ trình dạng mẫu, KHÔNG gọi mô hình ngôn ngữ.

    Dùng khi chạy hàng loạt: 60 hồ sơ mà mỗi hồ sơ một lời gọi API thì vừa
    chậm vừa tốn. Nội dung nghèo hơn về văn phong nhưng mọi con số y hệt bản
    do mô hình ngôn ngữ viết, vì cùng lấy từ một danh sách dữ kiện.
    """
    h, d = tt.ho_so, tt.dac_trung
    lay = lambda tu_khoa: [x for x in du_kien if tu_khoa in x]
    manh = [t for t, g in tt.dong_gop_shap[:4] if g < 0]
    yeu = [t for t, g in tt.dong_gop_shap[:4] if g > 0]
    ten = lambda k: TEN_DAC_TRUNG.get(k, k)

    p = ["TÓM TẮT HỒ SƠ",
         f"Hộ kinh doanh {h.ten_ho} (mã {h.ma_ho}), ngành {h.nganh}, địa bàn {h.tinh_thanh}, "
         f"hoạt động {d['so_nam']:.1f} năm. Doanh thu trung bình {_tien(d['dt_trung_binh'])} mỗi tháng. "
         f"Đề nghị vay {_tien(h.so_tien_de_nghi_vay)}, tương đương {d['vay_tren_dt']:.2f} lần "
         f"doanh thu một tháng.", "",
         "BẰNG CHỨNG HOẠT ĐỘNG KINH DOANH"]
    for k in ("dt_bien_dong", "dt_xu_huong", "nh_tren_dt", "dien_lech_nganh", "gd_ngay", "co_cic"):
        p.append("- " + dien_giai(k, d[k]))
    p += ["",
          "ĐÁNH GIÁ RỦI RO",
          f"Mô hình thống kê tính xác suất vỡ nợ 12 tháng ở mức {tt.xac_suat_vo_no:.1%}, "
          f"xếp hạng rủi ro {tt.hang_rui_ro}. "
          + (f"Các yếu tố làm giảm rủi ro: {', '.join(ten(x) for x in manh)}. " if manh else "")
          + (f"Các yếu tố làm tăng rủi ro: {', '.join(ten(x) for x in yeu)}." if yeu else ""), "",
          "ĐIỂM CẦN LƯU Ý"]
    luu_y = []
    if d["dt_bien_dong"] >= 0.25:
        luu_y.append("- Doanh thu biến động mạnh giữa các tháng, cần xem kỹ dòng tiền mùa thấp điểm.")
    if d["nh_tren_dt"] < 0.55:
        luu_y.append("- Dòng tiền qua ngân hàng lệch đáng kể so với doanh thu hóa đơn.")
    if d["dien_lech_nganh"] < -0.25:
        luu_y.append("- Tiền điện thấp hơn chuẩn ngành, cần đối chiếu lại quy mô hoạt động thực tế.")
    if d["vay_tren_dt"] >= 2.0:
        luu_y.append("- Số tiền đề nghị vay cao so với doanh thu tháng.")
    if not d["co_cic"]:
        luu_y.append("- Chưa có lịch sử tín dụng trên CIC nên thiếu căn cứ đối chiếu bên ngoài.")
    p += (luu_y or ["- Không phát hiện dấu hiệu bất thường trong dữ liệu hiện có."])
    p += ["",
          "KHUYẾN NGHỊ",
          f"{tt.khuyen_nghi['quyet_dinh']}. Hạn mức đề xuất duyệt {_tien(tt.khuyen_nghi['de_xuat_duyet'])} "
          f"trên đề nghị {_tien(tt.khuyen_nghi['de_nghi'])}. "
          f"Quyết định cuối cùng thuộc thẩm quyền của cán bộ tín dụng.",
          "",
          "(Tờ trình dạng mẫu — sinh bằng quy tắc tất định, không dùng mô hình ngôn ngữ.)"]
    return "\n".join(p)


def chay(tt, dung_llm: bool = True) -> None:
    h = tt.ho_so
    d = tt.dac_trung

    du_kien = [
        f"- Hộ kinh doanh: {h.ten_ho} ({h.ma_ho}), ngành {h.nganh}, tại {h.tinh_thanh}",
        f"- Số năm hoạt động: {d['so_nam']:.1f} năm",
        f"- Doanh thu trung bình: {_tien(d['dt_trung_binh'])}/tháng",
        f"- {dien_giai('dt_bien_dong', d['dt_bien_dong'])}",
        f"- {dien_giai('dt_xu_huong', d['dt_xu_huong'])}",
        f"- {dien_giai('nh_tren_dt', d['nh_tren_dt'])}",
        f"- {dien_giai('dien_lech_nganh', d['dien_lech_nganh'])}",
        f"- {dien_giai('gd_ngay', d['gd_ngay'])}",
        f"- {dien_giai('co_cic', d['co_cic'])}",
        f"- Số tiền đề nghị vay: {_tien(h.so_tien_de_nghi_vay)}"
        f" ({d['vay_tren_dt']:.2f} lần doanh thu một tháng)",
        f"- Xác suất vỡ nợ 12 tháng do mô hình tính: {tt.xac_suat_vo_no:.1%}",
        f"- Hạng rủi ro: {tt.hang_rui_ro}",
        f"- Hạn mức đề xuất duyệt: {_tien(tt.khuyen_nghi['de_xuat_duyet'])}",
        f"- Khuyến nghị theo quy tắc: {tt.khuyen_nghi['quyet_dinh']}",
    ]
    for ten, gt in tt.dong_gop_shap[:4]:
        huong = "làm tăng" if gt > 0 else "làm giảm"
        du_kien.append(f"- Yếu tố '{TEN_DAC_TRUNG.get(ten, ten)}' {huong} rủi ro "
                       f"(mức đóng góp {gt:+.3f})")
    for k, v in h.thong_tin_mem.items():
        du_kien.append(f"- Chủ hộ tự khai ({k}): {v}")

    tt.du_kien = du_kien
    if dung_llm:
        tt.to_trinh = goi_llm(
            HE_THONG,
            "DỮ KIỆN:\n" + "\n".join(du_kien) +
            "\n\nViết tờ trình theo đúng 5 phần, mỗi phần có tiêu đề in hoa: "
            "TÓM TẮT HỒ SƠ / BẰNG CHỨNG HOẠT ĐỘNG KINH DOANH / ĐÁNH GIÁ RỦI RO / "
            "ĐIỂM CẦN LƯU Ý / KHUYẾN NGHỊ.",
            max_tokens=1600)
        cach = "mô hình ngôn ngữ"
    else:
        tt.to_trinh = _to_trinh_mau(tt, du_kien)
        cach = "quy tắc tất định"
    tt.ghi(TEN_TAC_TU, f"Đã soạn tờ trình dài {len(tt.to_trinh)} ký tự "
                       f"từ {len(du_kien)} dữ kiện, bằng {cach}.")
