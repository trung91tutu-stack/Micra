"""Tác tử 2 — Phỏng vấn hội thoại qua Zalo.

Đây là tác tử thay thế trực tiếp cho khâu khảo sát thực địa — nơi tập trung
phần lớn chi phí thẩm định C. Nó thu thập THÔNG TIN MỀM mà dữ liệu số không có.

Hai chế độ:
  - tu_dong=True : dùng câu trả lời mô phỏng, để chạy hàng loạt và kiểm thử
  - tu_dong=False: hỏi thật trên terminal hoặc giao diện web
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm import goi_llm

TEN_TAC_TU = "2. Phỏng vấn hội thoại"

CAU_HOI = [
    ("thoi_gian_kinh_doanh", "Chào anh/chị ạ. Em là trợ lý thẩm định của quỹ tín dụng. "
                             "Anh/chị mở cửa hàng ở địa điểm hiện tại được bao lâu rồi ạ?"),
    ("mua_vu",               "Trong năm thì tháng nào anh/chị đông khách nhất, tháng nào ế nhất ạ?"),
    ("muc_dich_vay",         "Lần này anh/chị định dùng khoản vay vào việc gì ạ?"),
    ("ke_hoach_tra_no",      "Anh/chị dự kiến trả nợ từ nguồn nào và trong bao lâu ạ?"),
    ("nhan_su",              "Cửa hàng hiện có bao nhiêu người phụ giúp ạ?"),
    ("kho_khan",             "Thời gian qua việc buôn bán có gặp khó khăn gì đáng kể không ạ?"),
]

HE_THONG = (
    "Bạn là chuyên viên tín dụng của một quỹ tín dụng nhân dân tại Việt Nam, đang phỏng vấn "
    "một hộ kinh doanh nhỏ xin vay vốn. Hãy dùng tiếng Việt đời thường, xưng 'em' gọi 'anh/chị', "
    "thân thiện và ngắn gọn. Tuyệt đối không bịa ra con số. Không hứa hẹn về kết quả xét duyệt."
)


def _tra_loi_mo_phong(khoa: str, ho_so) -> str:
    return {
        "thoi_gian_kinh_doanh": f"Tôi bán ở đây được khoảng {ho_so.so_nam_hoat_dong:.0f} năm rồi.",
        "mua_vu": "Đông nhất là dịp cuối năm và Tết, ế nhất là mấy tháng giữa năm.",
        "muc_dich_vay": "Tôi cần tiền nhập thêm hàng cho mùa cao điểm sắp tới.",
        "ke_hoach_tra_no": "Tôi trả dần từ tiền bán hàng hằng tháng, khoảng một năm là xong.",
        "nhan_su": "Có hai người nhà phụ, lúc đông thì thuê thêm một người.",
        "kho_khan": "Cũng có lúc chậm khách nhưng vẫn xoay được, chưa nợ ai bao giờ.",
    }[khoa]


def chay(tt, tu_dong: bool = True, ham_hoi=None) -> None:
    h = tt.ho_so
    for khoa, cau in CAU_HOI:
        if tu_dong:
            tra_loi = _tra_loi_mo_phong(khoa, h)
        elif ham_hoi is not None:
            tra_loi = ham_hoi(cau)
        else:
            tra_loi = input(f"\n[MICRA] {cau}\n> ").strip()
        tt.hoi_thoai.append({"hoi": cau, "dap": tra_loi})
        h.thong_tin_mem[khoa] = tra_loi

    tom_tat = goi_llm(
        HE_THONG,
        "Tóm tắt cuộc phỏng vấn dưới đây thành các gạch đầu dòng ngắn gọn, "
        "mỗi ý một dòng bắt đầu bằng dấu '-'. Chỉ dùng thông tin có trong hội thoại.\n\n"
        + "\n".join(f"Hỏi: {t['hoi']}\nĐáp: {t['dap']}" for t in tt.hoi_thoai),
        max_tokens=500)
    tt.du_lieu_tho["tom_tat_phong_van"] = tom_tat

    tt.them_bang_chung("Mục đích vay (chủ hộ tự khai)",
                       h.thong_tin_mem.get("muc_dich_vay", ""), "Phỏng vấn qua Zalo")
    tt.them_bang_chung("Kế hoạch trả nợ (chủ hộ tự khai)",
                       h.thong_tin_mem.get("ke_hoach_tra_no", ""), "Phỏng vấn qua Zalo")
    tt.ghi(TEN_TAC_TU, f"Đã hỏi {len(CAU_HOI)} câu, thu {len(h.thong_tin_mem)} thông tin mềm.")
