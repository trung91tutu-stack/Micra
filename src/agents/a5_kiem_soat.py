"""Tác tử 5 — Kiểm soát viên (chống ảo giác). Hai lớp độc lập.

LỚP 1 — Đối chiếu số học (tất định, luôn chạy)
    Mọi con số trong tờ trình phải xuất hiện trong KHỐI DỮ KIỆN đã đưa cho
    mô hình ngôn ngữ. Con số không có ở đó nghĩa là mô hình tự nghĩ ra.

LỚP 2 — Rà soát ngữ nghĩa (dùng mô hình ngôn ngữ, chỉ chạy khi có khóa API)
    Lớp 1 không phân biệt được "27%" hợp lệ với "27 nhân viên" bịa đặt, vì
    xét thuần con số thì cả hai đều là 27. Lớp 2 đọc từng khẳng định và đối
    chiếu ý nghĩa với dữ kiện.

Thiết kế hai lớp là có chủ đích: lớp 1 rẻ, tất định, kiểm toán được và luôn
có mặt; lớp 2 bắt được lỗi ngữ nghĩa nhưng tốn phí và không tất định.
Hệ thống không bao giờ phụ thuộc riêng vào lớp 2.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm import goi_llm, dang_dung

TEN_TAC_TU = "5. Kiểm soát"
BO_QUA = {"1", "2", "3", "4", "5", "12", "2024", "2025", "2026"}
SAI_SO = 0.01


def _cac_so(vb: str) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)*", vb or "")


def _cach_hieu(s: str) -> list[float]:
    """Diễn giải một chuỗi số theo quy ước viết số của tiếng Việt.

    Quy tắc: dấu phân nhóm hàng nghìn luôn tách thành đúng 3 chữ số.
    Vì vậy "1.14" chỉ có thể là số thập phân 1,14 — KHÔNG phải 114.
    Nhận sai chỗ này khiến bộ kiểm soát bỏ lọt số bịa, nên phải làm chặt.
    """
    ra: set[float] = set()
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", s):          # 1.234.567 -> 1234567
        ra.add(float(s.replace(".", "")))
        if s.count(".") == 1:                          # "1.800" cũng có thể là 1,8
            ra.add(float(s.replace(".", ".", 1)))
    elif re.fullmatch(r"\d{1,3}(,\d{3})+", s):          # 1,234,567 -> 1234567
        ra.add(float(s.replace(",", "")))
    elif re.fullmatch(r"\d+[.,]\d+", s):                # 2.3 / 16,5 -> thập phân
        ra.add(float(s.replace(",", ".")))
    else:
        try:
            ra.add(float(s))
        except ValueError:
            pass
    return list(ra)


def _kho_so(tt) -> list[float]:
    """Kho số hợp lệ = số trong khối dữ kiện + số trong danh sách bằng chứng."""
    kho: list[float] = []
    for dong in getattr(tt, "du_kien", []):
        for s in _cac_so(dong):
            kho += _cach_hieu(s)
    for bc in tt.bang_chung:
        v = bc.gia_tri
        if isinstance(v, (int, float)):
            x = float(v)
            kho += [x, abs(x), x/1e6, x/1e3, round(abs(x)*100, 4),
                    round(abs(x), 3), round(abs(x), 2), round(abs(x), 1)]
        else:
            for s in _cac_so(str(v)):
                kho += _cach_hieu(s)
    return kho


def _khop(x: float, kho: list[float]) -> bool:
    return any(abs(x - k) <= max(abs(k) * SAI_SO, 0.005) for k in kho)


def _lop_1(tt) -> dict:
    kho = _kho_so(tt)
    thieu, tong = [], 0
    for s in _cac_so(tt.to_trinh):
        if s in BO_QUA:
            continue
        tong += 1
        cach = [n for n in _cach_hieu(s) if n not in (0, 1)]
        if cach and not any(_khop(n, kho) for n in cach):
            thieu.append(s)
    thieu = sorted(set(thieu))
    return {"tong_so_con_so": tong, "so_khong_truy_duoc": thieu,
            "ty_le_truy_vet": round(1 - len(thieu)/tong, 3) if tong else 1.0,
            "dat": not thieu}


HE_THONG_2 = (
    "Bạn là kiểm soát viên tuân thủ của tổ chức tín dụng. Nhiệm vụ duy nhất: phát hiện "
    "những khẳng định trong tờ trình KHÔNG được dữ kiện hỗ trợ. Tuyệt đối không sửa văn bản, "
    "không bình luận về chất lượng. Chỉ trả lời bằng JSON."
)


def _lop_2(tt) -> dict:
    if dang_dung() == "mock":
        return {"kich_hoat": False,
                "ly_do": "Đang chạy chế độ mock — lớp ngữ nghĩa cần khóa API để hoạt động."}
    try:
        tra = goi_llm(
            HE_THONG_2,
            "DỮ KIỆN ĐƯỢC PHÉP DÙNG:\n" + "\n".join(getattr(tt, "du_kien", [])) +
            "\n\nTỜ TRÌNH CẦN KIỂM TRA:\n" + tt.to_trinh +
            '\n\nTrả về đúng JSON dạng: {"vi_pham": [{"cau": "...", "ly_do": "..."}]}. '
            'Nếu không có vi phạm, trả về {"vi_pham": []}.',
            max_tokens=900)
        m = re.search(r"\{.*\}", tra, re.S)
        vp = json.loads(m.group(0))["vi_pham"] if m else []
        return {"kich_hoat": True, "vi_pham": vp, "dat": len(vp) == 0}
    except Exception as e:
        return {"kich_hoat": False, "ly_do": f"Lỗi khi gọi mô hình ngôn ngữ: {e}"}


def chay(tt) -> None:
    l1 = _lop_1(tt)
    l2 = _lop_2(tt)
    dat = l1["dat"] and l2.get("dat", True)

    tt.ket_qua_kiem_soat = {**l1, "so_bang_chung": len(tt.bang_chung),
                            "lop_2": l2, "dat": dat}
    if not l1["dat"]:
        tt.canh_bao.append(
            "Tờ trình chứa con số không truy được về dữ kiện: "
            + ", ".join(l1["so_khong_truy_duoc"][:8])
            + ". Cán bộ tín dụng cần kiểm tra thủ công trước khi duyệt.")
    for v in l2.get("vi_pham", []):
        tt.canh_bao.append(f"Khẳng định không có căn cứ: “{v.get('cau','')}” — {v.get('ly_do','')}")

    mo_ta = (f"Lớp 1: đối chiếu {l1['tong_so_con_so']} con số, truy vết {l1['ty_le_truy_vet']:.0%}. "
             + ("Lớp 2: " + (f"{len(l2.get('vi_pham', []))} vi phạm ngữ nghĩa."
                             if l2.get("kich_hoat") else "chưa kích hoạt.")))
    tt.ghi(TEN_TAC_TU, mo_ta + (" ĐẠT." if dat else " CẦN KIỂM TRA THỦ CÔNG."))
