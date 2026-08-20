"""Định nghĩa cấu trúc dữ liệu dùng chung cho toàn hệ thống MICRA."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HoSo:
    """Hồ sơ một hộ kinh doanh đề nghị vay vốn."""
    ma_ho: str
    ten_ho: str
    nganh: str
    tinh_thanh: str
    so_nam_hoat_dong: float
    doanh_thu_hoa_don: list[float]      # 12 tháng
    dong_tien_ngan_hang: list[float]    # 12 tháng
    tien_dien: list[float]              # 12 tháng
    so_giao_dich_ngay: int
    co_cic: bool
    so_tien_de_nghi_vay: float
    thong_tin_mem: dict[str, str] = field(default_factory=dict)  # thu từ phỏng vấn

    @classmethod
    def tu_dong_csv(cls, r: dict) -> "HoSo":
        f = lambda p: [float(r[f"{p}_T{i}"]) for i in range(1, 13)]
        return cls(
            ma_ho=r["ma_ho"], ten_ho=r["ten_ho"], nganh=r["nganh"],
            tinh_thanh=r["tinh_thanh"], so_nam_hoat_dong=float(r["so_nam_hoat_dong"]),
            doanh_thu_hoa_don=f("doanh_thu_hoa_don"),
            dong_tien_ngan_hang=f("dong_tien_ngan_hang"),
            tien_dien=f("tien_dien"),
            so_giao_dich_ngay=int(r["so_giao_dich_trung_binh_ngay"]),
            co_cic=(str(r["co_lich_su_tin_dung_CIC"]).strip() == "Có"),
            so_tien_de_nghi_vay=float(r["so_tien_de_nghi_vay"]),
        )


@dataclass
class BangChung:
    """Một mẩu bằng chứng có nguồn gốc truy vết được.

    Mọi con số xuất hiện trong tờ trình BẮT BUỘC phải tồn tại ở đây.
    Đây là nền tảng của cơ chế chống ảo giác.
    """
    ma: str          # ví dụ: BC01
    noi_dung: str    # diễn giải
    gia_tri: Any
    nguon: str       # nguồn dữ liệu gốc


@dataclass
class TrangThai:
    """Trạng thái dùng chung mà các tác tử lần lượt bổ sung vào."""
    ho_so: HoSo
    du_lieu_tho: dict = field(default_factory=dict)
    hoi_thoai: list[dict] = field(default_factory=list)
    dac_trung: dict = field(default_factory=dict)
    bang_chung: list[BangChung] = field(default_factory=list)
    xac_suat_vo_no: float | None = None
    hang_rui_ro: str | None = None
    dong_gop_shap: list[tuple[str, float]] = field(default_factory=list)
    canh_bao: list[str] = field(default_factory=list)
    du_kien: list = field(default_factory=list)
    to_trinh: str = ""
    ket_qua_kiem_soat: dict = field(default_factory=dict)
    khuyen_nghi: dict = field(default_factory=dict)
    nhat_ky: list[str] = field(default_factory=list)

    def ghi(self, tac_tu: str, viec: str) -> None:
        self.nhat_ky.append(f"[{tac_tu}] {viec}")

    def them_bang_chung(self, noi_dung: str, gia_tri, nguon: str) -> BangChung:
        bc = BangChung(f"BC{len(self.bang_chung) + 1:02d}", noi_dung, gia_tri, nguon)
        self.bang_chung.append(bc)
        return bc
