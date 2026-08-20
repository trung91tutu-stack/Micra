#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chẩn đoán vì sao MICRA chưa kết nối được API.

Chạy:  python chan_doan.py
Rồi chụp/copy toàn bộ chữ hiện ra.
"""
import os, sys, glob
from pathlib import Path

GOC = Path(__file__).resolve().parent
loi = []
D = lambda s: print(s)

print("=" * 64)
print("  CHẨN ĐOÁN KẾT NỐI API — MICRA")
print("=" * 64)

# ---------- 1. Python & môi trường ảo ----------
print("\n[1] PYTHON VÀ MÔI TRƯỜNG ẢO")
print(f"    Phiên bản Python : {sys.version.split()[0]}")
print(f"    Đang chạy từ     : {sys.executable}")
trong_venv = (hasattr(sys, "real_prefix") or sys.base_prefix != sys.prefix)
print(f"    Trong môi trường ảo? {'CÓ' if trong_venv else 'KHÔNG'}")
if not trong_venv:
    loi.append("Chưa bật môi trường ảo. Chạy:  .\\.venv\\Scripts\\Activate.ps1")

# ---------- 2. Thư mục làm việc ----------
print("\n[2] THƯ MỤC LÀM VIỆC")
print(f"    Thư mục dự án    : {GOC}")
print(f"    Đang đứng ở      : {Path.cwd()}")
if Path.cwd() != GOC:
    loi.append(f"Bạn đang đứng ở thư mục khác. Chạy:  cd {GOC}")

# ---------- 3. Tìm file .env ----------
print("\n[3] TÌM FILE .env")
env = GOC / ".env"
print(f"    Đường dẫn cần có : {env}")
print(f"    Có tồn tại?        {'CÓ' if env.exists() else 'KHÔNG'}")

if not env.exists():
    loi.append("KHÔNG CÓ FILE .env. Chạy:  copy .env.example .env")
    gan = [p for p in glob.glob(str(GOC / ".env*"))]
    if gan:
        print("    Các file gần giống tìm thấy:")
        for p in gan:
            print(f"       - {Path(p).name}")
        if any(Path(p).name.endswith((".txt", ".example")) for p in gan):
            loi.append("Có file .env.example hoặc .env.txt — hệ thống KHÔNG đọc "
                       "hai file này. Tên file phải đúng là .env")
else:
    print(f"    Kích thước         : {env.stat().st_size} byte")
    print("\n    NỘI DUNG (khóa đã che):")
    co_llm = co_key = False
    for i, dong in enumerate(env.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        d = dong.strip()
        if not d or d.startswith("#"):
            continue
        if "=" in d:
            ten, gt = d.split("=", 1)
            ten, gt = ten.strip(), gt.strip()
            if "KEY" in ten.upper() and gt:
                gt_hien = gt[:8] + "..." + gt[-4:] if len(gt) > 14 else "(quá ngắn?)"
            else:
                gt_hien = gt if gt else "(TRỐNG)"
            print(f"      dòng {i:>2}: {ten} = {gt_hien}")
            if ten == "MICRA_LLM":
                co_llm = True
                if gt.lower() != "gemini":
                    loi.append(f"Dòng MICRA_LLM đang là '{gt}'. Phải sửa thành:  MICRA_LLM=gemini")
            if ten == "GEMINI_API_KEY":
                co_key = bool(gt)
                if gt:
                    # Google đang chuyển định dạng khóa: loại cũ 'AIza...' (Standard key),
                    # loại mới 'AQ.Ab...' (Auth key) do AI Studio cấp từ 2026.
                    # CẢ HAI ĐỀU HỢP LỆ — đừng bắt buộc phải bắt đầu bằng AIza.
                    if gt.startswith("AIzaAQ."):
                        loi.append("Khóa đang bị THỪA tiền tố 'AIza' ở đầu. Xóa 4 ký tự "
                                   "'AIza', chỉ giữ lại phần bắt đầu bằng 'AQ.'")
                    elif not (gt.startswith("AIza") or gt.startswith("AQ.")):
                        loi.append("Khóa không đúng dạng. Khóa Gemini bắt đầu bằng 'AQ.' "
                                   "(loại mới) hoặc 'AIza' (loại cũ).")
                    if gt != gt.strip() or " " in gt or gt.startswith(("'", '"')):
                        loi.append("Khóa có khoảng trắng hoặc dấu nháy thừa. "
                                   "Viết liền: GEMINI_API_KEY=AQ.Ab...")
    if not co_llm:
        loi.append("Trong .env thiếu dòng:  MICRA_LLM=gemini")
    if not co_key:
        loi.append("Trong .env thiếu khóa:  GEMINI_API_KEY=AIza...")

# ---------- 4. Thư viện ----------
print("\n[4] THƯ VIỆN")
for ten, goi in [("python-dotenv", "dotenv"), ("google-genai", "google.genai"),
                 ("streamlit", "streamlit"), ("scikit-learn", "sklearn")]:
    try:
        __import__(goi)
        print(f"    {ten:<16} ĐÃ CÀI")
    except ImportError:
        print(f"    {ten:<16} CHƯA CÀI")
        if ten in ("python-dotenv", "google-genai"):
            loi.append(f"Thiếu thư viện. Chạy:  python -m pip install {ten}")

# ---------- 5. Hệ thống đọc được gì ----------
print("\n[5] HỆ THỐNG THỰC SỰ ĐỌC ĐƯỢC GÌ")
sys.path.insert(0, str(GOC / "src"))
try:
    import llm
    che_do = llm.dang_dung()
    key = os.getenv("GEMINI_API_KEY") or ""
    print(f"    MICRA_LLM đọc được : {che_do}")
    print(f"    Khóa đọc được      : {'CÓ (' + str(len(key)) + ' ký tự)' if key else 'KHÔNG'}")
    if che_do == "mock":
        loi.append("Hệ thống vẫn đang ở chế độ mock — .env chưa được đọc đúng.")
except Exception as e:
    print(f"    LỖI khi nạp src/llm.py: {e}")
    loi.append(f"Không nạp được src/llm.py: {e}")

# ---------- 6. Gọi thử API ----------
print("\n[6] GỌI THỬ API")
try:
    ok, tb = llm.kiem_tra_ket_noi()
    print(f"    {'THÀNH CÔNG' if ok else 'THẤT BẠI'}: {tb[:160]}")
    if not ok:
        loi.append(f"Gọi API thất bại: {tb[:120]}")
except Exception as e:
    print(f"    THẤT BẠI: {e}")
    loi.append(f"Gọi API lỗi: {e}")

# ---------- KẾT LUẬN ----------
print("\n" + "=" * 64)
if not loi:
    print("  KẾT LUẬN: MỌI THỨ ỔN.")
    print("  Chạy tiếp:  streamlit run app.py")
    print("  Rồi bấm Ctrl+F5 trong trình duyệt để nạp lại trang.")
else:
    print(f"  TÌM THẤY {len(loi)} VẤN ĐỀ — sửa theo đúng thứ tự:")
    for i, l in enumerate(loi, 1):
        print(f"\n  {i}. {l}")
print("=" * 64)
