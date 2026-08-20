#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Soi kỹ khóa API Gemini để tìm chỗ hỏng.  Chạy:  python kiem_tra_khoa.py"""
from __future__ import annotations
import os, sys, re
from pathlib import Path

GOC = Path(__file__).resolve().parent
sys.path.insert(0, str(GOC / "src"))

print("=" * 64)
print("  KIỂM TRA KHÓA API GEMINI")
print("=" * 64)

env = GOC / ".env"
if not env.exists():
    print(f"\n  ✗ Không có tệp .env tại {env}")
    print("    Chạy:  copy .env.example .env")
    sys.exit(1)

tho = env.read_text(encoding="utf-8", errors="replace")
dong_khoa = None
for i, d in enumerate(tho.splitlines(), 1):
    if d.strip().startswith("GEMINI_API_KEY"):
        dong_khoa = (i, d)
if not dong_khoa:
    print("\n  ✗ Trong .env không có dòng GEMINI_API_KEY")
    sys.exit(1)

i, d = dong_khoa
raw = d.split("=", 1)[1] if "=" in d else ""
print(f"\n  Dòng {i} trong .env, phần sau dấu = dài {len(raw)} ký tự")

import llm  # nạp .env qua dotenv
k = os.getenv("GEMINI_API_KEY") or ""

loi = []
print("\n" + "-" * 64)
print("  SOI TỪNG LỖI THƯỜNG GẶP")
print("-" * 64)


def kt(ten, ok, ghi=""):
    print(f"  {'✓' if ok else '✗'} {ten}" + (f"   {ghi}" if ghi else ""))
    if not ok:
        loi.append(ten)


kt("Khóa không rỗng", bool(k), f"độ dài {len(k)}")
# Google dùng hai định dạng khóa: loại cũ bắt đầu bằng 'AIza' (dài 39 ký tự)
# và loại mới bắt đầu bằng 'AQ.'. CẢ HAI đều hợp lệ.
kt("Tiền tố hợp lệ ('AQ.' hoặc 'AIza')", k.startswith(("AQ.", "AIza")),
   f"bắt đầu bằng '{k[:6]}'" if k else "")
if k.startswith("AIzaAQ."):
    kt("Không thừa tiền tố 'AIza'", False,
       "khóa đang là AIzaAQ... — xóa 4 ký tự 'AIza' ở đầu, chỉ giữ phần từ 'AQ.'")
kt("Độ dài hợp lý", (len(k) == 39 if k.startswith("AIza") else len(k) >= 30),
   f"đang là {len(k)} ký tự")
kt("Không có dấu nháy bao quanh", not (k.startswith(('"', "'")) or k.endswith(('"', "'"))))
kt("Không thừa khoảng trắng", k == k.strip())
kt("Toàn ký tự ASCII", all(ord(c) < 128 for c in k),
   "có ký tự lạ — thường do dán từ Word làm cong dấu nháy" if not all(ord(c) < 128 for c in k) else "")
kt("Chỉ gồm chữ, số, dấu chấm, gạch ngang, gạch dưới",
   bool(re.fullmatch(r"[A-Za-z0-9._\-]*", k)) if k else False)
kt("Không còn chữ mẫu", "dán" not in k and "khóa" not in k.lower() and "..." not in k)

if k:
    print(f"\n  Khóa hiện tại (che giữa): {k[:8]}...{k[-4:]}")

print("\n" + "-" * 64)
print("  GỌI THỬ API")
print("-" * 64)
ok, tb = llm.kiem_tra_ket_noi()
if ok:
    print("  ✓ THÀNH CÔNG —", tb[:80].replace("\n", " "))
else:
    print("  ✗ THẤT BẠI")
    t = tb.lower()
    if "api_key_invalid" in t or "not valid" in t:
        loi.append("Google từ chối khóa")
        print("\n    Google trả về API_KEY_INVALID. Nghĩa là khóa tồn tại nhưng sai.")
    elif "quota" in t or "429" in t:
        loi.append("Hết hạn mức trong ngày")
        print("\n    Hết hạn mức trong ngày. Chờ sang ngày mới hoặc đổi sang")
        print("    MICRA_MODEL_GEMINI=gemini-2.5-flash-lite")
    elif "chưa cài thư viện" in t or "google-genai" in t:
        loi.append("Chưa cài thư viện google-genai")
        print("\n    Chạy:  .\\.venv\\Scripts\\pip.exe install google-genai")
    else:
        loi.append("Gọi thử API thất bại")
        print("   ", tb[:200])

print("\n" + "=" * 64)
if loi:
    print("  CẦN SỬA:")
    for l in loi:
        print(f"    - {l}")
    print("\n  Cách chắc ăn nhất: tạo khóa MỚI")
    print("    1. Vào  https://aistudio.google.com/apikey")
    print("    2. Xóa khóa cũ, bấm Create API key")
    print("    3. Bấm nút copy (đừng bôi đen bằng chuột — dễ thiếu ký tự)")
    print("    4. Mở .env bằng Notepad, xóa sạch dòng GEMINI_API_KEY rồi gõ lại:")
    print("       GEMINI_API_KEY=AQ....             (không dấu nháy, không khoảng trắng)")
    print("    5. Ctrl+S lưu, đóng Notepad, chạy lại lệnh này")
else:
    print("  KHÓA HOẠT ĐỘNG BÌNH THƯỜNG.")
print("=" * 64)
