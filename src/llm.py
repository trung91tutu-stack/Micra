"""Lớp trừu tượng gọi mô hình ngôn ngữ.

Bốn chế độ, đặt qua biến MICRA_LLM trong tệp .env:

  mock      — không cần khóa API, ghép văn bản từ chính dữ kiện đầu vào.
              Dùng để chạy thử, kiểm thử tự động và demo khi mất mạng.
  gemini    — Google Gemini. CÓ GÓI MIỄN PHÍ, khuyến nghị cho sinh viên.
  anthropic — Claude, cần ANTHROPIC_API_KEY (trả phí)
  openai    — GPT, cần OPENAI_API_KEY (trả phí)

Toàn bộ phần còn lại của hệ thống chỉ biết đến hàm goi_llm(). Đổi nhà cung cấp
không phải sửa một dòng nào trong sáu tác tử.
"""
from __future__ import annotations
import os, sys, time, random, textwrap

try:                                    # nạp .env nếu có thư viện python-dotenv
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

NHA_CUNG_CAP = os.getenv("MICRA_LLM", "mock").lower().strip()
MODEL_GEMINI = os.getenv("MICRA_MODEL_GEMINI", "gemini-2.5-flash")
MODEL_ANTHROPIC = os.getenv("MICRA_MODEL_ANTHROPIC", "claude-sonnet-5")
MODEL_OPENAI = os.getenv("MICRA_MODEL_OPENAI", "gpt-4.1-mini")

# Gói miễn phí của Gemini giới hạn số lượt gọi mỗi phút. Tự giãn nhịp để
# không dính lỗi 429. Xem bảng hạn mức trong HUONG_DAN_CAI_DAT.docx.
RPM = int(os.getenv("MICRA_RPM", "10"))
_lan_goi_cuoi = 0.0


class LoiLLM(RuntimeError):
    pass


def _gian_nhip() -> None:
    """Bảo đảm hai lượt gọi cách nhau đủ xa so với hạn mức mỗi phút."""
    global _lan_goi_cuoi
    if RPM <= 0:
        return
    toi_thieu = 60.0 / RPM
    cho = toi_thieu - (time.time() - _lan_goi_cuoi)
    if cho > 0:
        time.sleep(cho)
    _lan_goi_cuoi = time.time()


def _thu_lai(ham, so_lan: int = 5):
    """Gặp lỗi hạn mức (429) thì chờ tăng dần rồi thử lại: 2s, 4s, 8s, 16s."""
    for lan in range(so_lan):
        try:
            return ham()
        except Exception as e:
            thong_bao = str(e).lower()
            het_han_muc = ("429" in thong_bao or "quota" in thong_bao
                           or "rate limit" in thong_bao or "resource_exhausted" in thong_bao)
            if not het_han_muc or lan == so_lan - 1:
                raise
            cho = 2 ** (lan + 1) + random.uniform(0, 1)
            print(f"   [LLM] Chạm hạn mức, chờ {cho:.0f}s rồi thử lại "
                  f"(lần {lan + 1}/{so_lan - 1})…")
            time.sleep(cho)


def _gemini(he_thong: str, nguoi_dung: str, max_tokens: int) -> str:
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise LoiLLM("Chưa cài thư viện. Chạy: pip install google-genai") from e
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise LoiLLM("Thiếu GEMINI_API_KEY trong tệp .env. "
                     "Lấy khóa miễn phí tại https://aistudio.google.com/apikey")

    client = genai.Client(api_key=key)

    # QUAN TRỌNG — Gemini 2.5 là mô hình "biết suy nghĩ": số token dùng cho phần
    # suy nghĩ nội bộ ĐƯỢC TÍNH VÀO max_output_tokens. Nếu để ngân sách sát nút,
    # model tiêu gần hết cho phần suy nghĩ rồi chỉ kịp in ra một đoạn cụt giữa câu.
    # Hai biện pháp: (1) tắt hẳn phần suy nghĩ vì tác vụ này chỉ là diễn đạt lại
    # dữ kiện, không cần suy luận; (2) nới rộng ngân sách gấp ba cho chắc.
    ngan_sach = max(int(max_tokens * 3), 3000)

    def _tao(tat_suy_nghi: bool):
        cfg = {
            "system_instruction": he_thong,
            "max_output_tokens": ngan_sach,
            "temperature": 0.3,
        }
        if tat_suy_nghi:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        _gian_nhip()
        return client.models.generate_content(
            model=MODEL_GEMINI, contents=nguoi_dung,
            config=types.GenerateContentConfig(**cfg))

    def goi():
        try:
            r = _tao(tat_suy_nghi=True)
        except Exception as e:
            # gemini-2.5-pro không cho tắt hẳn phần suy nghĩ -> gọi lại kiểu thường
            if "thinking" in str(e).lower() or "budget" in str(e).lower():
                r = _tao(tat_suy_nghi=False)
            else:
                raise

        vb = (r.text or "").strip()
        if not vb:
            ly_do = ""
            try:
                ly_do = str(r.candidates[0].finish_reason)
            except Exception:
                pass
            raise LoiLLM(
                "Gemini trả về rỗng"
                + (f" (finish_reason={ly_do})" if ly_do else "")
                + ". Thường do hết ngân sách token cho phần suy nghĩ — "
                  "thử đổi MICRA_MODEL_GEMINI=gemini-2.5-flash-lite trong .env.")
        return vb

    return _thu_lai(goi)


def _anthropic(he_thong: str, nguoi_dung: str, max_tokens: int) -> str:
    try:
        import anthropic
    except ImportError as e:
        raise LoiLLM("Chưa cài thư viện. Chạy: pip install anthropic") from e
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise LoiLLM("Thiếu ANTHROPIC_API_KEY trong tệp .env")
    client = anthropic.Anthropic(api_key=key)

    def goi():
        _gian_nhip()
        r = client.messages.create(
            model=MODEL_ANTHROPIC, max_tokens=max_tokens, system=he_thong,
            temperature=0.3, messages=[{"role": "user", "content": nguoi_dung}])
        return "".join(b.text for b in r.content if b.type == "text").strip()

    return _thu_lai(goi)


def _openai(he_thong: str, nguoi_dung: str, max_tokens: int) -> str:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise LoiLLM("Chưa cài thư viện. Chạy: pip install openai") from e
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise LoiLLM("Thiếu OPENAI_API_KEY trong tệp .env")
    client = OpenAI(api_key=key)

    def goi():
        _gian_nhip()
        r = client.chat.completions.create(
            model=MODEL_OPENAI, max_tokens=max_tokens, temperature=0.3,
            messages=[{"role": "system", "content": he_thong},
                      {"role": "user", "content": nguoi_dung}])
        return (r.choices[0].message.content or "").strip()

    return _thu_lai(goi)


def _mock(he_thong: str, nguoi_dung: str, max_tokens: int) -> str:
    """Bản mô phỏng: ghép lại chính các dữ kiện được đưa vào.

    Chủ ý thiết kế: bản mock KHÔNG tự nghĩ ra con số nào. Nhờ vậy khi chạy ở
    chế độ mock, tác tử kiểm soát vẫn hoạt động đúng và kết quả vẫn hợp lệ.
    """
    dong = [d.strip() for d in nguoi_dung.splitlines() if d.strip().startswith(("-", "•"))]
    than = "\n".join(dong) if dong else textwrap.shorten(nguoi_dung, 600)
    return (f"{than}\n\n(Văn bản do chế độ mock ghép từ dữ kiện đầu vào. "
            f"Đặt MICRA_LLM=gemini trong tệp .env để dùng mô hình ngôn ngữ thật.)")


_BANG = {"gemini": _gemini, "anthropic": _anthropic, "openai": _openai, "mock": _mock}


def goi_llm(he_thong: str, nguoi_dung: str, max_tokens: int = 1200) -> str:
    return _BANG.get(NHA_CUNG_CAP, _mock)(he_thong, nguoi_dung, max_tokens)


def dang_dung() -> str:
    return NHA_CUNG_CAP


def kiem_tra_ket_noi() -> tuple[bool, str]:
    """Thử một lượt gọi ngắn để xác nhận khóa API hoạt động."""
    try:
        tra = goi_llm("Bạn là trợ lý kiểm tra kết nối.",
                      "Trả lời đúng hai từ: Kết nối thành công.", max_tokens=30)
        return True, f"[{NHA_CUNG_CAP}] {tra[:80]}"
    except Exception as e:
        return False, f"[{NHA_CUNG_CAP}] {e}"


def liet_ke_model() -> list[str]:
    """Hỏi thẳng Google xem khóa hiện tại dùng được những model nào.

    Google liên tục cho ngừng model cũ và ra model mới, nên đừng chép cứng tên
    model từ bài hướng dẫn trên mạng. Hãy chạy lệnh này rồi chọn từ danh sách thật.
    """
    if NHA_CUNG_CAP != "gemini":
        raise LoiLLM("Lệnh này chỉ dùng cho Gemini. Đặt MICRA_LLM=gemini trong .env")
    try:
        from google import genai
    except ImportError as e:
        raise LoiLLM("Chưa cài thư viện. Chạy: pip install google-genai") from e
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise LoiLLM("Thiếu GEMINI_API_KEY trong tệp .env")
    ra = []
    for m in genai.Client(api_key=key).models.list():
        hanh_dong = list(getattr(m, "supported_actions", None) or [])
        if not hanh_dong or "generateContent" in hanh_dong:
            ra.append(str(m.name).replace("models/", ""))
    return sorted(ra)


if __name__ == "__main__":
    if "--models" in sys.argv:
        try:
            ds = liet_ke_model()
            print(f"Khóa của bạn dùng được {len(ds)} model:\n")
            for t in ds:
                print("   " + t)
            uu_tien = [t for t in ds if "flash" in t and "lite" in t] or \
                      [t for t in ds if "flash" in t] or ds
            if uu_tien:
                print(f"\nNên chọn: {uu_tien[0]}")
                print(f"Sửa trong .env:  MICRA_MODEL_GEMINI={uu_tien[0]}")
        except Exception as e:
            print("Lỗi:", e)
    else:
        ok, tb = kiem_tra_ket_noi()
        print(("✓ " if ok else "✗ ") + tb)
