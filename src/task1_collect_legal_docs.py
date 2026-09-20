"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Hướng dẫn:
    1. Chọn chủ đề của nhóm.
    2. Tìm tối thiểu 3 tài liệu PDF/DOCX từ nguồn công khai.
    3. Lưu file gốc vào data/landing/legal/.
    4. Đặt tên không dấu và thể hiện đúng nội dung.

Nguồn: Thư viện Pháp luật (nội dung văn bản công khai).
Website dùng Cloudflare nên requests thường bị 403; nội dung đã được
lấy từ trang công khai rồi lưu PDF có thể tìm kiếm.
"""

from pathlib import Path

from src.corpus_utils import write_text_pdf


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
RAW_DIR = Path(__file__).parent.parent / "data" / "_raw" / "legal"

LEGAL_SOURCES = [
    {
        "filename": "nghi-dinh-349-2026-nd-cp-lua-chon-nha-thau.pdf",
        "raw_name": "nghi-dinh-349-2026.json",
        "url": "https://thuvienphapluat.vn/van-ban/Dau-tu/Nghi-dinh-349-2026-ND-CP-sua-doi-cac-Nghi-dinh-huong-dan-Luat-Dau-thau-lua-chon-nha-thau-710002.aspx",
        "title": "Nghị định 349/2026/NĐ-CP sửa đổi các Nghị định hướng dẫn Luật Đấu thầu về lựa chọn nhà thầu",
    },
    {
        "filename": "luat-cong-nghe-cao-2025-so-133-2025-qh15.pdf",
        "raw_name": "luat-cong-nghe-cao-2025.json",
        "url": "https://thuvienphapluat.vn/van-ban/Linh-vuc-khac/Luat-Cong-nghe-cao-2025-so-133-2025-QH15-675211.aspx",
        "title": "Luật Công nghệ cao 2025 số 133/2025/QH15",
    },
    {
        "filename": "thong-tu-34-2026-tt-byt-cham-soc-suc-khoe-dan-so.pdf",
        "raw_name": "thong-tu-34-2026-tt-byt.json",
        "url": "https://thuvienphapluat.vn/van-ban/The-thao-Y-te/Thong-tu-34-2026-TT-BYT-huong-dan-Chuong-trinh-muc-tieu-quoc-gia-cham-soc-suc-khoe-dan-so-724066.aspx",
        "title": "Thông tư 34/2026/TT-BYT hướng dẫn Chương trình mục tiêu quốc gia chăm sóc sức khỏe dân số",
    },
]


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_documents() -> None:
    """Tạo PDF từ nội dung văn bản công khai đã thu thập."""
    import json

    setup_directory()
    missing = []
    for source in LEGAL_SOURCES:
        output = DATA_DIR / source["filename"]
        raw_path = RAW_DIR / source["raw_name"]
        if not raw_path.exists():
            missing.append(source["url"])
            continue
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        write_text_pdf(
            output,
            title=payload.get("title") or source["title"],
            text=payload["text"],
            source_url=payload.get("url") or source["url"],
        )
        print(f"Saved: {output}")

    if missing:
        raise FileNotFoundError(
            "Thiếu file raw cho: " + "; ".join(missing)
        )


if __name__ == "__main__":
    setup_directory()
    download_documents()
