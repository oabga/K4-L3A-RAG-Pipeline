"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Hướng dẫn:
    1. Chọn chủ đề của nhóm.
    2. Tìm tối thiểu 3 tài liệu PDF/DOCX từ nguồn công khai.
    3. Lưu file gốc vào data/landing/legal/.
    4. Đặt tên không dấu và thể hiện đúng nội dung.

Ví dụ tài liệu: học phí, học bổng, ký túc xá, quy trình đăng ký.
Nếu website chặn crawler, hãy chọn nguồn công khai khác; không vượt WAF.
"""

from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx"}
MIN_DOCUMENTS = 3
MIN_FILE_SIZE = 1024


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_documents() -> None:
    """Kiểm tra các tài liệu đã tải thủ công vào data/landing/legal/."""
    files = sorted(
        path for path in DATA_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in DOCUMENT_EXTENSIONS
    )

    print(f"Found {len(files)} legal document(s) in {DATA_DIR}:")
    for path in files:
        print(f"  - {path.name} ({path.stat().st_size / 1024:.0f} KB)")
        if not path.name.isascii():
            print("    WARNING: tên file có dấu, hãy đổi thành tên không dấu")
        if path.stat().st_size <= MIN_FILE_SIZE:
            print("    WARNING: file quá nhỏ, có thể bị lỗi khi tải")

    if len(files) < MIN_DOCUMENTS:
        raise RuntimeError(
            f"Cần ít nhất {MIN_DOCUMENTS} tài liệu PDF/DOCX, hiện có {len(files)}"
        )


if __name__ == "__main__":
    setup_directory()
    download_documents()
