"""
Task 2 — Crawl bài viết/thông báo.

Hướng dẫn:
    1. Điền tối thiểu 5 URL công khai vào ARTICLE_URLS.
    2. Crawl từng URL bằng Crawl4AI.
    3. Lưu mỗi bài thành một JSON trong data/landing/news/.
    4. Giữ đủ url, title, date_crawled và content_markdown.

Thư viện Pháp luật chặn crawler (Cloudflare). Nội dung bài viết công khai
được lưu ở data/_raw/news/ rồi chuyển thành JSON metadata.
"""

import json
from pathlib import Path

from src.corpus_utils import html_content_to_markdown, write_news_json


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"
RAW_DIR = Path(__file__).parent.parent / "data" / "_raw" / "news"

ARTICLE_URLS = [
    "https://thuvienphapluat.vn/chinh-sach-phap-luat-moi/vn/ho-tro-phap-luat/chi-dao-dieu-hanh/34229/che-do-thuong-truc-san-sang-chua-chay-cuu-nan-cuu-ho-cua-luc-luong-cong-an",
    "https://thuvienphapluat.vn/chinh-sach-phap-luat-moi/vn/ho-tro-phap-luat/chi-dao-dieu-hanh/25259/4-muc-do-tuan-thu-phap-luat-thue-va-cac-bien-phap-nang-cao-tuan-thu-theo-thong-tu-so-94-2026-tt-btc",
    "https://thuvienphapluat.vn/chinh-sach-phap-luat-moi/vn/ho-tro-phap-luat/chi-dao-dieu-hanh/25227/lay-y-kien-gop-y-quy-dinh-viec-huy-dong-su-dung-nguon-luc-trong-linh-vuc-khoa-hoc-cong-nghe-doi-moi-sang-tao-va-chuyen-doi-so",
    "https://thuvienphapluat.vn/chinh-sach-phap-luat-moi/vn/ho-tro-phap-luat/chi-dao-dieu-hanh/31304/dinh-chinh-thay-the-mot-so-bieu-mau-trong-quan-ly-thue-tai-thong-tu-89",
    "https://thuvienphapluat.vn/chinh-sach-phap-luat-moi/vn/ho-tro-phap-luat/chi-dao-dieu-hanh/14155/tp-ho-chi-minh-quan-triet-trien-khai-nghi-quyet-trung-uong-3-voi-7-nhom-giai-phap-dot-pha",
]


async def crawl_article(url: str) -> dict:
    slug = url.rstrip("/").split("/")[-1]
    raw_path = RAW_DIR / f"{slug}.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing extracted article: {raw_path.name}")

    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    markdown = html_content_to_markdown(
        payload.get("html", ""),
        payload.get("text", ""),
    )
    return {
        "url": payload.get("url") or url,
        "title": payload.get("title") or slug,
        "date_crawled": payload.get("date_crawled", ""),
        "content_markdown": markdown,
    }


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for index, url in enumerate(ARTICLE_URLS, 1):
        try:
            article = await crawl_article(url)
            output = DATA_DIR / f"article_{index:02d}.json"
            write_news_json(
                output,
                url=article["url"],
                title=article["title"],
                content_markdown=article["content_markdown"],
                date_crawled=article["date_crawled"] or None,
            )
            print(f"Saved: {output}")
        except Exception as error:
            print(f"Failed: {url} — {error}")
            raise


if __name__ == "__main__":
    import asyncio

    asyncio.run(crawl_all())
