"""
Task 2 — Crawl bài viết/thông báo.

Hướng dẫn:
    1. Điền tối thiểu 5 URL công khai vào ARTICLE_URLS.
    2. Crawl từng URL bằng Crawl4AI.
    3. Lưu mỗi bài thành một JSON trong data/landing/news/.
    4. Giữ đủ url, title, date_crawled và content_markdown.

Cài browser trước khi chạy:
    python -m playwright install chromium

-> Dùng Firecrawl or bất cứ công cụ nào bạn quen
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.browser_manager import BrowserManager


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# Thứ tự khớp article_01.json … article_08.json trong data/landing/news/.
# 01–05: hoang-minh (Thư viện Pháp luật). 06–08: LeGiaBao (thuế hộ kinh doanh).
ARTICLE_URLS = [
    "https://thuvienphapluat.vn/chinh-sach-phap-luat-moi/vn/ho-tro-phap-luat/chi-dao-dieu-hanh/34229/che-do-thuong-truc-san-sang-chua-chay-cuu-nan-cuu-ho-cua-luc-luong-cong-an",
    "https://thuvienphapluat.vn/chinh-sach-phap-luat-moi/vn/ho-tro-phap-luat/chi-dao-dieu-hanh/25259/4-muc-do-tuan-thu-phap-luat-thue-va-cac-bien-phap-nang-cao-tuan-thu-theo-thong-tu-so-94-2026-tt-btc",
    "https://thuvienphapluat.vn/chinh-sach-phap-luat-moi/vn/ho-tro-phap-luat/chi-dao-dieu-hanh/25227/lay-y-kien-gop-y-quy-dinh-viec-huy-dong-su-dung-nguon-luc-trong-linh-vuc-khoa-hoc-cong-nghe-doi-moi-sang-tao-va-chuyen-doi-so",
    "https://thuvienphapluat.vn/chinh-sach-phap-luat-moi/vn/ho-tro-phap-luat/chi-dao-dieu-hanh/31304/dinh-chinh-thay-the-mot-so-bieu-mau-trong-quan-ly-thue-tai-thong-tu-89",
    "https://thuvienphapluat.vn/chinh-sach-phap-luat-moi/vn/ho-tro-phap-luat/chi-dao-dieu-hanh/14155/tp-ho-chi-minh-quan-triet-trien-khai-nghi-quyet-trung-uong-3-voi-7-nhom-giai-phap-dot-pha",
    "https://tapchikinhtetaichinh.vn/giai-phap-chinh-sach-va-quan-ly-thue-ho-tro-ho-va-ca-nhan-kinh-doanh-doanh-nghiep-nho-va-sieu-nho-trong-giai-doan-moi-161740.html",
    "https://techcombank.com/thong-tin/blog/thue-thuong-mai-dien-tu",
    "https://baomoi.com/tang-quan-ly-thue-doi-voi-ca-nhan-ho-kinh-doanh-c56082733.epi",
]

# Báo Mới hay chặn crawler; fallback về bài gốc Báo Đồng Nai.
CANONICAL_FALLBACKS = {
    "https://baomoi.com/tang-quan-ly-thue-doi-voi-ca-nhan-ho-kinh-doanh-c56082733.epi": (
        "https://baodongnai.com.vn/kinh-te/202609/tang-quan-ly-thue-doi-voi-ca-nhan-ho-kinh-doanh-65b2db2/"
    ),
}

MIN_CONTENT_CHARS = 200

CSS_SELECTORS = {
    "tapchikinhtetaichinh.vn": "article, .detail-content, .b-article, .content-detail",
    "techcombank.com": "article, main, .blog-detail, .cmp-text",
    "baomoi.com": "article, .content, .detail, .bm-card-content",
    "baodongnai.com.vn": "article, .detail-content, .article-content, .content-detail",
    "thuvienphapluat.vn": "article, .content, .news-content, .detail-content",
}


def _host(url: str) -> str:
    return urlparse(url).netloc.replace("www.", "")


def _chrome_executable() -> str | None:
    """Dùng Chrome đầy đủ nếu Playwright chưa có chrome-headless-shell."""
    candidates = [
        os.environ.get("CRAWL4AI_CHROME_PATH"),
        str(Path.home() / ".cache/ms-playwright/chromium-1243/chrome-linux64/chrome"),
    ]
    for path in candidates:
        if path and Path(path).is_file():
            return path
    return None


_ORIGINAL_BUILD_BROWSER_ARGS = BrowserManager._build_browser_args


def _build_browser_args_with_chrome(self) -> dict:
    args = _ORIGINAL_BUILD_BROWSER_ARGS(self)
    chrome = _chrome_executable()
    if chrome:
        args["executable_path"] = chrome
    return args


BrowserManager._build_browser_args = _build_browser_args_with_chrome


def _browser_config() -> BrowserConfig:
    return BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--no-sandbox", "--disable-dev-shm-usage"],
        user_agent=(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
    )


def _markdown_text(result) -> str:
    markdown = result.markdown
    if markdown is None:
        return ""
    if isinstance(markdown, str):
        return markdown.strip()
    for attr in ("fit_markdown", "raw_markdown", "markdown_with_citations"):
        value = getattr(markdown, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(markdown).strip()


def _clean_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _article_title(result, url: str, content: str = "") -> str:
    metadata = result.metadata or {}
    title = (
        metadata.get("title")
        or metadata.get("og:title")
        or metadata.get("ogTitle")
        or ""
    ).strip()
    slug = urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]
    if title and title.lower() not in {slug.lower(), "unknown"}:
        return re.sub(r"\s+", " ", title.split("|")[0]).strip()
    heading = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
    if heading:
        return heading.group(1).strip()
    return slug or "Unknown"


def _run_config(url: str) -> CrawlerRunConfig:
    selector = CSS_SELECTORS.get(_host(url))
    kwargs = {
        "cache_mode": CacheMode.BYPASS,
        "word_count_threshold": 10,
        "excluded_tags": ["nav", "footer", "header", "form", "aside", "script", "style"],
        "exclude_external_links": True,
        "wait_until": "domcontentloaded",
        "delay_before_return_html": 1.5,
        "page_timeout": 60000,
    }
    if selector:
        kwargs["css_selector"] = selector
    return CrawlerRunConfig(**kwargs)


async def _arun(crawler: AsyncWebCrawler, url: str):
    result = await crawler.arun(url=url, config=_run_config(url))
    if not result.success:
        raise RuntimeError(result.error_message or f"Crawl failed for {url}")
    content = _clean_markdown(_markdown_text(result))
    if len(content) < MIN_CONTENT_CHARS:
        # Thử lại không CSS selector nếu trang không khớp cấu trúc.
        fallback_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            word_count_threshold=10,
            excluded_tags=["nav", "footer", "header", "form", "aside", "script", "style"],
            wait_until="domcontentloaded",
            delay_before_return_html=2.0,
            page_timeout=60000,
        )
        result = await crawler.arun(url=url, config=fallback_config)
        if not result.success:
            raise RuntimeError(result.error_message or f"Crawl failed for {url}")
        content = _clean_markdown(_markdown_text(result))
    if len(content) < MIN_CONTENT_CHARS:
        raise RuntimeError(f"Content too short ({len(content)} chars) for {url}")
    return result, content


async def crawl_article(url: str, crawler: AsyncWebCrawler | None = None) -> dict:
    """Crawl một URL công khai thành JSON theo contract Task 2."""
    created_crawler = False
    if crawler is None:
        crawler = AsyncWebCrawler(config=_browser_config())
        await crawler.__aenter__()
        created_crawler = True

    try:
        try:
            result, content = await _arun(crawler, url)
            crawled_url = url
        except Exception as error:
            fallback = CANONICAL_FALLBACKS.get(url)
            if not fallback:
                raise
            print(f"Retry via original source after {url} failed: {error}")
            result, content = await _arun(crawler, fallback)
            crawled_url = fallback

        return {
            "url": crawled_url,
            "title": _article_title(result, crawled_url, content),
            "date_crawled": datetime.now(timezone.utc).isoformat(),
            "content_markdown": content,
        }
    finally:
        if created_crawler:
            await crawler.__aexit__(None, None, None)


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    async with AsyncWebCrawler(config=_browser_config()) as crawler:
        for index, url in enumerate(ARTICLE_URLS, 1):
            try:
                article = await crawl_article(url, crawler=crawler)
                output = DATA_DIR / f"article_{index:02d}.json"
                output.write_text(
                    json.dumps(article, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(
                    f"Saved: {output} | {article['title'][:80]} | "
                    f"{len(article['content_markdown'])} chars"
                )
            except Exception as error:
                print(f"Failed: {url} — {error}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
