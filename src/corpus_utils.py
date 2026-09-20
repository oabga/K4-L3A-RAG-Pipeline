"""Helpers for collecting and converting the public legal/news corpus."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from fpdf import FPDF
from markdownify import markdownify as html_to_markdown


WINDOWS_FONT = Path(r"C:\Windows\Fonts\times.ttf")
RELATED_MARKERS = (
    "\nBài liên quan:",
    "\nBài viết liên quan",
    "\nHỏi đáp pháp luật",
    "\nPHÁP LUẬT DOANH NGHIỆP",
    "\nPháp Luật Thuế",
)
NEWS_CUT_MARKERS = (
    "\nTra cứu Mã số thuế",
    "\nBài viết về",
)


def clean_extracted_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\t", " ")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    for marker in RELATED_MARKERS:
        index = text.find(marker)
        if index > 3000:
            text = text[:index].strip()
            break
    return text


def _safe_pdf_line(text: str, max_token: int = 70) -> str:
    cleaned = "".join(ch if ch.isprintable() else " " for ch in text)
    parts: list[str] = []
    for token in cleaned.split(" "):
        if len(token) > max_token:
            parts.extend(token[i : i + max_token] for i in range(0, len(token), max_token))
        else:
            parts.append(token)
    return " ".join(parts).strip()


def write_text_pdf(path: Path, title: str, text: str, source_url: str) -> None:
    """Write a searchable Unicode PDF so MarkItDown can convert it later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = clean_extracted_text(text)
    if not WINDOWS_FONT.exists():
        raise FileNotFoundError(f"Missing Unicode font: {WINDOWS_FONT}")

    pdf = FPDF(format="A4")
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("TimesVN", fname=str(WINDOWS_FONT))
    pdf.add_page()
    pdf.set_font("TimesVN", size=14)
    pdf.multi_cell(0, 8, _safe_pdf_line(title), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("TimesVN", size=10)
    pdf.multi_cell(0, 6, _safe_pdf_line(f"Nguồn: {source_url}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("TimesVN", size=12)
    for line in body.splitlines() or [body]:
        rendered = _safe_pdf_line(line) or " "
        pdf.multi_cell(0, 6, rendered, new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))


def clean_news_text(text: str) -> str:
    cleaned = clean_extracted_text(text)
    for marker in NEWS_CUT_MARKERS:
        index = cleaned.find(marker)
        if index > 400:
            cleaned = cleaned[:index].strip()
            break
    return cleaned


def text_to_markdown(text: str) -> str:
    cleaned = clean_news_text(text)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
    return "\n\n".join(paragraphs)


def html_content_to_markdown(html: str, fallback_text: str) -> str:
    """Prefer cleaned article text; HTML on this site includes ads/CSS/sidebar."""
    markdown = text_to_markdown(fallback_text)
    if len(markdown) >= 200:
        return markdown
    if html.strip():
        converted = html_to_markdown(
            html, heading_style="ATX", strip=["script", "style", "img"]
        )
        converted = re.sub(r"\n{3,}", "\n\n", converted).strip()
        if converted:
            return converted
    return markdown


def write_news_json(
    path: Path,
    *,
    url: str,
    title: str,
    content_markdown: str,
    date_crawled: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "url": url,
        "title": title.strip(),
        "date_crawled": date_crawled or datetime.now().isoformat(timespec="seconds"),
        "content_markdown": content_markdown.strip(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
