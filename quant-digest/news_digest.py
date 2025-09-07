import os
import re
import sys
import time
import socket
import requests
import feedparser
from datetime import datetime
from urllib.parse import urlparse
from textwrap import shorten

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from openai import OpenAI
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Env / Config
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
EMAIL = os.getenv("EMAIL", "").strip()
PASSWORD = os.getenv("PASSWORD", "").strip()

SUMMARY_MODEL = "gpt-4.1-mini"
SUMMARY_MAX_TOKENS = 400
OVERVIEW_MAX_TOKENS = 600
OPENAI_TEMP = 0.2

REQUEST_TIMEOUT = 15
MAX_PER_RSS_FEED = 5
MAX_NEWSAPI_MAIN = 10
MAX_NEWSAPI_AI = 5

# Topics
TOPICS = ["quant finance", "derivatives", "hedge funds", "trading", "markets"]
EXTRA_TOPICS = ["AI in finance", "machine learning trading", "algorithmic trading AI"]

# RSS feeds
RSS_FEEDS = [
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # CNBC Markets
    "https://www.reutersagency.com/feed/?best-topics=markets", # Reuters Markets
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC,^IXIC,^DJI&region=US&lang=en-US", # Yahoo Finance
    "https://ftalphaville.ft.com/feed/"  # FT Alphaville
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _safe_get(url: str, params=None):
    try:
        return requests.get(url, params=params or {}, timeout=REQUEST_TIMEOUT)
    except (requests.RequestException, socket.timeout) as e:
        print(f"[WARN] Request failed: {url} — {e}")
        return None

def _host_from_url(u: str) -> str:
    try:
        return urlparse(u).netloc or ""
    except Exception:
        return ""

def _norm(s: str) -> str:
    return " ".join((s or "").strip().split())

def _truncate(s: str, lim: int = 260) -> str:
    s = _norm(s)
    if len(s) <= lim:
        return s
    return shorten(s, width=lim, placeholder="…")

def _strip_hot_marker(summary: str):
    """
    Return (clean_summary, is_hot).
    Accepts variants like:
      HOT LIST=Yes / HOT LIST=No
      Hot List: Yes / No
      (case-insensitive, with or without trailing period)
    """
    text = summary or ""
    # Determine hot before stripping
    is_hot = "HOT LIST=YES" in text.upper() or re.search(
        r"^\s*hot\s*list\s*[:=]\s*yes\s*\.?\s*$",
        text,
        re.IGNORECASE | re.MULTILINE,
    ) is not None

    # Remove any HOT LIST line entirely
    pattern = re.compile(r"^\s*hot\s*list\s*[:=]\s*(yes|no)\s*\.?\s*$",
                         re.IGNORECASE | re.MULTILINE)
    text = re.sub(pattern, "", text).strip()
    return text, is_hot

def _dedupe_articles(articles):
    """
    Deduplicate by URL if present, else by normalized title.
    Keep first occurrence.
    """
    seen_urls = set()
    seen_titles = set()
    out = []
    for a in articles:
        url = (a.get("url") or "").strip()
        title = _norm(a.get("title") or "")
        key_url = url.lower()
        key_title = title.lower()

        if url and key_url in seen_urls:
            continue
        if (not url) and title and key_title in seen_titles:
            continue

        if url:
            seen_urls.add(key_url)
        if title:
            seen_titles.add(key_title)
        out.append(a)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Fetch news
# ─────────────────────────────────────────────────────────────────────────────
def fetch_news():
    """Fetch from NewsAPI + RSS feeds"""
    articles = []

    # NewsAPI - main finance topics
    if NEWS_API_KEY:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": " OR ".join(TOPICS),
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": MAX_NEWSAPI_MAIN,
            "apiKey": NEWS_API_KEY,
            # optional: "searchIn": "title,description"
        }
        resp = _safe_get(url, params=params)
        if resp and resp.ok:
            data = resp.json()
            for art in data.get("articles", []):
                articles.append({
                    "title": art.get("title"),
                    "desc": art.get("description") or "",
                    "url": art.get("url"),
                    "source": (art.get("source") or {}).get("name") or _host_from_url(art.get("url") or "") or "Unknown",
                    "category": "finance"
                })
        else:
            print("[WARN] NewsAPI main topics request failed or no response.")

        # NewsAPI - AI in finance
        params_ai = {
            "q": " OR ".join(EXTRA_TOPICS),
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": MAX_NEWSAPI_AI,
            "apiKey": NEWS_API_KEY,
        }
        resp_ai = _safe_get(url, params=params_ai)
        if resp_ai and resp_ai.ok:
            data_ai = resp_ai.json()
            for art in data_ai.get("articles", []):
                articles.append({
                    "title": art.get("title"),
                    "desc": art.get("description") or "",
                    "url": art.get("url"),
                    "source": (art.get("source") or {}).get("name") or _host_from_url(art.get("url") or "") or "Unknown",
                    "category": "ai"
                })
        else:
            print("[WARN] NewsAPI AI topics request failed or no response.")
    else:
        print("[INFO] NEWS_API_KEY not set — skipping NewsAPI and using RSS only.")

    # RSS feeds
    for feed in RSS_FEEDS:
        try:
            parsed = feedparser.parse(feed)
            feed_title = (parsed.feed.get("title") if hasattr(parsed, "feed") else None) or _host_from_url(feed) or "RSS Feed"
            for entry in parsed.entries[:MAX_PER_RSS_FEED]:
                articles.append({
                    "title": getattr(entry, "title", "") or "(untitled)",
                    "desc": getattr(entry, "summary", "") or "",
                    "url": getattr(entry, "link", "") or "",
                    "source": feed_title,
                    "category": "finance"
                })
        except Exception as e:
            print(f"[WARN] RSS parse failed for {feed}: {e}")

    articles = _dedupe_articles(articles)
    return articles

# ─────────────────────────────────────────────────────────────────────────────
# Summarization
# ─────────────────────────────────────────────────────────────────────────────
def summarize_articles(articles):
    """Use GPT to summarize + flag hot items"""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    summaries, hot_list, ai_summaries = [], [], []

    system_prompt = (
        "Summarize the given financial news in ~100 words (4–6 sentences). "
        "Focus on market/quant relevance: data points, price action, policy, flows, positioning. "
        "Avoid fluff. End your response with a single new line containing exactly either "
        "'HOT LIST=Yes' or 'HOT LIST=No'. No extra commentary."
    )

    for idx, art in enumerate(articles, start=1):
        content = (
            f"Title: {art.get('title')}\n"
            f"Source: {art.get('source')}\n"
            f"Description: {art.get('desc')}\n"
            f"URL: {art.get('url')}"
        )

        try:
            chat = client.chat.completions.create(
                model=SUMMARY_MODEL,
                temperature=OPENAI_TEMP,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                max_tokens=SUMMARY_MAX_TOKENS,
            )
            raw = (chat.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[WARN] Summarization failed for article {idx}: {e}")
            continue

        clean, is_hot = _strip_hot_marker(raw)
        item = (art.get("title") or "(untitled)", clean, art.get("url") or "", art.get("source") or "Unknown")

        if (art.get("category") or "").lower() == "ai":
            ai_summaries.append(item)
        else:
            summaries.append(item)

        if is_hot:
            hot_list.append(item)

        print(f"[OK] Summarized {idx}/{len(articles)} | HOT={is_hot} | {item[0][:70]}")

        # brief pause to be polite to API (optional)
        time.sleep(0.1)

    return summaries, hot_list, ai_summaries

# ─────────────────────────────────────────────────────────────────────────────
# Daily Overview (briefing)
# ─────────────────────────────────────────────────────────────────────────────
def build_overview_context(summaries, hot_list):
    """
    Build structured context for the briefing.
    summaries/hot_list: list[tuple(title, summary, url, source)]
    """
    lines = []
    today = datetime.now().strftime("%Y-%m-%d")
    lines.append(f"DATE: {today}")

    if hot_list:
        lines.append("HOT ITEMS (prioritise these signals first):")
        for title, summary, url, source in hot_list[:5]:
            lines.append(f"• {title} — {source}: {_truncate(summary, 240)}")

    lines.append("KEY ARTICLES:")
    for title, summary, url, source in summaries[:12]:
        lines.append(f"• {title} — {source}: {_truncate(summary, 240)}")

    return "\n".join(lines)

def overall_summary(summaries, hot_list=None):
    """Generate a 200–250 word quant-focused daily briefing."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    hot_list = hot_list or []
    context = build_overview_context(summaries, hot_list)

    sys_prompt = (
        "You are a buy-side macro/quant analyst. Write a tight 200–250 word daily market briefing "
        "for a quant audience. DO NOT ask clarifying questions. If information is thin, infer the "
        "most likely market tone without fabricating exact numbers. Prioritise items listed under "
        "'HOT ITEMS'. Keep it objective and structured as:\n\n"
        "1) Macro & Sentiment\n"
        "2) Equities / Rates\n"
        "3) FX & Commodities (only if context exists)\n"
        "4) What to Watch (1–3 bullets)\n\n"
        "No hyperlinks. No preambles. No ‘as an AI’ phrasing. 200–250 words."
    )

    chat = client.chat.completions.create(
        model=SUMMARY_MODEL,
        temperature=OPENAI_TEMP,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": context},
        ],
        max_tokens=OVERVIEW_MAX_TOKENS,
    )

    return (chat.choices[0].message.content or "").strip()

# ─────────────────────────────────────────────────────────────────────────────
# PDF
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf(summaries, hot_list, overview, ai_summaries):
    """Make a nice PDF"""
    output_dir = os.path.join("quant-digest", "GeneratedDigests")
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(
        output_dir, f"Quant Digest_{datetime.now().strftime('%Y-%m-%d')}.pdf"
    )

    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    hot_style = ParagraphStyle("HotStyle", parent=styles["Heading1"], textColor=colors.red)
    source_style = ParagraphStyle("SourceStyle", parent=styles["Normal"], textColor=colors.grey, fontSize=9)

    # Title
    story.append(Paragraph("Quant Daily Digest", styles["Title"]))
    story.append(Spacer(1, 12))

    # Overview first
    story.append(Paragraph("📊 Daily Overview", styles["Heading1"]))
    story.append(Paragraph(overview or "No overview available.", styles["BodyText"]))
    story.append(Spacer(1, 12))

    # Hot List
    if hot_list:
        story.append(Paragraph("🔥 Hot List", hot_style))
        story.append(Spacer(1, 6))
        for title, summary, url, source in hot_list:
            story.append(Paragraph(f"<b>{_norm(title)}</b>", styles["Heading2"]))
            story.append(Paragraph(_norm(summary), styles["BodyText"]))
            if url:
                story.append(Paragraph(f"<font color='blue'>Read more: <u>{url}</u></font>", styles["Normal"]))
            story.append(Paragraph(f"{_norm(source)}", source_style))
            story.append(Spacer(1, 12))

    # Articles (skip those already in hot list)
    hot_titles = {h[0] for h in hot_list}
    story.append(Paragraph("📰 Articles", styles["Heading1"]))
    any_article = False
    for title, summary, url, source in summaries:
        if title in hot_titles:
            continue
        any_article = True
        story.append(Paragraph(f"<b>{_norm(title)}</b>", styles["Heading2"]))
        story.append(Paragraph(_norm(summary), styles["BodyText"]))
        if url:
            story.append(Paragraph(f"<font color='blue'>Read more: <u>{url}</u></font>", styles["Normal"]))
        story.append(Paragraph(f"{_norm(source)}", source_style))
        story.append(Spacer(1, 12))
    if not any_article:
        story.append(Paragraph("No additional articles (all key items are in Hot List).", styles["BodyText"]))
        story.append(Spacer(1, 12))

    # AI & Finance section
    if ai_summaries:
        story.append(Paragraph("🤖 AI & Finance", styles["Heading1"]))
        for title, summary, url, source in ai_summaries:
            story.append(Paragraph(f"<b>{_norm(title)}</b>", styles["Heading2"]))
            story.append(Paragraph(_norm(summary), styles["BodyText"]))
            if url:
                story.append(Paragraph(f"<font color='blue'>Read more: <u>{url}</u></font>", styles["Normal"]))
            story.append(Paragraph(f"{_norm(source)}", source_style))
            story.append(Spacer(1, 12))

    doc.build(story)
    return filename

# ─────────────────────────────────────────────────────────────────────────────
# Email
# ─────────────────────────────────────────────────────────────────────────────
def send_email(file_path):
    if not EMAIL or not PASSWORD:
        print("[WARN] EMAIL or PASSWORD missing — skipping email send.")
        return

    sender = EMAIL
    recipient = EMAIL
    subject = "Your Quant Daily Digest"
    body = "Attached is today's quant digest PDF."

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    with open(file_path, "rb") as f:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(file_path)}')
        msg.attach(part)

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, PASSWORD)
        server.sendmail(sender, recipient, msg.as_string())
        server.quit()
        print(f"[OK] Email sent to {recipient}")
    except Exception as e:
        print(f"[WARN] Failed to send email: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[*] Fetching news…")
    articles = fetch_news()
    print(f"[*] Collected {len(articles)} raw articles after dedupe.")

    if not articles:
        print("[ERR] No articles fetched. Check your API keys / network / feeds.")
        sys.exit(1)

    print("[*] Summarizing articles…")
    summaries, hot_list, ai_summaries = summarize_articles(articles)
    print(f"[*] Summaries: {len(summaries)} | Hot: {len(hot_list)} | AI: {len(ai_summaries)}")

    if not summaries and not ai_summaries:
        print("[ERR] No summaries produced. Aborting before PDF/email.")
        sys.exit(1)

    print("[*] Building daily overview…")
    overview = overall_summary(summaries, hot_list)

    print("[*] Generating PDF…")
    pdf = generate_pdf(summaries, hot_list, overview, ai_summaries)
    print(f"[OK] Generated {pdf}")

    print("[*] Sending email (if configured)…")
    send_email(pdf)
    print("[DONE] Quant Daily Digest complete.")
