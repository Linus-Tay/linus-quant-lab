import os
import re
import sys
import time
import json
import socket
import requests
import feedparser
from datetime import datetime, timedelta
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

try:
    from ics import Calendar
    from dateutil import tz
    _HAS_CAL = True
except Exception:
    _HAS_CAL = False

# ─────────────────────────────────────────────────────────────────────────────
# Env / Config
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
EMAIL = os.getenv("EMAIL", "").strip()
EMAIL2 = os.getenv("EMAIL2", "").strip()
PASSWORD = os.getenv("PASSWORD", "").strip()
RESEARCH_TOP_N = 10
print("EMAIL2 =", repr(EMAIL2))
# Optional calendar feeds (comma-separated .ics URLs). If empty, calendar lane skips.
CALENDAR_ICS_URLS = "https://www.bls.gov/schedule/news_release/bls.ics,https://tradingeconomics.com/calendar/ics/united-states"
SUMMARY_MODEL = "gpt-4.1-mini"     # cheap & capable; you can swap to "gpt-4o-mini"
SUMMARY_MAX_TOKENS = 400
OVERVIEW_MAX_TOKENS = 600
GLOSSARY_MAX_TOKENS = 900
OPENAI_TEMP = 0.2

REQUEST_TIMEOUT = 15
MAX_PER_RSS_FEED = 5
MAX_NEWSAPI_MAIN = 10
MAX_NEWSAPI_AI = 5

# Topics
TOPICS = ["quant finance", "derivatives", "hedge funds", "trading", "markets"]
EXTRA_TOPICS = ["AI in finance", "machine learning trading", "algorithmic trading AI"]
RESEARCH_EXTRA_FEEDS = ["https://www.aqr.com/Insights/rss","https://www.man.com/maninstitute/rss.xml"]

# RSS feeds
RSS_FEEDS = [
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # CNBC Markets
    "https://www.reutersagency.com/feed/?best-topics=markets", # Reuters Markets
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC,^IXIC,^DJI&region=US&lang=en-US",
    "https://ftalphaville.ft.com/feed/"
]

# Research feeds (reliable, low-flake)
# Research feeds (mix of academic + practitioner). All are RSS/Atom.
RESEARCH_FEEDS = [
    # arXiv (quant finance)
    "https://export.arxiv.org/rss/q-fin",
    "https://export.arxiv.org/rss/q-fin.TR",
    "https://export.arxiv.org/rss/q-fin.ST",

    # BIS research & Central Bank Research Hub
    "https://www.bis.org/doclist/bis_fsi_publs.rss",
    "https://www.bis.org/doclist/reshub_papers.rss",

    # CEPR Discussion Papers
    "https://cepr.org/rss/discussion-paper",

    # Fed Board working papers
    "https://www.federalreserve.gov/feeds/working_papers.xml",
    "https://www.federalreserve.gov/feeds/feds.xml",
    "https://www.federalreserve.gov/feeds/ifdp.xml",

    # NBER (if it ever looks empty that day, that’s normal)
    "https://www.nber.org/rss/new.xml",
]

# Practitioner/quant houses with working RSS
RESEARCH_EXTRA_FEEDS = [
    "https://www.man.com/maninstitute/rss.xml",
    "https://alphaarchitect.com/feed/",
    "https://quantocracy.com/feed/",
]


# Calendar settings
CALENDAR_LOOKAHEAD_DAYS = 7
CALENDAR_KEYWORDS = [
    "CPI","inflation","nonfarm","payroll","NFP","FOMC","Fed","ECB","BoE","BoJ",
    "rate decision","PCE","ISM","PMI","GDP","retail sales","jobless","claims","auction"
]

# Research cadence toggles
RUN_RESEARCH = os.getenv("RUN_RESEARCH", "weekly").lower()  # "weekly", "always", "off"
RESEARCH_DAY_UTC = 6  # 0=Mon ... 6=Sun
FORCE_RESEARCH = os.getenv("FORCE_RESEARCH", "").lower() in ("1", "true", "yes")

# Per-feed caps (more generous on weekly)
MAX_RESEARCH_PER_FEED_DAILY  = int(os.getenv("MAX_RESEARCH_PER_FEED_DAILY",  "5"))
MAX_RESEARCH_PER_FEED_WEEKLY = int(os.getenv("MAX_RESEARCH_PER_FEED_WEEKLY", "15"))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _ts_from_entry(e):
    try:
        if getattr(e, "published_parsed", None):
            return int(time.mktime(e.published_parsed))
        if getattr(e, "updated_parsed", None):
            return int(time.mktime(e.updated_parsed))
    except Exception:
        pass
    return 0  # fallback if feed has no dates

def select_latest_research(items, top_n):
    return sorted(items, key=lambda x: x.get("ts", 0), reverse=True)[:max(0, top_n)]

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
    Return (clean_summary, is_hot). Accepts 'HOT LIST=Yes/No' variants.
    """
    text = summary or ""
    is_hot = "HOT LIST=YES" in text.upper() or re.search(
        r"^\s*hot\s*list\s*[:=]\s*yes\s*\.?\s*$", text, re.IGNORECASE | re.MULTILINE
    ) is not None
    pattern = re.compile(r"^\s*hot\s*list\s*[:=]\s*(yes|no)\s*\.?\s*$",
                         re.IGNORECASE | re.MULTILINE)
    text = re.sub(pattern, "", text).strip()
    return text, is_hot

def _dedupe_articles(articles):
    """Deduplicate by URL if present, else by normalized title."""
    seen_urls, seen_titles, out = set(), set(), []
    for a in articles:
        url = (a.get("url") or "").strip().lower()
        title = _norm(a.get("title") or "").lower()
        if url and url in seen_urls:   continue
        if (not url) and title and title in seen_titles:  continue
        if url:   seen_urls.add(url)
        if title: seen_titles.add(title)
        out.append(a)
    return out

def _unique_by_title(items):
    """[(title, summary, url, source)] -> unique by normalized title"""
    seen, out = set(), []
    for t, s, u, src in items:
        key = _norm(t).lower()
        if key in seen: continue
        seen.add(key)
        out.append((t, s, u, src))
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
        }
        resp = _safe_get(url, params=params)
        if resp and resp.ok:
            for art in resp.json().get("articles", []):
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
            for art in resp_ai.json().get("articles", []):
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
# Research lane
# ─────────────────────────────────────────────────────────────────────────────
def fetch_research(max_per_feed: int):
    items = []
    feeds = RESEARCH_FEEDS + RESEARCH_EXTRA_FEEDS
    for feed in feeds:
        try:
            parsed = feedparser.parse(feed)
            take = max_per_feed if max_per_feed > 0 else len(parsed.entries)
            for e in parsed.entries[:take]:
               items.append({
                    "title": getattr(e, "title", "") or "(untitled)",
                    "desc": getattr(e, "summary", "") or "",
                    "url": getattr(e, "link", "") or "",
                    "source": (parsed.feed.get("title") if hasattr(parsed, "feed") else "RSS"),
                    "ts": _ts_from_entry(e),
                })
        except Exception as ex:
            print(f"[WARN] Research RSS failed: {feed} — {ex}")
    return _dedupe_articles(items)


def summarize_research(research_items):
    """Short practitioner blurbs for research."""
    if not research_items:
        return []
    if not OPENAI_API_KEY:
        return []

    client = OpenAI(api_key=OPENAI_API_KEY)
    out = []
    prompt = (
        "Summarize this quant finance paper in 3–5 sentences for a practitioner. "
        "Cover: problem, method, data, key result. End with one bullet 'Why it matters' "
        "in 1 sentence. No equations, no citations."
    )
    for idx, art in enumerate(research_items, 1):
        content = f"Title: {art['title']}\nAbstract: {art['desc']}\nURL: {art['url']}"
        try:
            chat = client.chat.completions.create(
                model=SUMMARY_MODEL, temperature=0.2, max_tokens=350,
                messages=[{"role":"system","content":prompt},{"role":"user","content":content}]
            )
            summary = (chat.choices[0].message.content or "").strip()
            out.append((art["title"], summary, art["url"], art["source"]))
            print(f"[OK] Research {idx}/{len(research_items)}")
            time.sleep(0.1)
        except Exception as e:
            print(f"[WARN] Research summarize failed: {e}")
    return _unique_by_title(out)

# ─────────────────────────────────────────────────────────────────────────────
# Summarization (news)
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
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": content}],
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
        time.sleep(0.1)

    # De-dup lists themselves
    return _unique_by_title(summaries), _unique_by_title(hot_list), _unique_by_title(ai_summaries)

# ─────────────────────────────────────────────────────────────────────────────
# Daily Overview (briefing)
# ─────────────────────────────────────────────────────────────────────────────
def build_overview_context(summaries, hot_list):
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
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing.")
    client = OpenAI(api_key=OPENAI_API_KEY)
    hot_list = hot_list or []
    context = build_overview_context(summaries, hot_list)

    sys_prompt = (
        "You are a buy-side macro/quant analyst. Write a tight 200–250 word daily market briefing "
        "for a quant audience. DO NOT ask clarifying questions. If information is thin, infer the "
        "likely market tone without fabricating exact numbers. Prioritise items under 'HOT ITEMS'. "
        "Structure:\n1) Macro & Sentiment\n2) Equities / Rates\n3) FX & Commodities (if context)\n"
        "4) What to Watch (1–3 bullets)\nNo links/preambles. 200–250 words."
    )

    chat = client.chat.completions.create(
        model=SUMMARY_MODEL,
        temperature=OPENAI_TEMP,
        messages=[{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": context}],
        max_tokens=OVERVIEW_MAX_TOKENS,
    )
    return (chat.choices[0].message.content or "").strip()

# ─────────────────────────────────────────────────────────────────────────────
# Macro Calendar lane (ICS)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_calendar_events(ics_urls_env: str):
    if not _HAS_CAL:
        print("[INFO] ics/dateutil not installed — skipping calendar lane.")
        return []
    urls = [u.strip() for u in (ics_urls_env or "").split(",") if u.strip()]
    if not urls:
        print("[INFO] No CALENDAR_ICS_URLS provided — skipping calendar lane.")
        return []
    now = datetime.utcnow().replace(tzinfo=tz.UTC)
    until = now + timedelta(days=CALENDAR_LOOKAHEAD_DAYS)

    events = []
    for url in urls:
        resp = _safe_get(url, params={})
        if not (resp and resp.ok):
            print(f"[WARN] Calendar fetch failed (HTTP): {url}")
            continue

        ctype = resp.headers.get("Content-Type", "")
        if "text/calendar" not in ctype and not url.lower().endswith(".ics"):
            print(f"[WARN] Not an ICS (Content-Type={ctype!r}): {url}")
            continue

        try:
            cal = Calendar(resp.text)
            total, kept = 0, 0
            for ev in cal.events:
                total += 1
                start = getattr(ev, "begin", None)
                if not start:
                    continue
                dt = start.datetime.replace(tzinfo=start.tzinfo or tz.UTC)
                if now <= dt <= until:
                    title = _norm(getattr(ev, "name", "") or "")
                    desc  = _norm(getattr(ev, "description", "") or "")
                    blob  = f"{title} {desc}".lower()
                    if any(k.lower() in blob for k in CALENDAR_KEYWORDS):
                        events.append((dt, title or "(untitled)", desc))
                        kept += 1
            print(f"[INFO] Calendar scanned: {url} | events={total} | matched={kept}")
        except Exception as ex:
            print(f"[WARN] Calendar parse failed for {url}: {ex}")

    events.sort(key=lambda x: x[0])
    return events[:20]

# ─────────────────────────────────────────────────────────────────────────────
# Keywords / Mini-Glossary
# ─────────────────────────────────────────────────────────────────────────────
def extract_glossary(summaries, hot_list, ai_summaries, max_terms=25):
    """Return list[(term, explanation)] as one-liners."""
    if not OPENAI_API_KEY:
        return []

    client = OpenAI(api_key=OPENAI_API_KEY)

    def pack(items, cap=10):
        lines = []
        for t, s, u, src in items[:cap]:
            lines.append(f"{t}: {s}")
        return "\n".join(lines)

    source_text = (
        "HOT:\n" + pack(hot_list, 12) + "\n\n" +
        "ARTICLES:\n" + pack(summaries, 20) + "\n\n" +
        "AI:\n" + pack(ai_summaries, 10)
    )


    sys_prompt = (
        "From the provided market/quant summaries, extract the most relevant quantitative finance terms "
        f"(max {max_terms}). Return strict JSON: "
        "[{\"term\":\"...\",\"explain\":\"short, plain-English one-liner\"}, ...]. "
        "Prefer derivatives/risk/microstructure/macro metrics (basis, carry, convexity, term premium, "
        "vol surface, order flow imbalance, VaR/CVaR, realized vs implied vol, etc.)."
    )

    chat = client.chat.completions.create(
        model=SUMMARY_MODEL,
        temperature=0.1,
        messages=[{"role":"system","content":sys_prompt},
                  {"role":"user","content":source_text}],
        max_tokens=GLOSSARY_MAX_TOKENS,
    )

    raw = (chat.choices[0].message.content or "").strip()
    try:
        data = json.loads(raw)
        out = []
        for item in data:
            term = _norm(str(item.get("term", "")))
            expl = _norm(str(item.get("explain", "")))
            if term and expl:
                out.append((term, expl))
        return out[:max_terms]
    except Exception:
        print("[WARN] Glossary JSON parse failed; skipping.")
        return []
    
    # Starter pack for days when the model returns too few terms
CORE_GLOSSARY = {
    "alpha": "Excess return vs a benchmark after adjusting for risk.",
    "beta": "Sensitivity of an asset to market moves; ~1 means market-like.",
    "volatility": "How much returns vary; higher vol = wider swings.",
    "implied volatility": "Vol inferred from option prices; a forward-looking gauge.",
    "realized volatility": "Vol computed from past returns.",
    "carry": "Return from holding an asset ignoring price changes (e.g., yield, roll).",
    "term premium": "Extra yield for holding longer-maturity bonds.",
    "basis": "Futures price minus spot; reflects carry, funding, and demand.",
    "convexity": "Nonlinear price sensitivity to yield changes (second-order duration).",
    "duration": "Bond price sensitivity to yield changes (first-order).",
    "skew": "Asymmetry of the return distribution; options: downside risk premium.",
    "VaR": "Loss threshold not expected to be exceeded with given probability.",
    "CVaR": "Average loss beyond VaR; tail risk measure.",
    "order flow imbalance": "Net buying vs selling pressure in the order book/tape.",
    "market microstructure": "How trading mechanisms & frictions affect prices."
}

def ensure_core_glossary(glossary, min_total=12):
    """
    If fewer than min_total terms, top up with CORE_GLOSSARY entries not already present. Avoid generic macro buzzwords. Prefer instrument-level terms (e.g., term structure, curve steepener/flatteners, realized vs implied basis, order book depth, gamma exposure, vanna/charm, convexity hedging).
    """
    seen = { (term or "").strip().lower() for term, _ in glossary }
    for term, expl in CORE_GLOSSARY.items():
        if len(glossary) >= min_total:
            break
        if term.lower() not in seen:
            glossary.append((term, expl))
    return glossary


# ─────────────────────────────────────────────────────────────────────────────
# Weekly Deep Dive (Sundays)
# ─────────────────────────────────────────────────────────────────────────────
def pick_weekly_deep_dive(research_summaries):
    return research_summaries[0] if research_summaries else None

def summarize_deep_dive(item):
    if not item:
        return None
    client = OpenAI(api_key=OPENAI_API_KEY)
    title, summary, url, source = item
    sys_prompt = (
        "Write a 150–250 word note to help a junior quant read this paper fast. "
        "Structure: (1) Problem (2) Approach (intuition > math) (3) Why it matters "
        "(trading/risk) (4) How to skim (which sections first). No equations, no citations, no links."
    )
    chat = client.chat.completions.create(
        model=SUMMARY_MODEL, temperature=0.3, max_tokens=450,
        messages=[{"role":"system","content":sys_prompt},
                  {"role":"user","content":f"TITLE: {title}\nSUMMARY: {summary}\nURL: {url}"}]
    )
    return (title, (chat.choices[0].message.content or "").strip(), url, source)

# ─────────────────────────────────────────────────────────────────────────────
# PDF
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf(summaries, hot_list, overview, ai_summaries,
                 research_summaries, calendar_events, weekly_deep_dive, glossary):
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

    # Overview
    story.append(Paragraph("📊 Daily Overview", styles["Heading1"]))
    story.append(Paragraph(overview or "No overview available.", styles["BodyText"]))
    story.append(Spacer(1, 12))

    # Hot List
    if hot_list:
        story.append(Paragraph("🔥 Hot List", hot_style))
        story.append(Spacer(1, 6))
        hot_list = _unique_by_title(hot_list)
        for title, summary, url, source in hot_list:
            story.append(Paragraph(f"<b>{_norm(title)}</b>", styles["Heading2"]))
            story.append(Paragraph(_norm(summary), styles["BodyText"]))
            if url:
                story.append(Paragraph(f"<font color='blue'>Read more: <u>{url}</u></font>", styles["Normal"]))
            story.append(Paragraph(f"{_norm(source)}", source_style))
            story.append(Spacer(1, 12))

    # Articles (skip Hot List)
    hot_titles = { _norm(h[0]).lower() for h in hot_list }
    story.append(Paragraph("📰 Articles", styles["Heading1"]))
    any_article = False
    for title, summary, url, source in _unique_by_title(summaries):
        if _norm(title).lower() in hot_titles:
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

    # AI & Finance (also skip Hot List)
    if ai_summaries:
        story.append(Paragraph("🤖 AI & Finance", styles["Heading1"]))
        for title, summary, url, source in _unique_by_title(ai_summaries):
            if _norm(title).lower() in hot_titles:
                continue
            story.append(Paragraph(f"<b>{_norm(title)}</b>", styles["Heading2"]))
            story.append(Paragraph(_norm(summary), styles["BodyText"]))
            if url:
                story.append(Paragraph(f"<font color='blue'>Read more: <u>{url}</u></font>", styles["Normal"]))
            story.append(Paragraph(f"{_norm(source)}", source_style))
            story.append(Spacer(1, 12))

    # Research Highlights
    if research_summaries:
        story.append(Paragraph("📑 Research Highlights", styles["Heading1"]))
        for title, summary, url, source in _unique_by_title(research_summaries):
            if _norm(title).lower() in hot_titles:
                continue
            story.append(Paragraph(f"<b>{_norm(title)}</b>", styles["Heading2"]))
            story.append(Paragraph(_norm(summary), styles["BodyText"]))
            if url:
                story.append(Paragraph(f"<font color='blue'>Read: <u>{url}</u></font>", styles["Normal"]))
            story.append(Paragraph(f"{_norm(source)}", source_style))
            story.append(Spacer(1, 12))

    # Weekly Deep Dive (Sundays)
    if weekly_deep_dive:
        dd_title, dd_note, dd_url, dd_src = weekly_deep_dive
        story.append(Paragraph("📘 Weekly Deep Dive", styles["Heading1"]))
        story.append(Paragraph(f"<b>{_norm(dd_title)}</b>", styles["Heading2"]))
        story.append(Paragraph(_norm(dd_note), styles["BodyText"]))
        if dd_url:
            story.append(Paragraph(f"<font color='blue'>Paper: <u>{dd_url}</u></font>", styles["Normal"]))
        story.append(Paragraph(f"{_norm(dd_src)}", source_style))
        story.append(Spacer(1, 12))

    # Macro Calendar (Next 7 Days)
    if calendar_events:
        story.append(Paragraph("🗓 Macro Calendar (Next 7 Days)", styles["Heading1"]))
        for when, title, desc in calendar_events:
            try:
                local_ts = when.astimezone(tz.gettz("Asia/Singapore")).strftime("%a %d %b %H:%M")
            except Exception:
                local_ts = when.strftime("%a %d %b %H:%M UTC")
            story.append(Paragraph(f"<b>{local_ts}</b> — {title}", styles["BodyText"]))
            if desc:
                story.append(Paragraph(_norm(desc), styles["BodyText"]))
            story.append(Spacer(1, 6))

    # Keywords / Mini-Glossary (bottom)
    if glossary:
        story.append(Spacer(1, 6))
        story.append(Paragraph("📚 Quant Keywords & Mini-Glossary", styles["Heading1"]))
        for term, expl in glossary:
            story.append(Paragraph(f"<b>{_norm(term)}</b> — [{_norm(expl)}]", styles["BodyText"]))
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
    # Use app password with no spaces
    app_pw = PASSWORD.replace(" ", "")

    # Build a clean recipient list (skip blanks/whitespace)
    recipients = [r.strip() for r in [EMAIL, EMAIL2] if r and r.strip()]
    if not recipients:
        print("[WARN] No valid recipients found — skipping email send.")
        return

    subject = "Your Quant Daily Digest"
    body = "Attached is today's quant digest PDF."

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)  # header (display only)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with open(file_path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f'attachment; filename="{os.path.basename(file_path)}"'
        )
        msg.attach(part)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender, app_pw)
            refused = server.sendmail(sender, recipients, msg.as_string())

        if refused:
            # Dict of { "bad@addr": (code, b"message"), ... }
            print(f"[WARN] Some recipients were refused: {refused}")
            delivered = [r for r in recipients if r not in refused]
            if delivered:
                print(f"[OK] Delivered to: {', '.join(delivered)}")
        else:
            print(f"[OK] Email sent to: {', '.join(recipients)}")

    except smtplib.SMTPAuthenticationError as e:
        print(f"[ERR] Gmail auth failed (check app password): {e}")
    except smtplib.SMTPRecipientsRefused as e:
        print(f"[ERR] All recipients refused: {e.recipients}")
    except Exception as e:
        print(f"[ERR] SMTP error: {e}")

    if not EMAIL or not PASSWORD:
        print("[WARN] EMAIL or PASSWORD missing — skipping email send.")
        return

    sender = EMAIL

    # Build a clean recipient list (skip blanks/None)
    raw_recipients = [EMAIL, EMAIL2]
    recipients = [r.strip() for r in raw_recipients if r and r.strip()]
    if not recipients:
        print("[WARN] No valid recipients found — skipping email send.")
        return

    subject = "Your Quant Daily Digest"
    body = "Attached is today's quant digest PDF."

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)   # header only (for display)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with open(file_path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f'attachment; filename="{os.path.basename(file_path)}"'
        )
        msg.attach(part)

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(sender, PASSWORD)
        errors = server.sendmail(sender, recipients, msg.as_string())
        server.quit()

        if errors:
            # errors is a dict: { "bad@addr": (code, b"message"), ... }
            print(f"[WARN] Some recipients were refused: {errors}")
            delivered = [r for r in recipients if r not in errors]
            if delivered:
                print(f"[OK] Email delivered to: {', '.join(delivered)}")
        else:
            print(f"[OK] Email sent to: {', '.join(recipients)}")

    except Exception as e:
        print(f"[WARN] Failed to send email: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[*] Fetching news…")
    articles = fetch_news()
    print(f"[*] Collected {len(articles)} articles (post-dedupe).")

    if not articles:
        print("[ERR] No articles fetched. Check your API keys/network/feeds.")
        sys.exit(1)

    print("[*] Summarizing news…")
    summaries, hot_list, ai_summaries = summarize_articles(articles)
    print(f"[*] Summaries: {len(summaries)} | Hot: {len(hot_list)} | AI: {len(ai_summaries)}")

    if not summaries and not ai_summaries:
        print("[ERR] No summaries produced. Aborting before PDF/email.")
        sys.exit(1)

    # Optional lanes
    print("[*] Research cadence decision…")
    DO_RESEARCH = FORCE_RESEARCH or (RUN_RESEARCH == "always") or (
        RUN_RESEARCH == "weekly" and datetime.utcnow().weekday() == RESEARCH_DAY_UTC
    )
    research_items = []
    research_summaries = []
    # Weekly deep dive (Sundays UTC)
    weekly_deep_dive = None
    if datetime.utcnow().weekday() == 6:
        print("[*] Weekly deep dive…")
        weekly_deep_dive = summarize_deep_dive(pick_weekly_deep_dive(research_summaries))

    if DO_RESEARCH:
        max_per = MAX_RESEARCH_PER_FEED_WEEKLY if (RUN_RESEARCH == "weekly" or FORCE_RESEARCH) else MAX_RESEARCH_PER_FEED_DAILY
        print(f"[*] Fetching research… (max_per_feed={max_per})")
        research_items = select_latest_research(research_items, RESEARCH_TOP_N)
        research_summaries = summarize_research(research_items)
        print(f"[*] Research items: {len(research_items)} | Summarized: {len(research_summaries)}")

        # Deep dive only if it's the configured research day (default Sunday UTC),
        # or if you force it.
        if FORCE_RESEARCH or datetime.utcnow().weekday() == RESEARCH_DAY_UTC:
            print("[*] Weekly deep dive…")
            weekly_deep_dive = summarize_deep_dive(pick_weekly_deep_dive(research_summaries))
    else:
        print("[*] Skipping research this run (RUN_RESEARCH=%r, RESEARCH_DAY_UTC=%d)" % (RUN_RESEARCH, RESEARCH_DAY_UTC))

    # Weekly deep dive (Sundays UTC)
    weekly_deep_dive = None
    if datetime.utcnow().weekday() == 6:
        print("[*] Weekly deep dive…")
        weekly_deep_dive = summarize_deep_dive(pick_weekly_deep_dive(research_summaries))

    print("[*] Fetching calendar…")
    calendar_events = fetch_calendar_events(CALENDAR_ICS_URLS)
    print(f"[*] Calendar events found: {len(calendar_events)}")

    print("[*] Building daily overview…")
    overview = overall_summary(summaries, hot_list)

    print("[*] Extracting glossary…")
    glossary = extract_glossary(summaries, hot_list, ai_summaries)
    glossary = ensure_core_glossary(glossary, min_total=18)

    print("[*] Generating PDF…")
    pdf = generate_pdf(summaries, hot_list, overview, ai_summaries,
                       research_summaries, calendar_events, weekly_deep_dive, glossary)
    print(f"[OK] Generated {pdf}")

    print("[*] Sending email (if configured)…")
    send_email(pdf)
    print("[DONE] Quant Daily Digest complete.")
