import os
import requests
import feedparser
from datetime import datetime
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

# Load .env
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")

TOPICS = ["quant finance", "derivatives", "hedge funds", "trading", "markets"]

RSS_FEEDS = [
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # CNBC Markets
    "https://www.reutersagency.com/feed/?best-topics=markets", # Reuters Markets
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC,^IXIC,^DJI&region=US&lang=en-US", # Yahoo Finance
    "https://ftalphaville.ft.com/feed/"  # FT Alphaville
]

def fetch_news():
    """Fetch from NewsAPI + RSS feeds"""
    articles = []

    # NewsAPI
    url = f"https://newsapi.org/v2/everything?q={' OR '.join(TOPICS)}&language=en&sortBy=publishedAt&pageSize=10&apiKey={NEWS_API_KEY}"
    resp = requests.get(url).json()
    for art in resp.get("articles", []):
        articles.append({
            "title": art.get("title"),
            "desc": art.get("description") or "",
            "url": art.get("url"),
            "source": art.get("source", {}).get("name", "Unknown")
        })

    # RSS feeds
    for feed in RSS_FEEDS:
        parsed = feedparser.parse(feed)
        for entry in parsed.entries[:5]:  # limit per feed
            articles.append({
                "title": entry.title,
                "desc": entry.get("summary", ""),
                "url": entry.link,
                "source": parsed.feed.get("title", "RSS Feed")
            })

    return articles

def summarize_articles(articles):
    """Use GPT to summarize + flag hot items"""
    client = OpenAI(api_key=OPENAI_API_KEY.strip())
    summaries, hot_list = [], []

    for art in articles:
        content = f"Title: {art['title']}\nSource: {art['source']}\nDescription: {art['desc']}\nURL: {art['url']}"

        chat = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Summarize this financial news for a quant engineer in training. Focus on quant/market-relevant details, explain key terms briefly. Then classify if this should go to a 'HOT LIST' of must-know news (yes/no)."},
                {"role": "user", "content": content},
            ],
            max_tokens=250,
        )

        summary_text = chat.choices[0].message.content
        summaries.append((art["title"], summary_text, art["url"], art["source"]))

        if "HOT LIST: YES" in summary_text.upper():  # simple trigger
            hot_list.append((art["title"], summary_text, art["url"], art["source"]))

    return summaries, hot_list

def overall_summary(summaries):
    """Generate daily overall summary + key terms"""
    client = OpenAI(api_key=OPENAI_API_KEY.strip())
    combined = "\n".join([s[1] for s in summaries])

    chat = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a financial analyst creating a quant daily briefing."},
            {"role": "user", "content": f"Here are today's article summaries:\n{combined}\n\nProvide:\n1. A 200-word overall summary of today's markets and quant-relevant news.\n2. A bullet list of top quant takeaways.\n3. A 'Key Terms' section with important financial/quant terms."}
        ],
        max_tokens=500,
    )
    return chat.choices[0].message.content

def generate_pdf(summaries, hot_list, overview):
    """Make a nice PDF"""
    output_dir = os.path.join("quant-digest", "GeneratedDigests")
    os.makedirs(output_dir, exist_ok=True)  # ensure folder exists

    filename = os.path.join(
        output_dir, f"Quant Digest_{datetime.now().strftime('%Y-%m-%d')}.pdf"
    )

    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    hot_style = ParagraphStyle("HotStyle", parent=styles["Heading1"], textColor=colors.red)
    source_style = ParagraphStyle("SourceStyle", parent=styles["Normal"], textColor=colors.grey, fontSize=9)

    # Title
    story.append(Paragraph("Quant Daily Digest", styles["Title"]))
    story.append(Spacer(1, 12))

    # HOT LIST
    if hot_list:
        story.append(Paragraph("🔥 HOT LIST: Must-Know News", hot_style))
        story.append(Spacer(1, 6))
        for title, summary, url, source in hot_list:
            story.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))
            story.append(Paragraph(summary, styles["BodyText"]))
            story.append(Paragraph(f"<font color='blue'>Read more: <u>{url}</u></font>", styles["Normal"]))
            story.append(Paragraph(f"{source}", source_style))
            story.append(Spacer(1, 12))

    # Overview
    story.append(Paragraph("📊 Daily Overview", styles["Heading1"]))
    story.append(Paragraph(overview, styles["BodyText"]))
    story.append(Spacer(1, 12))

    # Articles
    story.append(Paragraph("📰 Articles", styles["Heading1"]))
    for title, summary, url, source in summaries:
        story.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))
        story.append(Paragraph(summary, styles["BodyText"]))
        story.append(Paragraph(f"<font color='blue'>Read more: <u>{url}</u></font>", styles["Normal"]))
        story.append(Paragraph(f"{source}", source_style))
        story.append(Spacer(1, 12))

    doc.build(story)
    return filename

def send_email(file_path):
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
        part.add_header('Content-Disposition', f'attachment; filename={file_path}')
        msg.attach(part)

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender, PASSWORD)
    server.sendmail(sender, recipient, msg.as_string())
    server.quit()

if __name__ == "__main__":
    articles = fetch_news()
    summaries, hot_list = summarize_articles(articles)
    overview = overall_summary(summaries)
    pdf = generate_pdf(summaries, hot_list, overview)
    send_email(pdf)
    print(f"Generated {pdf}")
