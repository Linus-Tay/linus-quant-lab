# 🧮 Linus Quant Lab

Welcome to **Linus Quant Lab** — my personal sandbox for learning, experimenting, and building tools at the intersection of **Computer Science** and **Quantitative Finance**.  

This repository documents my journey from first principles all the way towards advanced quant research and engineering. It is both a **learning journal** and a **toolkit** that will keep expanding as I explore financial models, coding implementations, and real-world data applications.

---

## 📚 What You'll Find Here

This lab is split into multiple explorations:

- **Fundamental Quant Models**  
  - Classic pricing formulas (e.g. Black–Scholes, Binomial Trees).  
  - Risk/return metrics calculators.  
  - Simulations (Monte Carlo, stochastic processes).  
  - Small utilities to sharpen my math + coding intuition.  

- **Applied Quant Experiments**  
  - Backtesting frameworks.  
  - Predictive models.  
  - Integrations with APIs & real-world data.  

- **Quant Digest (Daily News Brief)**  
  - A fully automated financial news digest (see below).  

---

## 📰 Quant Digest

The **Quant Digest** runs daily at **07:00 SGT** (scheduled via GitHub Actions at `23:00 UTC`).

### What it does
1. **Fetches news** from NewsAPI and RSS (CNBC, Reuters, Yahoo Finance, FT Alphaville).
2. **Summarizes** with *GPT-4.1-mini* using quant-aware prompts.
3. Builds a polished **PDF** with:
   - **📊 Daily Overview** (macro tone, asset classes).
   - **🔥 Hot List** (top items only).
   - **📰 Articles** (no duplicates with Hot List).
   - **🤖 AI & Finance** (industry/ML in markets).
   - **📑 Research Highlights** (arXiv + optional AQR / Man / SSRN).
   - **🗓 Macro Calendar (7 days)** (optional; via ICS feeds).
   - **📘 Weekly Deep Dive** (Sundays: 1 paper explained in 150–250 words).
   - **📚 Quant Keywords & Mini-Glossary** (plain-English terms).
4. Uploads the PDF to the repo and as a workflow artifact. Emails a copy to me.

### Configure
Create `.env` locally or add **GitHub Secrets** (same names):

