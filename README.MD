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

- **Quant Notes**  
  - Short reflections on concepts from Hull, Shreve, Joshi, and other quant texts.  
  - Each note is light and structured around a few quick questions:  
    - **What clicked?** (the core idea in my own words)  
    - **What was tough?** (concepts I struggled with, need to revisit)  
    - **Takeaway?** (the one-liner worth remembering)  
    - *(Optional)* **How I applied it?** (quick code, payoff diagram, or example)  

  These are not full summaries — just snapshots of how I internalized a concept and reminders of where to revisit later.

---

## 📰 Quant Digest

The **Quant Digest** runs daily at **07:00 SGT** (scheduled via GitHub Actions at **23:00 UTC**).

### What it does
1. **Fetches news** from NewsAPI + RSS (CNBC, Reuters, Yahoo Finance, FT Alphaville).
2. **Summarizes** with *GPT-4.1-mini* using quant-aware prompts (focus on data, flows, policy, positioning).
3. Builds a polished **PDF** with:
   - **📊 Daily Overview** (macro tone by asset class).
   - **🔥 Hot List** (flagged must-reads).
   - **📰 Articles** (auto-deduped vs Hot List).
   - **🤖 AI & Finance** (ML/AI news relevant to markets).
   - **📑 Research Highlights** (arXiv q-fin, BIS/CEPR/Fed/NBER + Man Institute / Alpha Architect / Quantocracy).
   - **🗓 Macro Calendar (next 7 days)** (optional; from ICS feeds).
   - **📘 Weekly Deep Dive** (Sundays UTC: 1 paper in 150–250 words).
   - **📚 Quant Keywords & Mini-Glossary** (plain-English one-liners with a small “core” fallback).
4. **Outputs** the PDF to `quant-digest/GeneratedDigests/`, commits it (if changed), uploads as an artifact, and optionally **emails** it to you.

### Configuration
Set as repo **secrets** or local **env vars**:


