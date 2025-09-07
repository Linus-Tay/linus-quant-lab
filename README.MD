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

The **Quant Digest** is an automated system that runs every day at **7:00 AM SGT**.  

Here’s what it does:
1. Fetches financial and market news via [NewsAPI](https://newsapi.org/) and RSS feeds (CNBC, Reuters, Yahoo Finance, FT Alphaville).  
2. Summarizes and filters content using **OpenAI GPT-4.1-mini**, with prompts tailored for relevance to quants.  
3. Produces a polished PDF report that includes:  
   - **📊 Daily Overview** of the markets.  
   - **🔥 Hot List** (must-know events).  
   - **📰 Other Relevant Articles**, without duplication.  
   - A **Key Terms** glossary for financial/quant concepts.  
4. Uploads the digest to GitHub → accessible in:  
