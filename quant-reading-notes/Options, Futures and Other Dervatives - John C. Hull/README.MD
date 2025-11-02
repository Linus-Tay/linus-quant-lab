# 📘 Options, Futures, and Other Derivatives — John C. Hull  

Often referred to as the **“Quant Bible”**, this was my very first book related to the field of quantitative finance. I personally bought the **11th Global Edition (physical copy)**, and I was excited — even reassured — when I read the first sentence of the synopsis:  

> *“A go-to book not only for instructors and students.”*  

That line gave me the strength and confidence to commit to the purchase, marking the real beginning of my **quantitative journey**.  

---

## 📝 How I Document My Learning  

In this README, I write **simple reflections** guided by a few core questions:  

- **What clicked?** → the core idea in my own words.  
- **What was tough?** → where I struggled and need to revisit.  
- **Takeaway?** → the one-liner worth remembering.  

This acts as both a way to **internalize concepts** and to **document my journey of studying quant** while pursuing my Computer Science degree at the **Singapore Institute of Technology**.  

Additionally, I attempt all the **practice questions** for the chapters. I don’t attach my attempts here — since I use them as a learning tool rather than a test of skill — which feels fair given I’m still new (“green”) to this field.  

---

## 📖 Chapter 1: Introduction  

**1. What clicked?**  
The chapter introduced the basics: calls, puts, options, forwards, and the types of traders. I especially appreciated the snapshot on **Lehman Brothers** and their downfall during the 2008 financial crisis — it grounded the theory in a real-world event. Overall, it gave me a solid foundation in market basics.  

**2. What was tough?**  
Most of the chapter was straightforward. The one area that opened my eyes was the many combinations of long calls, short calls, long puts, and short puts. These can be structured into strategies like collars or bull spreads. It made me realize how deep the strategies go.  

**3. Takeaway**  
There are three main types of traders: **hedgers, speculators, and arbitrageurs** — and each uses different techniques worth studying in depth.  

---

## 📖 Chapter 2: Futures Markets and Central Counterparties  

**1. What clicked?**  
I enjoyed learning about the differences between **forwards and futures**, and the role of **OTC intermediaries** like CCPs and exchange **clearing houses**. I found it fascinating how all traders come together in one place, with risks of defaults and collateral managed by intermediaries. I also enjoyed connecting futures concepts to actual **day-trading tools** like limit orders, stop orders, and stop-limit orders — and understanding how traders can both protect themselves and take advantage of positions.  

**2. What was tough?**  
It was challenging to fully grasp Hull’s explanations here. For example, when he introduced concepts like **open interest** or the obligations of a short futures position, it often felt very subtle in the text. While doing practice questions, I realized my answers weren’t framed the way Hull expected. This forced me to dive deeper into each question, carefully breaking down keywords and key concepts.  

**3. Takeaway**  
Futures differ from forwards mainly because of **daily settlement, standardization, and the role of clearing houses/CCPs**. These features reduce credit risk but also introduce daily cash flow considerations — making futures both more secure and more dynamic than forwards.  

---

## 📖 Chapter 3: Hedging Strategies Using Futures

**1. What clicked?**  
I had always heard about hedging, but now I have a much better understanding of what it actually means. The purpose of hedging is to mitigate risk to the fullest extent possible against fluctuations in the price of goods, foreign exchange rates, and the level of the stock market. A particular point that I took to heart was that hedging is not always the right play. In a market where you might have competitors who are not hedging, you stand to lose out on profits and also deal with the same downside as they do. The strategies discussed were really intriguing and overall, I liked this chapter very much. 

**2. What was tough?**  
At times, I had to double back on the formulas presented, as I wanted to memorize them, but I realized that it is not about memorization but rather understanding. When calculating the minimum variance hedge ratio, if you understand the math but not the relationship that the value shows you regarding changes in the spot and futures prices, then there is no point at all. Hence, although grasping the mathematical concept was slightly challenging, pairing it with the logic behind the math made it more comprehensible.

**3. Takeaway**  
Calculations such as variance, measuring beta, or even comparing your portfolio against the market are performed to ensure that you get as close to the perfect hedge as possible are interesting and necessary. However, just because you get the perfect hedge does not mean you get the best possible outcome.

---

## 📖 Chapter 4: Interest Rates

**1. What clicked?**  
Through this chapter, I realised there are many more underlying considerations when it comes to interest rates, such as whether it is compounded and the frequency of compounding. However, what I really enjoyed learning about was bonds and their pricing, as I never really understood them until I read this chapter.

**2. What was tough?**  
Comprehending the differences between zero rates, forward rates, and forward rate agreements was quite challenging. To really be able to pinpoint the exact differences and know them by heart was challenging for me.

**3. Takeaway**  
All in all, I know interest rates and the relevant durations greatly affect the prices of bonds. Moreover, the convexity of a bond is what is important to really understand what happens to a bond over a duration of time.

---

## 📖 Chapter 5:  Determination of Forward and Futures Prices

**1. What clicked?**
Differences in forward and future prices can lead to many opportunities for arbitrageurs. However, to know exactly when you have an arbitrage opportunity, you would need to be able to determine the different kinds of investment assets as well as the situation (for example, whether it provides a known dollar income, yield, or does not provide income). This chapter was all about that, and it was intriguing to see the way that these would determine the relevant formulas to calculate arbitrage opportunities for future prices against forward prices.

**2. What was tough?**
Nothing was particularly challenging, but I had to spend more time on understanding and absorbing the exact differences between all the different variables that could affect the determination of a future price.

**3. Takeaway**
There might be some situations which may seem like arbitrage opportunities, but that is when you must always remember to take into consideration the cost of carry as well as possible storage fees.
