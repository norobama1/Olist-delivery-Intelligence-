# Case Study: Predicting and Preventing Late Deliveries on a Brazilian E-Commerce Marketplace

**Project:** Delivery Intelligence — Olist E-Commerce
**Code:** [github.com/norobama1/Olist-delivery-Intelligence](https://github.com/norobama1/Olist-delivery-Intelligence)
**Tools:** Python, XGBoost, SHAP, Power BI, FastAPI, Docker, pytest, sentence-transformers (RAG)

---

## The Business Problem

Late deliveries erode customer trust and drive refunds, bad reviews, and churn on marketplace platforms. Olist — a Brazilian e-commerce marketplace connecting small sellers to major retail channels — had no systematic way to know *which* orders were at risk of arriving late, or *why* delays happened in the first place.

Using 110,781 delivered orders (Sep 2016–Aug 2018) from the public Olist dataset, this project answers three questions a delivery operations team would actually ask:

1. Where and when are delays concentrated?
2. Which orders should we flag as high-risk before they even ship?
3. What do customers say when deliveries go wrong — and can that language tell us anything the numbers miss?

## Key Findings

Overall delay rate across the dataset: **6.6%**. But that average hides sharp, actionable patterns:

- **Seasonality is predictable, not random.** March and November/December delay rates spike to 14–18%, driven by post-Carnival logistics disruption and Black Friday volume. This is a planning window operations can staff around in advance.
- **Route matters more than raw distance.** DF→ES orders delay 30.4% of the time; DF→SC delays 19.2%; same-state SP→SP orders stay under 5%. A simple straight-line distance metric would have missed this — the *specific route*, not the mileage, is the driver.
- **A small group of sellers causes most of the damage.** Just 24% of active sellers (321 of 1,344) are responsible for 80% of all delays. This turns a diffuse company-wide problem into a targeted seller-management one.
- **Overpromising backfires badly.** Sellers who promise delivery in ≤7 days delay 16.5% of the time, versus 4.7% for 30+ day windows — a 3.5x gap. Carriers also deliver, on average, 15 days faster than the estimate given to customers, suggesting estimates are heavily padded in the other direction for some segments while badly underestimated for others.
- **Freight cost predicts risk; product weight doesn't.** High-freight orders delay at 8.0% vs. 5.2% for low-freight — a proxy for shipping complexity that a weight-based rule would miss entirely.

## The Model

A gradient-boosted model (XGBoost) was trained to predict delay probability at the time of purchase, using 15 features spanning temporal, physical, logistics, and geographic categories — including a leak-free "seller delay rate" feature computed only from each seller's *past* orders.

Three models were compared on a stratified 70/15/15 split, selected on validation PR-AUC (appropriate given the 6.6% class imbalance — plain accuracy would be misleading here):

| Model | Val PR-AUC |
|---|---|
| Logistic Regression | 0.139 |
| Random Forest | 0.221 |
| **XGBoost (tuned + calibrated)** | **0.233** ✅ |

**Business-aware thresholding:** rather than using a default 0.5 cutoff, the classification threshold (0.118) was chosen to minimize a cost function weighting a missed delay 5x worse than a false alarm — reflecting that failing to warn about a real delay is more costly to the business than an unnecessary flag. On the held-out test set, this gives 51% recall on actual delays at 23% precision — i.e., the model catches roughly half of all delays before they happen, trading some false positives to do so.

**Why it matters over a black box:** SHAP analysis confirms the top drivers are `purchase_month` (seasonality), `estimated_days` (short promise windows), and `customer_state` (destination complexity) — meaning the model's behavior lines up with, and reinforces, the EDA findings above rather than contradicting them. That agreement is what makes the model trustworthy enough to act on.

## From Model to Action

Two things were added specifically so this isn't just a notebook exercise:

- **A FastAPI prediction service** (Dockerized, with pytest smoke tests) exposing `/predict` and `/predict/batch` endpoints, returning a delay probability and risk band — the shape of what an operations dashboard or seller-facing alert system would call in production.
- **A semantic search layer over 40k customer reviews** (multilingual sentence-transformers, handling Portuguese natively), letting you compare *what customers actually say* on delayed vs. on-time orders for the same complaint type — turning free-text reviews into a queryable qualitative signal alongside the structured model.

## Recommendation

If I were presenting this to Olist's operations team, the recommendation would be:

1. **Triage the 321 high-delay sellers** identified above with targeted support or stricter listing requirements — this is the single highest-leverage lever, since it's a small, addressable population responsible for most of the problem.
2. **Recalibrate delivery estimates by route, not just distance** — the DF→ES/DF→SC pattern shows generic distance-based estimates are miscalibrated for specific corridors.
3. **Staff and message proactively around March and November** — since the seasonal spike is predictable a year out, it's a scheduling problem, not a surprise.
4. **Pilot the risk-scoring API** on new orders as a pre-shipment alert, using the cost-weighted threshold so operations isn't flooded with false alarms.

---

*Full code, EDA notebook, dashboard, and API are available in the [GitHub repository](https://github.com/norobama1/Olist-delivery-Intelligence).*
