# Context-Aware Cross-Sell & Upsell (CSAO) Recommendation Engine

A highly optimized, deep learning-based recommendation system designed to predict and rank add-on items (cross-sell/upsell) at checkout. The system processes real-time user metadata, current cart state, and historical features to deliver relevant item recommendations within strict latency budgets (<150ms).

## 🚀 Key Features
* **Sequential Cart Understanding:** Uses a Transformer Encoder to capture dependencies between items already in the cart.
* **Context-Aware Ranking:** Infuses real-time metadata (e.g., time of day, location, device) to surface time-sensitive recommendations (e.g., "Late Night Sweets").
* **Low-Latency Design:** Structured to score candidate items efficiently to meet real-time production delivery constraints.

---

## 📦 API Contract Reference

### 1. Recommendation Request
**Endpoint:** `/api/v1/recommend/csao`  
**Method:** `POST`

```json
{
  "metadata": {
    "user_id": "UX_9821",
    "city": "Mumbai",
    "time_slot": "21:30", 
    "device": "android"
  },
  "cart_state": {
    "items": ["101_double_cheese_burger"],
    "total_value": 350.00,
    "cuisine": "American"
  },
  "candidates_count": 50 
}
