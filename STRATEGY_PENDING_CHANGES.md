# Quantitative Trade Excursion Analysis: MAE vs. MFE

**Document Status:** Pending Strategy Changes  
**Author:** Quant Research & Strategy Team  
**Focus:** Intra-Trade Excursion Profiling & Risk/Reward Optimization  

---

## 1. Overview & Metrics Definition

This document analyzes the historical trade excursion profile to optimize stop-loss placement, trailing stops, and profit targets using **R-Multiples** ($1R = \text{Initial Risk Unit}$).

* **MAE (Maximum Adverse Excursion):** The maximum negative unrealized price movement (drawdown) experienced during a trade's lifetime before closing.
* **MFE (Maximum Favorable Excursion):** The maximum positive unrealized price movement (peak gain) achieved during a trade's lifetime before closing.

---

## 2. Key Empirical Findings from Scatter Plot Data

Based on the **Intra-Trade MAE vs. MFE Excursion Scatter (in R-Multiples)** analysis:

### 🛑 Finding 1: The "Cliff Edge" Hard Stop at `-1.0R`
* **Observation:** Winning trades (green circles) almost exclusively stay within an MAE range of **`0.0R` to `-0.85R`**.
* **Zero Recovery:** Historically, once a position breaches **`-1.0R`**, the probability of recovery to profitability approaches **0%**.
* **Catastrophic Outlier:** Observed a severe negative excursion outlier extending to **`-4.15R`**, indicating lack of hard risk cutoffs during anomalous market swings.

### 💔 Finding 2: "Tragic Losers" (Unprotected Winners)
* **Observation:** Several losing trades (red crosses) experienced substantial intra-trade gains of **`+1.0R` to `+1.8R` MFE** before completely reversing into losses.
* **Problem:** Holding positions through substantial initial expansion without profit preservation mechanisms degrades overall strategy expectancy.

### 🎯 Finding 3: The `2.0R` to `2.5R` Natural Momentum Target
* **Observation:** The core cluster of winning trades demonstrates strong momentum up to **`2.0R` – `2.5R` MFE**, with diminishing distribution density beyond `3.0R` (only one outlier reached `5.0R`).
* **Conclusion:** Targeting `2.2R` to `2.5R` captures the natural distribution peak without exposing equity to excessive giveback.

---

## 3. Pending Strategy Implementation Rules

```mermaid
flowchart TD
    Entry["Trade Entry (0.0R)"] --> Check{"Price Movement"}
    Check -->|"Dips to -1.0R MAE"| HardStop["🛑 Hard Stop Loss Triggered (Exit at -1.0R)"]
    Check -->|"Expands to +1.0R MFE"| MoveBE["⚡ Trigger Breakeven: Move SL to Entry (0.0R)"]
    MoveBE --> Trailing["Activate ATR Trailing Stop / TP at 2.2R - 2.5R"]
    Trailing --> Win["🎯 Close as Winner / Risk-Free Trade"]
```

### 📋 Rule Checklist for Code Implementation:

1. **Rule 1: Hard Stop-Loss Cutoff at `-1.0R`**
   * *Logic:* Enforce a strict liquidation order when adverse excursion touches $-1.0R$.
   * *Impact:* Eliminates extreme tail risk (such as the $-4.15R$ drawdown event).

2. **Rule 2: Automated Breakeven Activation at `+1.0R` MFE**
   * *Logic:* When high-water mark reaches $+1.0R$, adjust the protective stop to the initial entry price ($0.0R$).
   * *Impact:* Converts at least 30–40% of previous losers into neutral (break-even) trades.

3. **Rule 3: Optimized Take-Profit Objective (`2.2R` – `2.5R`)**
   * *Logic:* Place limit target or tight trailing acceleration within the $2.2R$ to $2.5R$ band.
   * *Impact:* Maximizes win rate and profit factor while avoiding runner decay.

---

## 4. Python Implementation Spec (Pseudocode)

```python
# Trade Excursion & Risk Management Logic
initial_risk_per_share = entry_price - initial_stop_price  # 1R

# Tracking during active candle loop:
current_r = (current_price - entry_price) / initial_risk_per_share
mfe_r = max(mfe_r, current_r)
mae_r = min(mae_r, current_r)

# Rule 1: Hard Stop Protection
if mae_r <= -1.0:
    close_position(reason="Hard Stop -1.0R Hit")

# Rule 2: Dynamic Breakeven Trigger
if mfe_r >= 1.0 and stop_loss_price < entry_price:
    stop_loss_price = entry_price  # Move SL to Breakeven (0R)

# Rule 3: Take-Profit Target
if current_r >= 2.25:
    close_position(reason="Target 2.25R Achieved")
```

---

## 5. Next Steps
- [ ] Implement Rules 1 & 2 in `backtest_optimizer.py` and `dashboard_app.py`.
- [ ] Run comparative backtest (Baseline vs. Excursion-Optimized).
- [ ] Validate updated MAE/MFE distribution on `http://localhost:8501`.
