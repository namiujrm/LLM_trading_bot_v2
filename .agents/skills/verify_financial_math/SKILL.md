---
name: verify_financial_math
description: >-
  Verifies quantitative financial formulas, Quarter-Kelly sizing invariants,
  Triple-Barrier evaluation, and Fee-to-Edge constraints via Hypothesis property-based testing.
---

# Financial Math Verification Runbook (`verify_financial_math`)

When modifying quantitative domain services or financial models:

## 1. Core Invariants to Verify
- **Fee-to-Edge Gate:** `raw_tp_pct >= min_tp_to_cost_multiple * cost_pct` must hold before any sizing is approved.
- **Quarter-Kelly Sizing:** Position size must never exceed `kelly_fraction * full_kelly * uncertainty_penalty * vol_scalar`.
- **Drawdown Circuit Breakers:** If `current_daily_dd_pct >= daily_drawdown_halt_pct` or `current_max_dd_pct >= max_drawdown_freeze_pct`, trading MUST be halted (`approved=False`).
- **Triple-Barrier Invariants:**
  - If low <= stop_loss_price on a LONG bar, SL must trigger.
  - If high >= take_profit_price on a LONG bar, TP must trigger.
  - MAE is the maximum adverse drawdown observed during the holding period.
  - MFE is the maximum favorable excursion observed during the holding period.

## 2. Execution Command
```powershell
uv run pytest tests/property/ -v
```
Runs Hypothesis property tests generating hundreds of random boundary price and volatility series to stress-test mathematical edge cases.
