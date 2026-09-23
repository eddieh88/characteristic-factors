# Pre-registration: do characteristic-driven betas change the answer?

Written before any model is fit. Nothing below is revised after seeing results.

## Why

Every result in this project used **PCA-5 residuals from prices alone**. The
literature's working models do not:

| | Gu–Kelly–Xiu / Chen–Pelger–Zhu | this project |
|---|---|---|
| inputs | firm characteristics | prices only |
| structure | factor model with no-arbitrage | free-form |
| innovation | the objective | ✓ Sharpe objective |

One of three. And when we tested nonlinear betas mid-project (quadratic,
asymmetric) they made things worse — AR(1) of +0.0149 and +0.0749 against
linear's −0.0037. GKX identifies the error: **the nonlinearity was in the wrong
argument.** Nonlinear in *factors* on 60 daily observations is hopeless.
Nonlinear in *characteristics*, with parameters shared across the whole panel, is
the version with an enormous effective sample.

This is also the authors' own best arm. Their IPCA-5 scores 4.16 against PCA-5's
3.36, and we never built it.

## Scope, fixed now

**Monthly**, 2002-01 → 2024-12 (OSAP ends 2024-12). Universe: top 1,500 by
trailing 21-day dollar volume, point-in-time, from the survivorship-free panel.
Characteristics: the 209 OSAP signals, **lagged one month**, cross-sectionally
z-scored, winsorised ±3, missing set to 0 (the cross-sectional mean after
z-scoring). Joined via `crosswalk_sf.csv` — 7,638 symbols, 44.8% delisted.

Three models, K = 5 factors throughout:

1. **PCA-5** — the existing construction, as benchmark.
2. **IPCA (linear)** — betas a *linear* map of characteristics; factors solved
   per period. Alternating least squares. This is Kelly–Pruitt–Su and the special
   case GKX generalises.
3. **Autoencoder (nonlinear)** — same structure, betas a small neural function of
   characteristics. One hidden layer, 32 units, so the parameter count stays far
   below the panel size.

Walk-forward: fit on the trailing 60 months, apply to the next 12, roll. No
in-sample evaluation anywhere.

## The statistics, fixed now

**Primary.** Net Sharpe of the long/short portfolio sorted on each model's
predicted return (β·f), monthly rebalance, **2017-01 → 2024-12**, after 2bp
one-way cost and 35bp/yr borrow. Gross, turnover and breakeven reported
alongside.

**Secondary, reported but not decisive.** Residual AR(1) under each model. If
characteristic-driven betas produce residuals that mean-revert where PCA-5's do
not, that would reopen Steps 1–7. Recorded as a check on those conclusions, not
as a strategy.

## The null, fixed now

Permute characteristics **across stocks within each month**, preserving every
marginal distribution and the cross-sectional spread, destroying only the link
between a firm's characteristics and its own subsequent return. Refit the whole
model on permuted data. 10 draws — refitting is expensive, and 10 bounds the
95th percentile adequately given the thresholds below.

## Decision thresholds

**WORKS** — autoencoder net Sharpe ≥ **+0.50**, above every null draw, breakeven
≥ 6bp, and **above IPCA by at least 0.20**. The last clause matters: if the
nonlinear version does not beat the linear one, the finding is about
characteristics, not about nonlinearity.

**DEAD** — net ≤ **+0.15**, or inside the null distribution.

**AMBIGUOUS** — anything else, reported as such. Not re-cut by K, hidden width,
universe size, or rebalance frequency.

## Prior, recorded now

**Weakly negative on tradeability, genuinely uncertain on the diagnostic.**

Avramov, Cheng & Metzker (2023) find ML return forecasts lose most profitability
once microcaps are excluded and costs charged — and our Step 3 found a
209-characteristic equal-weight composite netting +0.24, inside its own null.
This is a better-specified use of the same inputs, so it should do better than
+0.24, but the prior that it clears +0.50 net is weak.

The secondary statistic is where I am genuinely unsure, and it is the one that
could overturn earlier work: every conclusion in Steps 1–7 rests on PCA-5
residuals.

## Known hazards

- **The characteristics are the same ones Step 3 showed do not pay.** A better
  model of the same dead inputs may not help. The counter-argument is structural:
  Step 3 used an unweighted composite as a *signal*; this uses characteristics to
  build *betas*, which is a different object.
- **OSAP ends 2024-12** while the panel runs to 2026-09. Twenty-one months cannot
  be tested and must not be extrapolated into.
- **Monthly horizon, ~96 test months.** SE on an annualised Sharpe is near 0.35.
- **Sign alignment.** Step 3 found OSAP's `Sign` field worth +0.20 of Sharpe, and
  it was chosen in-sample. Use **raw, unsigned** characteristics — a factor model
  estimates loadings and does not need the orientation.
- **Ninth test on overlapping data.** Not independent evidence.
