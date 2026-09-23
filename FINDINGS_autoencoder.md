# Finding 9 — characteristic factor models (IPCA / autoencoder): **AMBIGUOUS** by the pre-registered rule, **dead** in substance

Pre-registered in [PREREG_autoencoder.md](PREREG_autoencoder.md). This document
records the result *and* a look-ahead bug that invalidated a first round of
results, because the bug is the more useful finding.

## Verdict

| statistic | pre-registered threshold | measured (corrected panel) |
|---|---|---|
| net Sharpe 2017-2024 | WORKS ≥ +0.50 | **+0.31** (5 seeds, sd 0.05) |
| same, returns capped at ±(95%,300%) | — | **−0.06** |
| market-hedged Sharpe | — | **+0.00** |
| rank IC | — | **+0.0061**, t = 0.60, hit rate 48.2% |
| deciles monotonic | — | **no** |
| permutation null (10 refits) | DEAD if inside | real +0.35 vs mean −0.47, **max +0.23** — above every draw |

Fails the primary threshold (+0.31 against +0.50) but clears the null and sits
above the +0.15 DEAD line, so the **pre-registered label is AMBIGUOUS**. It is
recorded as such rather than relabelled to match the conclusion.

Substantively it is dead, and the null is why that needs saying out loud.
Permuting characteristics destroys the firm→return link, so a permuted model
cannot tilt toward anything systematic — the draws are random dollar-neutral
books bleeding borrow cost, hence the −0.47 mean. The real model did find a
stable non-random tilt. **That tilt is beta +0.33**, and it paid because
2017-2024 was a bull market.

So the null confirms the model learned something real; it cannot tell alpha from
beta. This project's own principle applies — *a null must break the specific
claim*. Mine broke "characteristics predict returns" when the claim requiring a
null was "characteristics predict returns **beyond market exposure**." A
beta-neutralised permutation null is the correct design, and the market-hedged
Sharpe of **+0.00** is what it would have reported.

## What the strategy actually was

Monthly, ~1,400 US names, 209 lagged OSAP characteristics → 5 factor loadings via
a 6,885-parameter network → predicted return = loadings × trailing-60-month mean
factor return → long/short, monthly rebalance. A cross-sectional *ranking* model:
it never forecasts the market, only which names beat which others within a month.

## The look-ahead bug

A first round produced **+1.95 net Sharpe**, which passed a 10-draw permutation
null (max +0.11), a K sweep (+1.75 to +2.46), 7/7 positive years, a
publication-vintage test, and a 6-month accounting-lag test. All of it was void.

```python
mdv = dv.rolling(21).mean().resample("ME").last()
liq = mdv.loc[m].dropna().nlargest(1500)   # volume measured THROUGH month m
nxt = mret.loc[m]                           # return earned DURING month m
```

Universe membership required top-1500 dollar volume measured *over the month
whose return is being predicted*. A nano-cap that ran +300% in month *m* traded
enormous volume in month *m*, and therefore entered the universe **because it
spiked**. The rows of the dataset were selected using the outcome.

Measured cost of the bug — just *holding* the universe equal-weighted:

| liquidity screen | ann. return | Sharpe |
|---|---|---|
| volume through month *m* (as built) | **+27.5%** | **+1.26** |
| volume through *m−1* (correct) | +8.3% | +0.40 |
| volume through *m−2* | +8.6% | +0.42 |

19 points a year. Most of the +1.95 existed before any model did.

### Why every test missed it

| test | why it was blind |
|---|---|
| permutation null | shuffles characteristics *within the same contaminated universe*; both arms inherit the bias |
| 6-month accounting lag | lags the **features**; the leak is in which **rows exist** |
| K sweep, IPCA benchmark, publication vintage | all model-side; the defect is upstream of any model |

Every test interrogated the mapping from X to y. None asked whether the rows in
the table belonged there. **A point-in-time universe is part of the data, not
part of the methodology** — the pre-registration specified "point-in-time" and
the code was not, and no model-side discipline can catch that.

The tell was available early and was misread: a gross Sharpe of **3.78** on
monthly rebalancing is not a good result, it is an error message. Two further
signals were rationalised at the time — an out-of-sample R² of −10 billion in one
window (blamed on cosmetics), and seed-to-seed Sharpe swings of ±0.6 (blamed on
single-seed training). Both were the artifact. After the fix, seed sd fell to
0.05: instability was a *symptom*, not noise.

Fixed in `build_char_panel.py`; the corrected panel is `char_{R,Z}_fix.npy`.

## Corrected results

Five seeds, point-in-time universe:

| seed | raw | capped |
|---|---|---|
| 0 | +0.35 | −0.09 |
| 1 | +0.23 | −0.09 |
| 2 | +0.37 | +0.34 |
| 3 | +0.30 | −0.26 |
| 4 | +0.31 | −0.22 |
| **mean** | **+0.31** | **−0.06** |

Capping removes unadjusted reverse-split prints (CHK +100x, WLL +20x, SUNE
+2.9 billion %). That the capped arm is negative means what remains of +0.31
leans on prints that do not correspond to tradeable price moves.

### It is not a zero-beta trade, and the beta is the return

Dollar-neutral by construction (Σw = 2.6e−09) but **beta +0.33** to the
equal-weight universe — long leg 0.84, short leg −0.51. Market-hedged Sharpe
**+0.00** against raw +0.36. The residual was market premium in a rising market.

### The ordering carries nothing

Rank IC +0.0061 (t = 0.60) against the 0.02–0.03 an ordinary real signal shows;
hit rate 48.2%. Deciles are flat from D1 to D9 (+7% to +11%/yr) with everything
in D10 (+24.97%) — and D1, the short leg, returns +11.8%/yr, which is why
shorting it bled 5.2%/yr. Not monotonic.

This rules out the charitable reading. The construction *was* bad — cardinal
weights let a `max|rhat|` of 2.7e5 size positions — but fixing it would not help,
because there is no ordering underneath to rebuild around. D10 is high-beta
speculative names in a bull market, which is the same fact as the +0.33 beta.

## Panel defects still outstanding

- Unadjusted reverse splits (vendor adjusts forward splits; missed some reverse).
  Needs repair, not blanket capping — a cap also truncates real squeezes.
- Delisting returns dropped rather than recorded: 0.46% of name-months, median
  last price $37, so mostly M&A. Real but small.
- Compustat restatement bias is untestable without point-in-time Compustat. The
  6-month lag test catches *filing delay*, not restatement.

## Status against the project

Ninth pre-registered exploration test; eighth negative. Nothing here reopens
Steps 1–7 — the pre-registered secondary statistic (residual AR(1)) was computed
on contaminated returns and is **not reported**; it needs re-running on the
corrected panel before any claim is made either way.
