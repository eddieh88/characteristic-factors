# Finding 9 — characteristic factor models (IPCA / autoencoder): **AMBIGUOUS** by the pre-registered rule, **dead** in substance

Pre-registered in [PREREG_autoencoder.md](PREREG_autoencoder.md). This document
records the result *and* a look-ahead bug that invalidated a first round of
results, because the bug is the more useful finding.

## Verdict

| statistic | pre-registered threshold | measured (corrected panel) |
|---|---|---|
| net Sharpe 2017-2024 | WORKS ≥ +0.50 | **+0.31** (5 seeds, sd 0.05) |
| same, returns capped at ±(95%,300%) | — | **−0.06** |
| market-hedged Sharpe | — | **+0.28** (corrected — see below) |
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

### It is not a zero-beta trade

Dollar-neutral by construction (Σw = 2.6e−09) but **beta +0.33** to the
equal-weight universe — long leg 0.84, short leg −0.51.

**Correction.** This section first reported a market-hedged Sharpe of **+0.00**
and concluded the entire return was market premium. That was a bug. The hedge
was computed as `res = y − (a + b·x)` and the Sharpe taken of `res` — but OLS
residuals have zero mean by construction, so that statistic is 0.00 for any
input. Three different series printed ±0.00, the same tell that had caught an
AR(1) bug an hour earlier, and it was missed.

Removing only the market component (`net − β·mkt`, keeping the intercept) gives
**hedged Sharpe +0.28** at the pre-registered K=5. The beta is real; the claim
that beta was the *whole* return was not.

### Tweaking: capacity helps a little, the objective does not

| K | 1 | 3 | 5 | 8 | 15 |
|---|---|---|---|---|---|
| hedged Sharpe | +0.18 | −0.05 | **+0.28** | +0.34 | +0.36 |

| objective | hedged |
|---|---|
| reconstruction MSE (as built) | **+0.28** |
| cross-sectional IC | +0.04 |
| portfolio Sharpe (the DLSA move) | +0.13 |

The reconstruction loss finds factors explaining the *variance* of returns while
prediction needs factors carrying non-zero *mean* premia, so the objective looks
misaligned — and the prediction that IC or Sharpe objectives would beat it was
**wrong**. Both did worse. Reconstruction fits 60 × ~1,400 residuals; the Sharpe
objective optimises one scalar over 60 months. At this sample size the training
signal outweighs the alignment argument.

K=15 roughly doubles hedged Sharpe over K=5. K was pre-registered at 5 so that
this choice could not be made after seeing results; **the headline stays at K=5**.

### The ordering carries nothing

Rank IC +0.0061 (t = 0.60) against the 0.02–0.03 an ordinary real signal shows;
hit rate 48.2%. Deciles are flat from D1 to D9 (+7% to +11%/yr) with everything
in D10 (+24.97%) — and D1, the short leg, returns +11.8%/yr, which is why
shorting it bled 5.2%/yr. Not monotonic.

The construction *was* bad — cardinal weights let a `max|rhat|` of 2.7e5 size
positions — and the flat D1-D9 says most of the cross-section is noise, with
the spread coming from D10 alone.

### Against traditional factors

Same panel, window, costs and construction; classic characteristics used directly
as the score (signed via OSAP's `Sign` field, an in-sample choice worth +0.20 in
Step 3 — so the traditional arm is flattered):

| | net Sh | beta | hedged | turn |
|---|---|---|---|---|
| GP (gross profitability) alone | +0.58 | −0.01 | **+0.58** | 0.24 |
| autoencoder, K=15 | +0.46 | +0.29 | +0.36 | — |
| equal-weight composite of 9 classics | +0.47 | +0.33 | +0.22 | 0.49 |
| autoencoder, K=5 (pre-registered) | +0.35 | +0.33 | +0.28 | 0.87 |
| equal-weight all 209 (signed) | +0.34 | +0.18 | +0.13 | 0.63 |
| equal-weight universe, long only | +0.52 | — | — | 0 |

The autoencoder beats both naive composites on a hedged basis and loses to the
single best characteristic, at ~3.5x its turnover. GP's +0.58 is the best of nine
tried against an SE of ~0.38, so it will not survive a multiple-testing
correction either and is not tradeable on this evidence.

Value (−0.15), investment (−0.48) and momentum (−0.04) were all negative in
2017-2024. This window was hostile to factors generally, which is the
Avramov-Cheng-Metzker result reproduced: the sophisticated model did not beat the
simple ones by enough to pay for itself.

## Panel defects still outstanding

- Unadjusted reverse splits (vendor adjusts forward splits; missed some reverse).
  Needs repair, not blanket capping — a cap also truncates real squeezes.
- Delisting returns dropped rather than recorded: 0.46% of name-months, median
  last price $37, so mostly M&A. Real but small.
- Compustat restatement bias is untestable without point-in-time Compustat. The
  6-month lag test catches *filing delay*, not restatement.

## Secondary statistic: residual AR(1) — Steps 1–7 do not reopen

Pre-registered condition: *if characteristic-driven betas produce residuals that
mean-revert where PCA-5's do not, that would reopen Steps 1–7.* Corrected panel,
returns capped, pooled across stocks:

| model | AR(1) | pairs | t |
|---|---|---|---|
| cross-sectional demean only | −0.0099 | 265,743 | −5.12 |
| IPCA | −0.0135 | 265,743 | −6.95 |
| autoencoder | −0.0091 | 265,743 | −4.67 |

The condition is not met. IPCA and the autoencoder sit either side of plain
demeaning; no model creates reversion the others lack, and the differences are
third-decimal. All three are significant (|t| ≈ 5–7 on a quarter-million pairs)
and economically trivial — roughly 1% of a month's residual reverses next month.
Significance is cheap at this n and says nothing about tradeability.

Consistent with the project's daily figures (−0.0285 raw, −0.0018 residual), so
nothing here contradicts the mechanism work in Steps 1–7.

Note: a first attempt at this statistic ran on contaminated returns and returned
+0.0000 for all three arms with a standard deviation of 56,171 — that run
surfaced the reverse-split prints and is not a result.

## Status against the project

Ninth pre-registered exploration test. Primary statistic measured, null run,
secondary statistic reported: the pre-registration is closed. Formally
AMBIGUOUS, substantively not tradeable — hedged Sharpe +0.28 against gross
profitability's +0.58 at a quarter the turnover, with rank IC +0.006 and
non-monotonic deciles.

The transferable findings are the two design failures, not the verdict:
a point-in-time universe is **data, not methodology**, and a null must break
**the specific claim** — permuting characteristics tested "do characteristics
predict returns" when the claim needing a null was "beyond market exposure."

Three bugs were found in this step (universe leak, hedged-Sharpe intercept,
AR(1) on contaminated returns). Two were caught by the same tell: a statistic
that came back implausibly round or identical across series. **An implausibly
clean number should be investigated before it is reported, not after.**
