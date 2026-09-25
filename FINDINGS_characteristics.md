# Step 3 result: DEAD — the published catalogue does not pay out of sample

Pre-registered in [`PREREG_characteristics.md`](PREREG_characteristics.md) before
any characteristic was joined to any return. The pre-registration recorded a
negative as the base case.

## Prerequisite cleared first

`cache/crosswalk.csv` mapped permno→ticker for **1,438 live names and zero
delisted** — fingerprint-matched against a yfinance panel containing only
survivors. Using it would have reintroduced the bias the MarketParquet purchase
removed, in a test where survivorship inflates results.

Rebuilt against the survivorship-free panel. All five gates passed:

| gate | result | required |
|---|---|---|
| known anchors | 7/7 | all |
| median top-1 correlation | 1.0000 | ≥ 0.99 |
| **delisted coverage** | **44.8%** | ≥ 30% |
| held-out `Mom12m` (unused in matching) | 0.9998 | ≥ 0.95 |
| shuffled-pairing control | 0.2094 | ≤ 0.40 |

**1,443 → 7,638 symbols, nearly half of them dead companies.**

## Result

209 OSAP characteristics, lagged one month, cross-sectionally z-scored and
winsorised, **equal-weight composite with no fitting**, top 1,500 by dollar
volume, monthly rebalance, 2017-01 → 2024-12 (96 months).

**Corrected 2026-09-25.** The first version of this table had the two arms
swapped — see [the correction](#correction-the-two-arms-were-swapped) below.
The verdict does not change.

| arm | gross | net @2bp | breakeven | null mean | **null max** | verdict |
|---|---|---|---|---|---|---|
| **published directions** (sign-aligned) | +0.05 | **+0.04** | 8bp | −0.24 | **+0.25** | inside null |
| direction ignored (raw) | +0.25 | **+0.24** | 44bp | −0.22 | **+0.33** | inside null |

SE(Sharpe) ≈ 0.35 on 96 months, so even +0.24 is t ≈ 0.7 on its own terms.
Each arm is beaten by two of ten null draws that permute characteristics across
stocks within each month, destroying only the link between a firm and its own
subsequent return.

## The published directions did not help

Pointing each signal the way its paper says it works scored **+0.04**. Ignoring
direction altogether scored +0.24. Both are inside their nulls, so neither
number means much on its own — but the comparison is the opposite of what
the anomaly literature would predict. If these signals still worked, their
published orientation should be the one that pays.

The raw composite has no economic interpretation: it averages 209 signals whose
orientations were set arbitrarily by how each was first defined. Its +0.24 is
noise that happened to land above the signed version's.

## Interpretation

This is McLean & Pontiff (2016) taken to its limit. They documented ~58%
post-publication decay. Our window is almost entirely out-of-sample relative to
the original papers, and what survives collectively is not distinguishable from
zero.

A caveat worth stating: this tests the **equal-weight composite of the published
catalogue**, not every individual characteristic. Some almost certainly still
work; a few may work well. What fails is the naive claim that the documented
cross-section of expected returns, taken as a body, still pays. Fitting weights
over 209 anomalies on 96 months would have produced a beautiful backtest and
meant nothing, which is why the pre-registration forbade it.

## Where this leaves the exploration

| experiment | verdict |
|---|---|
| Step 1 — liquidity deciles | FALSIFIED — flat across all five bands |
| Step 2 — longer holding periods | dropped — ACF shows nothing beyond lag 10 |
| Step 3 — characteristics composite | **DEAD — inside its own null** |

Three tests, three negatives, and they are consistent rather than coincidental.
Residual reversal was payment for absorbing urgency; that service is no longer
needed. Published characteristics were payment for information processing;
publication removed the payment. **Both were documented sources of return, and
being documented is what ended them.**

Nothing here suggests the *method* is at fault — the same harness measured +1.01
net in 2009–2016 on reversal, and the crosswalk validates at r = 0.9998 on a
held-out signal. The measurements work. What they measure is gone.

Anything next should be something not in a public catalogue.

## Correction: the two arms were swapped

Found 2026-09-25 by a parallel analysis, then verified here by rerunning.

`cache/osap_all_raw.parquet` stores each signal already multiplied by its OSAP
sign, despite the name. Volatility signals whose sign is −1 are 99–100% negative
in the file, which raw volatility can never be. `char_test.py` then multiplied by
the sign again when asked to sign-align, which undid it — so the arm labelled
"sign-aligned" was raw, and the arm labelled "not sign-aligned" was signed.

Rerunning with the labels fixed reproduces both rows digit for digit, in swapped
places. The first version of this document concluded that sign alignment was
worth +0.20 of Sharpe and that *"what remains of these anomalies out of sample is
mostly the memory of which direction they used to work."* **Both statements were
wrong.** The published directions were worth −0.20, not +0.20, and the verdict
— dead, inside the null — holds either way.
