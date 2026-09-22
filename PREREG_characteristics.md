# Pre-registration: do firm characteristics pay at a monthly horizon, survivorship-free and net of costs?

Written before any characteristic is joined to any return. Nothing below is
revised after seeing results.

## Why this is a different bet from everything prior

Parts 1–2 and Step 1 tested **residual reversal**: payment for absorbing someone's
urgency. That is a service fee, and we established it no longer exists — AR(1)
collapsed 93% while dispersion held steady, flat across all five liquidity bands.

Firm characteristics at a monthly horizon are a different economic claim:
compensation for **information processing and risk transfer**, not for supplying
immediacy. Nothing established about liquidity provision carries over. It is also
the most data-mined area in finance, which shapes the hazards below.

## Prerequisite (gated): a survivorship-free crosswalk

`cache/crosswalk.csv` maps CRSP permno to ticker for **1,438 live names and zero
delisted** — it was fingerprint-matched against a yfinance panel containing only
survivors. Using it would reintroduce the bias the MarketParquet purchase
removed, in a test where survivorship inflates results.

It must be rebuilt against the MarketParquet panel, which carries ~10,000
delisted symbols, using the same method: recompute `MaxRet` per symbol-month and
match to OSAP on correlation.

**Gates, fixed now. If any fails, this experiment does not run.**

| gate | requirement |
|---|---|
| known anchors | AAPL, MSFT, IBM, XOM, GE, JNJ, KO all match their documented permnos |
| match quality | median top-1 correlation ≥ 0.99 |
| **held-out signal** (`Mom12m`, not used in matching) | median r ≥ 0.95 on matched pairs |
| **shuffled-pairing control** | median r ≤ 0.40 |
| **delisted coverage** | ≥ 30% of matched permnos map to symbols carrying `-DELISTED` |

The last gate is the point of the rebuild. A crosswalk that again matches only
survivors fails, regardless of how well it scores on the others.

## Construction

Universe: top 1,500 by trailing 21-day dollar volume from the survivorship-free
panel, re-selected monthly, point-in-time.

Characteristics: the OSAP set, joined on (permno, yyyymm), **lagged one month**
so only information available at formation is used. Each cross-sectionally
z-scored within the universe each month, winsorised at ±3, sign-aligned using
OSAP's published `Sign` field.

Signal: equal-weight composite of the z-scores. **Deliberately not fitted.** A
fitted combination on 200+ published anomalies would be curve-fitting a catalogue
assembled by curve-fitting; the equal-weight composite is the honest test of
whether the published set collectively still pays.

Portfolio: long-short, cross-sectionally demeaned, L1-normalised to gross 1,
rebalanced monthly, held one month.

## The statistic, fixed now

Net Sharpe of the monthly-rebalanced composite, **2017-01 → 2024-12** (OSAP ends
2024-12), after costs and borrow.

Costs: 2bp one-way on turnover, 35bp/yr borrow on short notional. Monthly
rebalancing means turnover is roughly one twentieth of the daily strategies, so
the cost hurdle is far lower — that is the structural reason this could work
where reversal cannot, and it must not be allowed to flatter the result. Report
breakeven cost alongside.

## The null, fixed now

Permute characteristic values **across stocks within each month**, preserving
every marginal distribution, the cross-sectional spread, and the universe — and
destroying only the link between a firm's characteristics and its own subsequent
return. Ten draws. Identical pipeline.

A generic shuffle would not do: the sign-flip null in Part 1 left `|residual|`
intact and passed a strategy that timed volatility. This null must break the
specific claim.

## Decision thresholds

**WORKS** — net Sharpe ≥ **+0.50**, above every null draw, and breakeven cost
≥ 3× the 2bp assumed.

**DEAD** — net Sharpe ≤ **+0.15**, or within the null distribution.

**AMBIGUOUS** — anything else. Reported as ambiguous. Not re-cut by universe
size, characteristic subset, horizon, or weighting until something clears.

## Known hazards, recorded in advance

- **Publication bias is the dominant risk.** OSAP is a catalogue of *published*
  anomalies. McLean & Pontiff (2016) document ~58% post-publication decay, and
  most were published on samples ending before 2017. Our window is almost
  entirely out-of-sample relative to the original papers, which is the point —
  but it also means a positive result is more surprising than a negative one, and
  a negative result is the base case.
- **A `Sign` field is itself in-sample.** OSAP aligns each signal so historical
  mean returns increase in it. That orientation was chosen on data including our
  test window's predecessors. Report results with and without sign alignment.
- **OSAP ends 2024-12** while the panel runs to 2026-09. The last 21 months
  cannot be tested. Do not extrapolate into them.
- **Monthly horizon means few independent observations.** 2017–2024 is ~96
  non-overlapping months; standard error on an annualised Sharpe is near 0.35.
  State it beside every number.
- **Characteristic staleness.** Accounting data arrives with a lag that varies by
  firm. The one-month lag is a convention, not a guarantee; any result that
  disappears at a two-month lag was look-ahead.
