# Do company fundamentals predict stock returns?

A large academic literature says that a company's **characteristics** — its
size, valuation, profitability, recent momentum and a few hundred more — predict
which stocks will outperform. The most advanced version, the **conditional
autoencoder** of Gu, Kelly & Xiu, uses a neural network to learn how those
characteristics drive returns, and reports strong results.

This repository tests that on clean, modern US data (2017–2024).

## Why this came next

It follows on from [statarb-replication](https://github.com/eddieh88/statarb-replication),
which found that trading on **price moves alone** had stopped working after
2016. The literature's answer was that price-only signals are the crude version:
models that also see what kind of company each stock is should do better. So we
tested exactly that.

## The short version

- **The published signals, combined: no better than luck.** 209 of them together
  net a Sharpe of **+0.24**, and two of ten random shuffles scored higher.
- **The neural network: falls short, and what it found was market exposure.**
  It netted **+0.31** against a pre-set bar of +0.50. It does beat random
  shuffles, but the thing it learned is to lean long the market, which paid
  because 2017–2024 was a bull market.
- **One old-fashioned factor beats it.** Buying profitable companies (gross
  profitability) alone scores **+0.58** with market exposure removed, trading a
  quarter as much.
- **Our first result was +1.95, and most of it was a bug.** The bug is more
  instructive than the result, so it gets its own section below.

## The story

### 1. First, the data had to be fixed

The characteristics come from [Open Source Asset Pricing](https://www.openassetpricing.com/),
keyed by an ID system (CRSP permno) our price data doesn't use. The mapping we
started with covered **1,438 companies and zero failed ones** — it had been
built from stocks that still trade today. Using it would have quietly removed
every company that went bust, which flatters any strategy.

Rebuilt against the full dataset, including delisted stocks: **1,443 → 7,638
companies, nearly half of them dead.**

### 2. The published signals, taken together

The simplest possible test: take all 209 published characteristics, point each
in the direction its paper says works, average them, and trade the result
monthly. No fitting, nothing to overfit.

| | Sharpe after costs |
|---|---|
| all 209 signals combined | **+0.24** |
| best of 10 random shuffles | +0.33 |

To build the shuffles, we scrambled which company each set of characteristics
belongs to and ran the same strategy. The real signals should beat that easily.
They don't — **two of ten shuffles did better.**

### 3. The neural network

Two models that learn from characteristics rather than just averaging them:
**IPCA** (linear) and the **conditional autoencoder** (a neural network). Both
the models and the bar they had to clear were written down before any run.

| | result | bar |
|---|---|---|
| Sharpe after costs | **+0.31** | needed +0.50 |
| beats random shuffles? | yes — +0.35 vs best shuffle +0.23 | must beat them |
| with bad data prints capped | **−0.06** | — |
| does its ranking of stocks match what happens? | barely — right in 48.2% of months | a coin flip is 50% |

By the rules set in advance, that is **ambiguous**: it misses the bar but beats
the shuffles. We recorded it that way rather than relabel it.

In substance it's dead, and the reason is in the table. It beat the shuffles
because it learned something consistent: **lean long the market** (beta
+0.33). In a bull market that pays. And with a handful of corrupted price
records capped — split adjustments the data vendor missed, one of them a
+2.9 billion percent "return" — the result goes negative.

Meanwhile, one textbook factor did better on its own:

| | Sharpe, market exposure removed | trading |
|---|---|---|
| gross profitability alone | **+0.58** | ¼ as much |
| the neural network | +0.28 | — |

## The bug that mattered more than the result

The first run scored **+1.95**. It was wrong.

The strategy traded the 1,500 most-traded stocks each month. But "most traded"
was measured *during the same month whose return we were predicting*. A tiny
stock that shot up 300% traded enormous volume that month — so it **entered the
sample because it had already spiked**.

| | yearly return, just holding the sample |
|---|---|
| sample picked using the same month (the bug) | **+27.5%** |
| sample picked using the month before (correct) | +8.3% |

**19 percentage points a year**, before any model did anything. The shuffle
test didn't catch it, because the shuffles were drawn from the same poisoned
sample. What caught it was looking at the actual holdings and seeing tiny
companies inside a list that was supposed to be the most liquid 1,500.

[`universe_leak_check.py`](universe_leak_check.py) now guards against it.

## Where to go next

| | |
|---|---|
| **[FINDINGS_autoencoder.md](FINDINGS_autoencoder.md)** | The neural network test in full: every table, and the bug |
| [FINDINGS_characteristics.md](FINDINGS_characteristics.md) | The 209-signal test and the data rebuild |
| [PREREG_autoencoder.md](PREREG_autoencoder.md), [PREREG_characteristics.md](PREREG_characteristics.md) | The rules, written before the runs |
| [LITERATURE.md](LITERATURE.md) | What the academic papers claim |
| [diagnostics/](diagnostics/) | Scripts behind individual numbers: market exposure, traditional factors, delisting audit |

## Words used here

| term | meaning |
|---|---|
| **Sharpe ratio** | Return divided by volatility, annualised. Above 1 is good for a real strategy. |
| **characteristic** | A measurable fact about a company: size, book-to-market, profitability, past returns, and so on. |
| **random shuffle / null** | The same test with characteristics scrambled across companies. A real result must beat it. |
| **beta** | How much a portfolio moves with the whole market. A true arbitrage should have beta near zero. |
| **delisted** | A company that no longer trades — acquired, or failed. Leaving these out flatters every backtest. |

## Running it

```bash
pip install -r requirements.txt
python3 data/mp_fetch.py && python3 data/mp_panel.py   # clean price panel
python3 osap_fetch.py && python3 build_char_panel.py   # characteristics, joined
python3 universe_leak_check.py                         # run this before trusting anything
python3 autoencoder_test.py                            # the main test
```

A MarketParquet key is expected at `~/.market_parquest/api_key.txt`, never in
the repo; the pre-commit hook in `hooks/` scans for it. If you already hold the
data archive, `ln -s /path/to/cache cache` instead of re-downloading.

## Related repositories

| repository | question | answer |
|---|---|---|
| [**statarb-replication**](https://github.com/eddieh88/statarb-replication) | Does deep-learning stat arb replicate, and does it still work? | Replicates. Does not survive past 2016. |
| **this one** | If prices alone stopped working, do models built on company fundamentals do better? | No. What they found was market exposure. |
| [**intraday-patterns**](https://github.com/eddieh88/intraday-patterns) | Do the chart setups taught in trading education work? | No. Every edge they have is plain momentum. |
