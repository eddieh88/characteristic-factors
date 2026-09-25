# Do characteristic factor models pay out of sample?

Two pre-registered tests of the "characteristics predict returns" literature on
a **survivorship-free US equity panel**: the published characteristic catalogue
(OSAP, 209 signals), and the conditional factor models built on top of it —
**IPCA** and the **conditional autoencoder** of Gu, Kelly & Xiu.

Both were pre-registered before any characteristic was joined to any return.

## The answer

**Step 3 — the published catalogue: DEAD.** Out of sample, it does not pay.

**Step 9 — IPCA / autoencoder: AMBIGUOUS by the pre-registered rule, dead in
substance.**

| statistic | pre-registered threshold | measured |
|---|---|---|
| net Sharpe 2017–2024 | WORKS ≥ +0.50 | **+0.31** (5 seeds, sd 0.05) |
| same, returns capped | — | **−0.06** |
| market-hedged Sharpe | — | **+0.28** |
| rank IC | — | +0.0061, t = 0.60, hit rate 48.2% |
| deciles monotonic | — | **no** |
| permutation null (10 refits) | DEAD if inside | real +0.35 vs max draw +0.23 — **outside** |

It fails the primary threshold but clears the null, so the pre-registered label
is **AMBIGUOUS** — recorded as such rather than relabelled to match the
conclusion.

Substantively it is dead, and the null is why that has to be said out loud.
Permuting characteristics destroys the firm→return link, so a permuted model
cannot tilt toward anything systematic — the draws are random dollar-neutral
books bleeding borrow cost. The real model *did* find a stable non-random tilt.
**That tilt is beta +0.33**, and it paid because 2017–2024 was a bull market.
Gross profitability alone gives **+0.58 hedged at a quarter the turnover.**

## The bug that matters more than the result

A first round of results was invalidated by a **look-ahead in the universe
screen**: stocks entered the sample by dollar volume measured *through* the
month whose return was being predicted, so names got in **because they had
already spiked.**

It was worth **19 percentage points a year.** Nine model-side tests passed
while it was live. What exposed it was not a statistic — it was looking at the
holdings and seeing nano-caps inside a "top-1500 by liquidity" universe.

[`universe_leak_check.py`](universe_leak_check.py) is the regression test that
now guards it. **A point-in-time universe is data, not methodology.**

## Where to read the findings

| | |
|---|---|
| **[FINDINGS_autoencoder.md](FINDINGS_autoencoder.md)** | **Start here.** Step 9: the verdict, the look-ahead bug, and why the null needed explaining. |
| [FINDINGS_characteristics.md](FINDINGS_characteristics.md) | Step 3: the published catalogue, and the crosswalk that had to be rebuilt first. |
| [PREREG_autoencoder.md](PREREG_autoencoder.md), [PREREG_characteristics.md](PREREG_characteristics.md) | Written before the runs. |
| [LITERATURE.md](LITERATURE.md) | What the academic record already says. |
| [diagnostics/](diagnostics/) | One-off scripts behind quoted numbers — delisting audit, market beta, traditional-factor benchmark. |

## Data hygiene

The panel is survivorship-free, which took real work. `cache/crosswalk.csv`
originally mapped permno→ticker for **1,438 live names and zero delisted** —
fingerprint-matched against a panel containing only survivors. Using it would
have reintroduced exactly the bias the data purchase removed, in a test where
survivorship inflates results. [`crosswalk_sf.py`](crosswalk_sf.py) rebuilds it
against the survivorship-free panel; all five gates pass.

## Running it

```bash
pip install -r requirements.txt
python3 data/mp_fetch.py          # MarketParquet daily archive
python3 data/mp_panel.py          # survivorship-free price/return panel
python3 osap_fetch.py             # OSAP characteristics
python3 build_char_panel.py       # join -> monthly panel
python3 autoencoder_test.py       # Step 9
python3 universe_leak_check.py    # the regression test; run it first if you touch the screen
```

Scripts read `cache/` relative to the repo root; if you already hold the
archive elsewhere, `ln -s /path/to/your/cache cache` (gitignored, never
committed) rather than re-downloading it.

Expects a MarketParquet key at `~/.market_parquest/api_key.txt`, **never in the
repo** — the pre-commit hook in `hooks/` scans for it. Enable with
`git config core.hooksPath hooks`. No market data is redistributed here;
`cache/` is gitignored.

## Related repositories

This work started as one repository and split into three when the questions
stopped being the same question.

| repository | question | answer |
|---|---|---|
| [**statarb-replication**](https://github.com/eddieh88/statarb-replication) | Does *Deep Learning Statistical Arbitrage* replicate, and does it still work? | Replicates; does not survive past 2016 |
| [**characteristic-factors**](https://github.com/eddieh88/characteristic-factors) | Do IPCA and the conditional autoencoder pay out of sample? | Ambiguous by the pre-registered rule, dead in substance |
| [**intraday-patterns**](https://github.com/eddieh88/intraday-patterns) | Do the chart setups taught in trading education work? | No entry beats a naive momentum rule at the same bar |

The shared thread is the error log: each repository records what went wrong and
what caught it, because in this kind of work that is the part that transfers.
