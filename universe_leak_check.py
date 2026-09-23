"""Regression test: is the liquidity screen point-in-time?

A universe selected on dollar volume measured *through* month m, and scored on
the return earned *during* month m, admits names because they spiked -- the rows
of the panel are chosen using the outcome.  No model-side test detects this:
a permutation null shuffles characteristics within the same contaminated
universe, and lagging the characteristics lags the features, not the rows.

The symptom is visible without any model.  Just hold the universe equal-weighted
and compare screen timings.  If lag 0 beats lag 1 by a wide margin, the screen
leaks.  Run this after any change to the panel build.

    $ python3 exploration/universe_leak_check.py
"""
import pandas as pd, numpy as np, sys

NTOP, LO, HI, EVAL = 1500, 200201, 202412, 201701

px = pd.read_parquet("cache/mp_close.parquet")
dv = pd.read_parquet("cache/mp_dv.parquet")
mret = px.resample("ME").last().pct_change(fill_method=None)
mdv  = dv.rolling(21, min_periods=10).mean().resample("ME").last()
ym = lambda i: i.year*100 + i.month
mret.index, mdv.index = ym(mret.index), ym(mdv.index)
months = sorted(m for m in mret.index if LO <= m <= HI)

def equal_weight_universe(lag):
    """annualised return of simply holding the screened universe"""
    out = []
    for k, m in enumerate(months):
        if k - lag < 0: continue
        sel = months[k-lag]
        if sel not in mdv.index: continue
        syms = list(mdv.loc[sel].dropna().nlargest(NTOP).index)
        r = mret.loc[m].reindex(syms).values.astype(float)
        r = np.clip(r[np.isfinite(r)], -0.95, 3.0)
        if len(r) > 300 and m >= EVAL: out.append(r.mean())
    o = np.array(out)
    return o.mean()*12, o.std()*np.sqrt(12), o.mean()/o.std()*np.sqrt(12)

print(f"Equal-weight universe return, {EVAL}-{HI}, by liquidity-screen timing")
print(f"{'screen':38s}{'ann.ret':>10s}{'vol':>9s}{'Sharpe':>9s}")
print("-" * 66)
res = {}
for lag, lab in ((0, "volume THROUGH month m  (LEAKS)"),
                 (1, "volume through m-1      (correct)"),
                 (2, "volume through m-2")):
    a, b, c = equal_weight_universe(lag); res[lag] = a
    print(f"{lab:38s}{a:+10.2%}{b:9.2%}{c:+9.2f}")

print(f"\nsize of the leak if the screen is mistimed: {res[0]-res[1]:+.2%}/yr")

# --- the actual regression test -------------------------------------------
# The table above is a property of the data and is always true.  The test is
# whether the BUILT panel used the correct timing: compare the equal-weight
# return of the panel's own rows against the two benchmarks.
import pickle, os
PANEL = "cache/char_R_fix.npy"
if not os.path.exists(PANEL):
    print(f"\n{PANEL} not found -- build the panel first; nothing to test.")
    sys.exit(0)
R = np.load(PANEL, allow_pickle=True)
IDX = pickle.load(open(PANEL.replace("char_R","char_idx").replace(".npy",".pkl"),"rb"))
ew = np.array([np.clip(np.asarray(r,dtype=float),-0.95,3.0).mean()
               for r,(m,_) in zip(R,IDX) if m >= EVAL])
built = ew.mean()*12
print(f"built panel ({PANEL}) equal-weight return: {built:+.2%}/yr")
midpoint = (res[0] + res[1]) / 2
if built > midpoint:
    print(f"FAIL: closer to the leaking screen ({res[0]:+.2%}) than to the "
          f"correct one ({res[1]:+.2%}).  Universe membership is conditioned "
          f"on the outcome; every downstream result is void.")
    sys.exit(1)
print(f"PASS: consistent with a point-in-time screen ({res[1]:+.2%}).")
