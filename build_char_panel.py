"""Monthly panel: returns + 209 OSAP characteristics, survivorship-free.

Joined through crosswalk_sf.csv (7,638 symbols, 44.8% delisted).  Characteristics
are lagged one month, cross-sectionally z-scored and winsorised at +-3, missing
set to 0.  Raw/unsigned -- a factor model estimates loadings and does not need
OSAP's Sign field, which Step 3 found was worth +0.20 of Sharpe and chosen
in-sample.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
LO, HI, NTOP = 200201, 202412, 1500

px = pd.read_parquet("cache/mp_close.parquet")
dv = pd.read_parquet("cache/mp_dv.parquet")
mpx = px.resample("ME").last()
mret = mpx.pct_change(fill_method=None)
mdv = dv.rolling(21, min_periods=10).mean().resample("ME").last()
ym = lambda i: i.year*100 + i.month
mret.index, mdv.index = ym(mret.index), ym(mdv.index)

cw = pd.read_csv("cache/crosswalk_sf.csv")
cw = cw[cw["corr"] > 0.99]
s2p = dict(zip(cw.symbol.str.upper(), cw.permno))
p2s = {}
for s, p in s2p.items():
    p2s.setdefault(p, s)                 # first symbol per permno
print(f"crosswalk {len(s2p)} symbols -> {len(p2s)} permnos, {cw.delisted.mean():.1%} delisted")

ch = pd.read_parquet("cache/osap_all_raw.parquet",
                     filters=[("yyyymm", ">=", LO), ("yyyymm", "<=", HI)])
cols = [c for c in ch.columns if c not in ("permno", "yyyymm")]
print(f"characteristics {len(cols)}, rows {len(ch):,}")

months = sorted(m for m in mret.index if LO <= m <= HI)
rows_r, rows_z, rows_i = [], [], []
for k, m in enumerate(months):
    if m not in mdv.index: continue
    # LOOK-AHEAD FIX: the liquidity screen must use volume known before month m
    # begins.  Screening on volume *through* m selects names because they spiked
    # during m -- conditioning universe membership on the outcome.  Measured cost
    # of the bug: equal-weight universe +27.5%/yr (Sharpe 1.26) vs +8.3% (0.40).
    if k == 0: continue                      # months[-1] would wrap to the last month
    msel = months[k-1]
    if msel not in mdv.index: continue
    liq = mdv.loc[msel].dropna().nlargest(NTOP)
    syms = [s for s in liq.index if s.upper() in s2p]
    if len(syms) < 300: continue
    pn = np.array([s2p[s.upper()] for s in syms])
    prev = months[k-1] if k else None
    if prev is None: continue
    sub = ch[ch.yyyymm == prev].set_index("permno")      # LAGGED one month
    X = sub.reindex(pn)[cols].astype("float32").values
    nxt = mret.loc[m].reindex(syms).values.astype("float32")
    ok = np.isfinite(nxt)
    if ok.sum() < 300: continue
    X, nxt, pn2, syms2 = X[ok], nxt[ok], pn[ok], list(np.array(syms)[ok])
    mu = np.nanmean(X, 0); sd = np.nanstd(X, 0)
    Z = np.clip((X - mu)/np.where(sd > 0, sd, 1.0), -3, 3)
    Z = np.nan_to_num(Z, nan=0.0)
    rows_r.append(nxt); rows_z.append(Z.astype("float32"))
    rows_i.append((m, syms2))
    if k % 48 == 0: print(f"  {m}  n={len(nxt)}", flush=True)

np.save("cache/char_R.npy", np.array(rows_r, dtype=object), allow_pickle=True)
np.save("cache/char_Z.npy", np.array(rows_z, dtype=object), allow_pickle=True)
import pickle; pickle.dump(rows_i, open("cache/char_idx.pkl","wb"))
print(f"\n{len(rows_r)} months, {months[0]}..{months[-1]}, "
      f"median n {int(np.median([len(r) for r in rows_r]))}, {len(cols)} chars")
