"""Survivorship-free permno<->ticker crosswalk.

The original crosswalk was fingerprint-matched against a yfinance panel that
contains only survivors: 1,438 live tickers, zero delisted.  This rebuilds it
against the MarketParquet panel (~10,000 delisted symbols) by the same method --
recompute MaxRet per symbol-month, match to OSAP on correlation -- so that dead
firms are matchable.

Gates are in exploration/PREREG_characteristics.md and are checked at the end.
"""
import numpy as np, pandas as pd, polars as pl

WIN = [(200601,200912),(201001,201312),(201401,201712),(201801,202112),(202201,202412)]
MIN_OVERLAP, MIN_CORR, MIN_MARGIN = 30, 0.98, 0.05

px = pd.read_parquet("cache/mp_close.parquet")
r = px.pct_change(fill_method=None).mask(lambda x: x.abs() > 1.0)
ym = r.index.year*100 + r.index.month
mx, cnt = r.groupby(ym).max(), r.groupby(ym).count()
ours = mx.mask(cnt < 15)
print(f"panel MaxRet {ours.shape}  {ours.index.min()} .. {ours.index.max()}")

o = pl.read_parquet("cache/osap_fingerprint.parquet").select("permno","yyyymm","MaxRet").drop_nulls()
theirs = o.to_pandas().pivot(index="yyyymm", columns="permno", values="MaxRet")
print(f"OSAP MaxRet  {theirs.shape}")

def corr_block(A, B):
    def z(df):
        X = df.to_numpy(float)
        s = np.nanstd(X, 0)
        Z = (X - np.nanmean(X, 0)) / np.where(s > 0, s, np.nan)
        return np.nan_to_num(Z), (~np.isnan(X)).astype(float)
    ZA, mA = z(A); ZB, mB = z(B)
    den = mA.T @ mB
    C = (ZA.T @ ZB) / np.maximum(den, 1)
    C[den < MIN_OVERLAP] = np.nan
    return C

hits = {}
for lo, hi in WIN:
    W = [m for m in ours.index if lo <= m <= hi and m in theirs.index]
    if len(W) < MIN_OVERLAP: continue
    A = ours.loc[W]; B = theirs.loc[W]
    A = A.loc[:, A.notna().sum() >= MIN_OVERLAP]
    B = B.loc[:, B.notna().sum() >= MIN_OVERLAP]
    C = corr_block(A, B)
    idx = np.argsort(-np.nan_to_num(C, nan=-9), axis=1)
    c1 = C[np.arange(len(idx)), idx[:,0]]; c2 = C[np.arange(len(idx)), idx[:,1]]
    n = 0
    for i, tk in enumerate(A.columns):
        if np.isfinite(c1[i]) and c1[i] >= MIN_CORR and (c1[i]-c2[i]) >= MIN_MARGIN:
            hits.setdefault(tk, []).append((c1[i], c1[i]-c2[i], int(B.columns[idx[i,0]]))); n += 1
    print(f"  {lo}-{hi}: {A.shape[1]:5d} symbols x {B.shape[1]:5d} permnos -> {n:5d} matched", flush=True)

rows = []
for tk, hs in hits.items():
    best = max(hs)
    rows.append(dict(symbol=tk, permno=best[2], corr=round(best[0],5),
                     margin=round(best[1],4), n_windows=len(hs),
                     delisted=bool(tk.upper().endswith("-DELISTED"))))
cw = pd.DataFrame(rows).sort_values("symbol")
cw.to_csv("cache/crosswalk_sf.csv", index=False)
print(f"\nwrote cache/crosswalk_sf.csv: {len(cw)} matches")
print(f"  median corr {cw['corr'].median():.4f}   min {cw['corr'].min():.4f}")
print(f"  delisted share: {cw.delisted.mean():.1%}  ({cw.delisted.sum()} of {len(cw)})")
ANCH = {"AAPL":14593,"MSFT":10107,"IBM":12490,"XOM":11850,"GE":12060,"JNJ":22111,"KO":11308}
ok = 0
for t, p in ANCH.items():
    m = cw[cw.symbol.str.upper()==t]
    got = int(m.permno.iloc[0]) if len(m) else None
    ok += (got == p)
    print(f"  {t:5s} -> {got} (expect {p}) {'OK' if got==p else 'MISS'}")
print(f"\nanchors {ok}/7")
