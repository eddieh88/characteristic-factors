"""How much of +1.95 is silently-deleted delisting returns?

build_char_panel.py drops rows whose next-month return is NaN.  A name that
delists mid-month has no next price, so its terminal return -- often the whole
loss -- never enters the panel.  Quantify: how many name-months are dropped this
way, and what happens if the delisting month is charged a realistic return.
"""
import pandas as pd, numpy as np, pickle
px = pd.read_parquet("cache/mp_close.parquet")
dv = pd.read_parquet("cache/mp_dv.parquet")
mpx = px.resample("ME").last(); mret = mpx.pct_change(fill_method=None)
mdv = dv.rolling(21, min_periods=10).mean().resample("ME").last()
ym = lambda i: i.year*100 + i.month
mret.index, mdv.index, mpx.index = ym(mret.index), ym(mdv.index), ym(mpx.index)
last_month = {c: (px[c].last_valid_index()) for c in px.columns}
last_ym = {c: (d.year*100+d.month if d is not None else None) for c,d in last_month.items()}
PANEL_END = 202609

months = sorted(m for m in mret.index if 200201 <= m <= 202412)
tot_in, tot_drop, drop_delist = 0, 0, 0
lastpx_drop = []
for k,m in enumerate(months):
    if k==0 or m not in mdv.index: continue
    liq = mdv.loc[m].dropna().nlargest(1500)
    syms = list(liq.index)
    nxt = mret.loc[m].reindex(syms).values
    ok = np.isfinite(nxt)
    tot_in += ok.sum(); tot_drop += (~ok).sum()
    for s in np.array(syms)[~ok]:
        lm = last_ym.get(s)
        if lm is not None and lm < PANEL_END - 1 and lm <= m:
            drop_delist += 1
            p = mpx[s].reindex([months[k-1]]).values[0] if months[k-1] in mpx.index else np.nan
            lastpx_drop.append(p)

print(f"name-months entering panel : {tot_in:,}")
print(f"name-months dropped (NaN)  : {tot_drop:,}  ({tot_drop/(tot_in+tot_drop):.2%})")
print(f"  of which the series had already ended (delisting): {drop_delist:,}")
lp = np.array([p for p in lastpx_drop if np.isfinite(p)])
if len(lp):
    print(f"  last price of those names: median ${np.median(lp):.2f}, "
          f"share under $5: {(lp<5).mean():.1%}")
