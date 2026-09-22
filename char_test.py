"""Step 3, exactly as pre-registered in PREREG_characteristics.md.

Universe   top 1500 by trailing 21d dollar volume, re-selected monthly, PIT
Chars      full OSAP set, joined on (permno, yyyymm) via the SURVIVORSHIP-FREE
           crosswalk, lagged one month, cross-sectionally z-scored, winsorised +-3
Signal     equal-weight composite -- deliberately unfitted
Portfolio  long/short, demeaned, L1-normalised to gross 1, monthly rebalance
Statistic  net Sharpe 2017-01 .. 2024-12 after 2bp cost and 35bp/yr borrow
Null       permute characteristics ACROSS STOCKS within each month, 10 draws
Thresholds WORKS   >= +0.50, above every null draw, breakeven >= 6bp
           DEAD    <= +0.15, or inside the null distribution
           else    AMBIGUOUS
"""
import sys, json
import numpy as np, pandas as pd

LO, HI, NTOP, COST_BP, BORROW = "2017-01", "2024-12", 1500, 2.0, 35.0
NDRAW = 10

def build():
    px = pd.read_parquet("cache/mp_close.parquet")
    dv = pd.read_parquet("cache/mp_dv.parquet")
    mpx = px.resample("ME").last()
    fwd = mpx.pct_change(fill_method=None).shift(-1)          # next-month return
    mdv = dv.rolling(21, min_periods=10).mean().resample("ME").last()
    ym = lambda idx: idx.year*100 + idx.month
    fwd.index, mdv.index = ym(fwd.index), ym(mdv.index)

    cw = pd.read_csv("cache/crosswalk_sf.csv")
    cw = cw[cw["corr"] > 0.99]
    s2p = dict(zip(cw.symbol.str.upper(), cw.permno))
    print(f"crosswalk: {len(s2p)} symbols, {cw.delisted.mean():.1%} delisted")

    # 1.9GB on disk -- filter to the test window on read
    ch = pd.read_parquet("cache/osap_all_raw.parquet",
                         filters=[("yyyymm", ">=", int(LO.replace("-",""))),
                                  ("yyyymm", "<=", int(HI.replace("-","")))])
    sig = json.load(open("cache/osap_signs.json"))
    cols = [c for c in ch.columns if c not in ("permno","yyyymm")]
    print(f"characteristics: {len(cols)}  rows {len(ch):,}")
    return fwd, mdv, s2p, ch, cols, sig

def composite(ch, cols, sig, permnos, months, use_sign):
    """cross-sectional z-score per month, sign-aligned, equal-weight mean"""
    sub = ch[ch.yyyymm.isin(months)]
    sub = sub[sub.permno.isin(permnos)]
    out = {}
    for m, g in sub.groupby("yyyymm"):
        X = g[cols].astype(float)
        z = (X - X.mean()) / X.std(ddof=0)
        z = z.clip(-3, 3)
        if use_sign:
            for c in cols:
                s = sig.get(c)
                if s: z[c] = z[c]*s
        out[m] = pd.Series(z.mean(axis=1, skipna=True).values, index=g.permno.values)
    return out

def run(fwd, mdv, s2p, comp, months, seed=None):
    """seed != None permutes the composite across stocks within each month"""
    rng = np.random.default_rng(seed) if seed is not None else None
    rets, tos = [], []
    prev = None
    for m in months:
        if m not in comp or m not in fwd.index or m not in mdv.index: continue
        liq = mdv.loc[m].dropna().nlargest(NTOP)
        syms = [s for s in liq.index if s.upper() in s2p]
        pn = np.array([s2p[s.upper()] for s in syms])
        c = comp[m].reindex(pn).values
        ok = np.isfinite(c)
        if ok.sum() < 200: continue
        syms = list(np.array(syms)[ok]); c = c[ok]
        if rng is not None: c = c[rng.permutation(len(c))]
        w = c - c.mean(); s = np.abs(w).sum()
        if s == 0: continue
        w = w/s
        r = fwd.loc[m, syms].values.astype(float)
        good = np.isfinite(r)
        rets.append(float(np.nansum(w[good]*r[good])))
        cur = pd.Series(w, index=syms)
        if prev is None: tos.append(np.abs(cur).sum())
        else:
            al = cur.align(prev, fill_value=0.0)
            tos.append(float(np.abs(al[0]-al[1]).sum()))
        prev = cur
        short = float(np.abs(np.clip(w, None, 0)).sum())
        rets[-1] -= short*BORROW*1e-4/12
    return np.array(rets), np.array(tos)

def sharpe_m(r): return float(r.mean()/r.std()*np.sqrt(12)) if len(r) > 6 else float("nan")

def main(use_sign="1"):
    use_sign = use_sign == "1"
    fwd, mdv, s2p, ch, cols, sig = build()
    months = [m for m in sorted(ch.yyyymm.unique())
              if int(LO.replace("-","")) <= m <= int(HI.replace("-",""))]
    print(f"months {months[0]} .. {months[-1]}  ({len(months)})")
    allp = set(s2p.values())
    comp = composite(ch, cols, sig, allp, months, use_sign)
    r, to = run(fwd, mdv, s2p, comp, months)
    net = r - COST_BP*1e-4*to
    lo_, hi_ = 0.0, 500.0
    for _ in range(60):
        mm = (lo_+hi_)/2
        if sharpe_m(r - mm*1e-4*to) > 0: lo_ = mm
        else: hi_ = mm
    print(f"\nsign-aligned={use_sign}   months traded {len(r)}   mean turnover {to.mean():.2f}")
    print(f"  gross {sharpe_m(r):+.2f}   net@{COST_BP:.0f}bp {sharpe_m(net):+.2f}   breakeven {lo_:.0f}bp")
    print(f"  SE(Sharpe) ~ {np.sqrt(12/len(r)):.2f}")
    nulls = []
    for i in range(NDRAW):
        rn, tn = run(fwd, mdv, s2p, comp, months, seed=1000+i)
        nulls.append(sharpe_m(rn - COST_BP*1e-4*tn))
    nulls = np.array(nulls)
    print(f"  null net: mean {nulls.mean():+.2f}  max {nulls.max():+.2f}  "
          f"draws {np.round(nulls,2).tolist()}")
    s = sharpe_m(net)
    if s >= 0.50 and s > nulls.max() and lo_ >= 6: v = "WORKS"
    elif s <= 0.15 or s <= nulls.max(): v = "DEAD"
    else: v = "AMBIGUOUS"
    print(f"\n  VERDICT: {v}")

if __name__ == "__main__":
    main(*sys.argv[1:])
