"""Is the autoencoder worse than just trading traditional factors?

Same corrected point-in-time panel, same window (2017-2024), same costs
(2bp one-way, 35bp/yr borrow), same dollar-neutral construction.  Each classic
characteristic is used directly as the score.  Signs from OSAP's Sign field --
NOTE this is the one in-sample choice here, worth +0.20 of Sharpe in Step 3, so
the traditional arm is if anything flattered relative to the autoencoder.

Reports raw Sharpe and market-hedged Sharpe, because the autoencoder's +0.31
was entirely beta (+0.33) and hedged to +0.00.
"""
import numpy as np, pickle, json, warnings
warnings.filterwarnings("ignore")
COST_BP, BORROW = 2.0, 35.0
R0=np.load("cache/char_R_fix.npy",allow_pickle=True)
Z=np.load("cache/char_Z_fix.npy",allow_pickle=True)
IDX=pickle.load(open("cache/char_idx_fix.pkl","rb"))
names=list(json.load(open("cache/osap_signs.json")).keys())
signs=json.load(open("cache/osap_signs.json"))
M=len(R0); R=[np.asarray(r,dtype=np.float32) for r in R0]
def sharpe(x): return float(x.mean()/x.std()*np.sqrt(12)) if len(x)>6 else float("nan")

def run_score(score_fn, use_sign=None):
    rets,tos,mos,mk=[],[],[],[]; prev=None
    for t in range(M):
        if IDX[t][0]<201701: continue
        s=score_fn(t)
        if s is None: continue
        w=s-s.mean(); n=np.abs(w).sum()
        if n==0: continue
        w=w/n; cur=dict(zip(IDX[t][1],w))
        if prev is None: tos.append(float(np.abs(w).sum()))
        else:
            keys=set(cur)|set(prev); tos.append(sum(abs(cur.get(k,0)-prev.get(k,0)) for k in keys))
        prev=cur
        rets.append(float(w@R[t])-float(np.abs(np.clip(w,None,0)).sum())*BORROW*1e-4/12)
        mk.append(float(R[t].mean())); mos.append(IDX[t][0])
    r,to,m_=np.array(rets),np.array(tos),np.array(mk)
    net=r-COST_BP*1e-4*to
    b=np.polyfit(m_,net,1)[0]
    hedged=net-b*m_
    return sharpe(net), b, sharpe(hedged), to.mean()

CLASSIC=[("BM (value)","BM"),("Mom12m (momentum)","Mom12m"),("GP (profitability)","GP"),
         ("OperProf (profitability)","OperProf"),("AssetGrowth (investment)","AssetGrowth"),
         ("Accruals","Accruals"),("IdioVol3F (low vol)","IdioVol3F"),
         ("Illiquidity","Illiquidity"),("Beta (betting against beta)","Beta")]
print(f"{'factor':32s}{'net Sh':>8s}{'beta':>7s}{'hedged':>8s}{'turn':>7s}")
print("-"*62)
res={}
for lab,nm in CLASSIC:
    j=names.index(nm); sg=signs[nm]
    s,b,h,tu=run_score(lambda t,j=j,sg=sg: Z[t][:,j]*sg)
    res[lab]=(s,h); print(f"{lab:32s}{s:+8.2f}{b:+7.2f}{h:+8.2f}{tu:7.2f}", flush=True)

idx=[names.index(n) for _,n in CLASSIC]; sg=np.array([signs[n] for _,n in CLASSIC])
s,b,h,tu=run_score(lambda t: (Z[t][:,idx]*sg).mean(1))
print("-"*62)
print(f"{'equal-weight composite of the 9':32s}{s:+8.2f}{b:+7.2f}{h:+8.2f}{tu:7.2f}")
allsg=np.array([signs[n] for n in names])
s,b,h,tu=run_score(lambda t: (Z[t]*allsg).mean(1))
print(f"{'equal-weight all 209 (signed)':32s}{s:+8.2f}{b:+7.2f}{h:+8.2f}{tu:7.2f}")
print("-"*62)
print(f"{'AUTOENCODER (for comparison)':32s}{+0.31:+8.2f}{+0.33:+7.2f}{+0.00:+8.2f}{0.88:7.2f}")
mk=np.array([R[t].mean() for t in range(M) if IDX[t][0]>=201701])
print(f"\nequal-weight universe (long only): {mk.mean()*12:+.2%}/yr, Sharpe {sharpe(mk):+.2f}")
