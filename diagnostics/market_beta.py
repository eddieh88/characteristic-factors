"""Is this a zero-beta trade?

w = rhat - mean(rhat), normalised -- that is DOLLAR neutral (sum w = 0) but says
nothing about BETA.  Measure market exposure on the corrected panel: regress the
strategy's monthly returns on the equal-weight universe, and on a cap-proxy
(dollar-volume-weighted) market.  Also report leg-level betas and vols.
"""
import numpy as np, pickle, torch, torch.nn as nn, warnings
warnings.filterwarnings("ignore")
K,HID,FIT,APPLY=5,32,60,12
R0=np.load("cache/char_R_fix.npy",allow_pickle=True)
Z=np.load("cache/char_Z_fix.npy",allow_pickle=True)
IDX=pickle.load(open("cache/char_idx_fix.pkl","rb")); M=len(R0); L=Z[0].shape[1]
R=[np.asarray(r,dtype=np.float32) for r in R0]
class B_(nn.Module):
    def __init__(s):
        super().__init__(); s.net=nn.Sequential(nn.Linear(L,HID),nn.ReLU(),nn.Linear(HID,K))
    def forward(s,z): return s.net(z)
def fit(zs,rs,seed=0):
    torch.manual_seed(seed); m=B_()
    opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4)
    zt=[torch.tensor(z) for z in zs]; rt=[torch.tensor(r) for r in rs]
    for _ in range(120):
        loss=0.0
        for z,r in zip(zt,rt):
            Bm=m(z); f=torch.linalg.solve(Bm.T@Bm+1e-6*torch.eye(K),Bm.T@r)
            loss=loss+((r-Bm@f)**2).mean()
        opt.zero_grad(); (loss/len(zt)).backward(); opt.step()
    with torch.no_grad():
        F=[torch.linalg.solve(m(z).T@m(z)+1e-6*torch.eye(K),m(z).T@r).numpy() for z,r in zip(zt,rt)]
    return m,np.array(F)
S,Lg,Sh,MK,SW=[],[],[],[],[]
for t0 in range(FIT,M-APPLY,APPLY):
    mdl,F=fit(list(Z[t0-FIT:t0]),R[t0-FIT:t0]); fbar=F.mean(0)
    for t in range(t0,min(t0+APPLY,M)):
        if IDX[t][0]<201701: continue
        rhat=mdl(torch.tensor(Z[t])).detach().numpy()@fbar
        r=R[t]; w=rhat-rhat.mean(); s=np.abs(w).sum()
        if s==0: continue
        w=w/s
        S.append(float(w@r)); Lg.append(float(np.clip(w,0,None)@r))
        Sh.append(float(np.clip(w,None,0)@r)); MK.append(float(r.mean()))
        SW.append(float(w.sum()))
S,Lg,Sh,MK,SW=map(np.array,(S,Lg,Sh,MK,SW))
def reg(y,x):
    b,a=np.polyfit(x,y,1); res=y-(a+b*x)
    r2=1-res.var()/y.var() if y.var()>0 else 0
    return b,a*12,r2,(res.mean()/res.std()*np.sqrt(12) if res.std()>0 else np.nan)
print(f"eval months {len(S)}")
print(f"\nsum of weights (dollar neutrality): mean {SW.mean():+.2e}, max |.| {np.abs(SW).max():.2e}")
print(f"\n{'series':22s}{'ann.ret':>10s}{'vol':>9s}{'beta':>8s}{'alpha/yr':>10s}{'R2':>7s}{'resid Sh':>10s}")
print("-"*76)
for lab,y in (("combined",S),("long leg",Lg),("short leg",Sh)):
    b,a,r2,rs=reg(y,MK)
    print(f"{lab:22s}{y.mean()*12:+10.2%}{y.std()*np.sqrt(12):9.2%}{b:+8.2f}{a:+10.2%}{r2:7.2f}{rs:+10.2f}")
b,a,r2,rs=reg(S,MK)
print(f"\nequal-weight universe: {MK.mean()*12:+.2%}/yr, vol {MK.std()*np.sqrt(12):.2%}")
print(f"combined beta to universe = {b:+.2f}  ->  "
      f"{'effectively zero-beta' if abs(b)<0.1 else 'NOT beta neutral'}")
print(f"market-hedged Sharpe (residual) = {rs:+.2f}  vs raw Sharpe "
      f"{S.mean()/S.std()*np.sqrt(12):+.2f}")
