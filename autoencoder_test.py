"""IPCA and autoencoder factor models, as pre-registered in PREREG_autoencoder.md.

  IPCA          beta_it = Gamma' z_it           (linear in characteristics)
  Autoencoder   beta_it = g(z_it), g a small NN (Gu-Kelly-Xiu)
  PCA-5         benchmark, no characteristics

Factors solved per period given betas in all three.  Predicted return uses the
training-window mean factor: rhat = beta_it' fbar.  Walk-forward, 60-month fit,
12-month application.
"""
import numpy as np, pickle, sys, torch, torch.nn as nn, warnings
warnings.filterwarnings("ignore")
K, HID, FIT, APPLY = 5, 32, 60, 12
COST_BP, BORROW = 2.0, 35.0

R = np.load("cache/char_R.npy", allow_pickle=True)
Z = np.load("cache/char_Z.npy", allow_pickle=True)
IDX = pickle.load(open("cache/char_idx.pkl","rb"))
M = len(R); L = Z[0].shape[1]
print(f"{M} months, {L} characteristics, median n {int(np.median([len(r) for r in R]))}")

def solve_f(B, r):
    """given betas (n x K) solve the period factor by OLS"""
    G = B.T @ B + 1e-6*np.eye(B.shape[1])
    return np.linalg.solve(G, B.T @ r)

def fit_ipca(zs, rs, iters=20):
    G = np.linalg.svd(np.concatenate(zs,0), full_matrices=False)[2][:K].T   # L x K
    for _ in range(iters):
        F = [solve_f(z @ G, r) for z, r in zip(zs, rs)]
        A = np.zeros((L*K, L*K)); b = np.zeros(L*K)
        for z, r, f in zip(zs, rs, F):
            zf = np.kron(z, f.reshape(1,-1))        # n x (L*K)
            A += zf.T @ zf; b += zf.T @ r
        G = np.linalg.solve(A + 1e-4*np.eye(L*K), b).reshape(L, K)
    F = np.array([solve_f(z @ G, r) for z, r in zip(zs, rs)])
    return G, F

class Beta(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(L, HID), nn.ReLU(), nn.Linear(HID, K))
    def forward(s, z): return s.net(z)

def fit_ae(zs, rs, epochs=120, seed=0):
    torch.manual_seed(seed)
    m = Beta(); opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    zt = [torch.tensor(z) for z in zs]; rt = [torch.tensor(r) for r in rs]
    for _ in range(epochs):
        loss = 0.0
        for z, r in zip(zt, rt):
            B = m(z)
            G = B.T @ B + 1e-6*torch.eye(K)
            f = torch.linalg.solve(G, B.T @ r)      # factors solved, not learned
            loss = loss + ((r - B @ f)**2).mean()
        opt.zero_grad(); (loss/len(zt)).backward(); opt.step()
    with torch.no_grad():
        F = []
        for z, r in zip(zt, rt):
            B = m(z); G = B.T @ B + 1e-6*torch.eye(K)
            F.append(torch.linalg.solve(G, B.T @ r).numpy())
    return m, np.array(F)

def pca_pred(zs, rs, z_new):
    """PCA-5 benchmark: no characteristics, so predict the cross-sectional mean"""
    return np.zeros(len(z_new))

def evaluate(pred_fn, label, permute_seed=None):
    rng = np.random.default_rng(permute_seed) if permute_seed is not None else None
    rets, tos, months = [], [], []
    prev = None
    for t0 in range(FIT, M-APPLY, APPLY):
        zs = list(Z[t0-FIT:t0]); rs = list(R[t0-FIT:t0])
        if permute_seed is not None:
            zs = [z[rng.permutation(len(z))] for z in zs]
        model = pred_fn["fit"](zs, rs)
        for t in range(t0, min(t0+APPLY, M)):
            z = Z[t].copy()
            if permute_seed is not None: z = z[rng.permutation(len(z))]
            rhat = pred_fn["pred"](model, z)
            w = rhat - rhat.mean(); s = np.abs(w).sum()
            if s == 0: continue
            w = w/s
            rets.append(float(w @ R[t]))
            syms = IDX[t][1]
            cur = dict(zip(syms, w))
            if prev is None: tos.append(np.abs(w).sum())
            else:
                keys = set(cur) | set(prev)
                tos.append(sum(abs(cur.get(k,0)-prev.get(k,0)) for k in keys))
            prev = cur
            months.append(IDX[t][0])
            rets[-1] -= float(np.abs(np.clip(w,None,0)).sum())*BORROW*1e-4/12
    return np.array(rets), np.array(tos), np.array(months)

def sharpe(x): return float(x.mean()/x.std()*np.sqrt(12)) if len(x) > 6 else float("nan")

MODELS = {
 "IPCA (linear betas)": {
    "fit": lambda zs, rs: (lambda GF: (GF[0], GF[1].mean(0)))(fit_ipca(zs, rs)),
    "pred": lambda m, z: (z @ m[0]) @ m[1]},
 "Autoencoder (NN betas)": {
    "fit": lambda zs, rs: (lambda MF: (MF[0], MF[1].mean(0)))(fit_ae(zs, rs)),
    "pred": lambda m, z: (m[0](torch.tensor(z)).detach().numpy()) @ m[1]},
}

print(f"\n{'model':26s}{'gross':>8s}{'turn':>7s}{'b/e bp':>8s}{'net':>8s}{'n mo':>7s}")
print("-"*64)
out = {}
for lab, fn in MODELS.items():
    r, to, mo = evaluate(fn, lab)
    m = mo >= 201701
    g, t = r[m], to[m]
    net = g - COST_BP*1e-4*t
    lo, hi = 0.0, 500.0
    for _ in range(50):
        mm = (lo+hi)/2
        if sharpe(g - mm*1e-4*t) > 0: lo = mm
        else: hi = mm
    out[lab] = sharpe(net)
    print(f"{lab:26s}{sharpe(g):+8.2f}{t.mean():7.2f}{lo:8.0f}{sharpe(net):+8.2f}{m.sum():7d}")
print(f"\nSE(Sharpe) on ~96 months ~ {np.sqrt(12/96):.2f}")
pickle.dump(out, open("cache/ae_results.pkl","wb"))
