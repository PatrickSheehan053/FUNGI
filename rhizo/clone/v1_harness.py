"""
obj_009 — harness.py : GPU-resident DNPN training + prediction (freeze-safe).

train_one(): fits a fresh DNPN on a fit set (residual targets), early-stops on a held-out val subset's RSC,
returns predictions on the eval set. GPU-resident: graph + all targets + signal mask are moved to CUDA ONCE
before the epoch loop; num_workers=0; no CPU↔GPU transfer inside the gradient loop (2070 freeze avoidance).

Loss: DEG-weighted MSE on the specific residual s over the train-only signal genes S, EXCLUDING the perturbed
gene, + 0.3·(1 − Pearson). Early-stop on val RSC (not val loss), patience 20, AdamW lr 1e-3 wd 1e-4, grad-clip
1.0, cosine schedule. delta_mode default neg_mu_ctrl (δ_p = −μ_ctrl[p]).
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from model import DNPN


def make_delta(pidx, mu_ctrl, mode="neg_mu_ctrl"):
    d = np.zeros(len(pidx), np.float64); v = pidx >= 0; idx = pidx[v]
    if mode == "neg_mu_ctrl": d[v] = -mu_ctrl[idx]
    elif mode == "neg_half_mu_ctrl": d[v] = -0.5 * mu_ctrl[idx]
    elif mode == "neg_one": d[v] = -1.0
    else: raise ValueError(mode)
    return d


def _gpu_rsc(pred, targ, sig, pidx_t):
    """Mean over perts of Pearson(pred, targ) over signal genes excl the perturbed gene; Fisher-averaged."""
    B = pred.shape[0]; rs = []
    for b in range(B):
        cols = sig[sig != pidx_t[b]] if pidx_t[b] >= 0 else sig
        a = pred[b, cols]; c = targ[b, cols]
        a = a - a.mean(); c = c - c.mean()
        da = torch.sqrt((a * a).sum()); dc = torch.sqrt((c * c).sum())
        if da < 1e-12 or dc < 1e-12:
            rs.append(torch.tensor(0.0, device=pred.device)); continue
        rs.append((a * c).sum() / (da * dc))
    r = torch.stack(rs).clamp(-0.999999, 0.999999)
    return torch.tanh(torch.arctanh(r).mean()).item()


def _predict(model, pidx_t, delta_t, src, tgt, w, N, batch=32):
    model.eval(); outs = []
    with torch.no_grad():
        for i in range(0, len(pidx_t), batch):
            outs.append(model(pidx_t[i:i+batch], delta_t[i:i+batch], src, tgt, w, N))
    return torch.cat(outs, 0)


def train_one(graph, fit_pidx, fit_s, eval_pidx, eval_s, signal_mask, mu_ctrl,
              cfg, seed=0, device="cuda", delta_mode="neg_mu_ctrl", verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    N = len(mu_ctrl)
    src = torch.as_tensor(graph[0], dtype=torch.long, device=dev)
    tgt = torch.as_tensor(graph[1], dtype=torch.long, device=dev)
    w = torch.as_tensor(graph[2], dtype=torch.float32, device=dev)
    sig = torch.as_tensor(np.where(signal_mask)[0], dtype=torch.long, device=dev)

    # internal val split of the FIT set for early-stopping (leakage-clean: never touches eval set)
    rng = np.random.default_rng(seed); order = rng.permutation(len(fit_pidx))
    n_val = max(8, int(0.15 * len(fit_pidx)))
    val_ix, tr_ix = order[:n_val], order[n_val:]
    d_fit = make_delta(fit_pidx, mu_ctrl, delta_mode)

    fit_s_t = torch.as_tensor(fit_s, dtype=torch.float32, device=dev)
    fit_pidx_t = torch.as_tensor(fit_pidx, dtype=torch.long, device=dev)
    d_fit_t = torch.as_tensor(d_fit, dtype=torch.float32, device=dev)
    tr_ix_t = torch.as_tensor(tr_ix, dtype=torch.long, device=dev)
    val_ix_t = torch.as_tensor(val_ix, dtype=torch.long, device=dev)

    model = DNPN(d=cfg["d"], d_hidden=cfg["d_hidden"], K=cfg["K"],
                 oversmooth=cfg.get("oversmooth", "jk"), dropout=cfg.get("dropout", 0.1)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 1e-4))
    epochs = cfg.get("epochs", 200); patience = cfg.get("patience", 20); bs = cfg.get("batch_perts", 32)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = -1e9; best_state = None; bad = 0
    for ep in range(epochs):
        model.train()
        perm = tr_ix_t[torch.randperm(len(tr_ix_t), device=dev)]
        for i in range(0, len(perm), bs):
            bidx = perm[i:i+bs]
            pred = model(fit_pidx_t[bidx], d_fit_t[bidx], src, tgt, w, N)   # (b,N)
            ps = pred[:, sig]; ts = fit_s_t[bidx][:, sig]                    # (b,|S|)
            # exclude perturbed gene: zero its column per pert
            excl = (sig.unsqueeze(0) != fit_pidx_t[bidx].unsqueeze(1)).float()   # (b,|S|)
            wdeg = ts.abs() * excl
            denom = wdeg.sum().clamp_min(1e-8)
            mse = (wdeg * (ps - ts) ** 2).sum() / denom
            # correlation term (per pert, over excluded signal genes)
            pm = ps - (ps * excl).sum(1, keepdim=True) / excl.sum(1, keepdim=True).clamp_min(1)
            tm = ts - (ts * excl).sum(1, keepdim=True) / excl.sum(1, keepdim=True).clamp_min(1)
            num = (pm * tm * excl).sum(1)
            # clamp INSIDE sqrt: sqrt(0) has an infinite gradient (NaN in backward); a per-pert constant
            # prediction (pm≈0) is common early in training, so this guard is load-bearing.
            den = torch.sqrt(((pm * pm * excl).sum(1) * (tm * tm * excl).sum(1)).clamp_min(1e-12))
            corr = (num / den)
            # RSC (the metric) IS scale-invariant per-pert correlation, so CORRELATION is the primary loss;
            # the DEG-weighted MSE is a light scale/sign anchor (a large-MSE-primary loss makes the optimizer
            # chase output magnitude instead of the pattern RSC rewards — the cause of the initial under-fit).
            loss = cfg.get("corr_weight", 1.0) * (1.0 - corr.mean()) + cfg.get("mse_weight", 0.1) * mse
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            opt.step()
        sched.step()
        # early-stop on val RSC
        vp = _predict(model, fit_pidx_t[val_ix_t], d_fit_t[val_ix_t], src, tgt, w, N, batch=bs)
        vrsc = _gpu_rsc(vp, fit_s_t[val_ix_t], sig, fit_pidx_t[val_ix_t].tolist())
        if vrsc > best_val + cfg.get("min_delta", 3e-4):   # require a REAL improvement (noise arms stop early)
            best_val = vrsc; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 10 == 0:
            print(f"  ep{ep} loss={loss.item():.4f} val_rsc={vrsc:+.4f} best={best_val:+.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    d_eval = make_delta(eval_pidx, mu_ctrl, delta_mode)
    ep_t = torch.as_tensor(eval_pidx, dtype=torch.long, device=dev)
    de_t = torch.as_tensor(d_eval, dtype=torch.float32, device=dev)
    pred = _predict(model, ep_t, de_t, src, tgt, w, N, batch=bs).cpu().numpy()
    return pred, dict(best_val_rsc=best_val, epochs_run=ep + 1)
