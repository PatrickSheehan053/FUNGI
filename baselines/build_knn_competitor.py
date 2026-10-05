"""
exp_029 Step 3 — the PURE kNN co-expression competitor arm (blueprint: borda_knn is a fused variant, NOT the
pure competitor). Built leakage-safe from the TRAIN metacell expression (RPE1_hybrid_train_lognorm.h5ad):
per gene, the top-k neighbours by |Pearson correlation| (directed i->j), capped to 200k edges.

  k = 40 per source × 5000 genes = 200,000 edges (matched to the other arms). Weight = |corr|.
Writes ship/arms/knn.npz (src/tgt/w) — the arbiter arm contract.
"""
import os, sys
os.environ.setdefault("PYTHONUTF8", "1")
from pathlib import Path
import numpy as np

EXP = Path(__file__).resolve().parents[1]
SHIP = EXP / "ship"
E25 = EXP.parent / "exp_025_hyphae_vs_shroom_ensemble" / "rpe1_run"
H5 = E25 / "RPE1_hybrid_train_lognorm.h5ad"
K_PER = 40
TARGET = 200000


def main():
    import scanpy as sc
    panel = np.load(SHIP / "lfc_targets_rpe1.npz", allow_pickle=True)["panel"]
    panel = np.asarray([str(x) for x in panel])
    ad = sc.read_h5ad(H5)
    # align to the 5000-gene panel
    vn = np.asarray([str(x) for x in ad.var_names])
    idx = {g: i for i, g in enumerate(vn)}
    cols = np.array([idx.get(g, -1) for g in panel])
    assert (cols >= 0).all(), f"panel genes missing from h5ad: {(cols<0).sum()}"
    X = ad.X[:, cols]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    X = np.asarray(X, np.float64)                       # (cells, 5000) train metacells
    N = X.shape[1]
    print(f"expr {X.shape} -> {N} genes; computing |corr| and per-gene top-{K_PER}")
    # gene-gene Pearson correlation
    Xc = X - X.mean(axis=0, keepdims=True)
    sd = Xc.std(axis=0, keepdims=True); sd[sd < 1e-12] = 1.0
    Xn = Xc / sd
    C = (Xn.T @ Xn) / X.shape[0]                        # (N,N) correlation
    A = np.abs(C); np.fill_diagonal(A, -1.0)            # exclude self
    src, tgt, w = [], [], []
    for i in range(N):
        row = A[i]
        k = min(K_PER, N - 1)
        top = np.argpartition(row, -k)[-k:]
        src.append(np.full(k, i, np.int64)); tgt.append(top.astype(np.int64)); w.append(row[top].astype(np.float64))
    src = np.concatenate(src); tgt = np.concatenate(tgt); w = np.concatenate(w)
    # cap to TARGET by global |corr| if over
    if len(src) > TARGET:
        keep = np.argpartition(w, -TARGET)[-TARGET:]
        src, tgt, w = src[keep], tgt[keep], w[keep]
    np.savez(SHIP / "arms" / "knn.npz", src=src, tgt=tgt, w=w.astype(np.float32),
             n_src=int(len(set(src.tolist()))), n_edges=int(len(src)))
    print(f"knn.npz: {len(src):,} edges, {len(set(src.tolist()))} sources -> {SHIP/'arms'/'knn.npz'}")


if __name__ == "__main__":
    main()
