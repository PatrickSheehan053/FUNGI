"""
obj_008 — graph_io.py

Edge list -> direction-preserving normalized sparse adjacency A-tilde for DWLP.

Convention (load-bearing):
  A[t, s] = w(s -> t)     row = target, col = source/regulator
  propagation  h^(k) = A-tilde @ h^(k-1),  h^(0) = e_p * delta_p
  => a shift on regulator p reaches target t iff there is a directed walk p -> ... -> t.

Normalization (HARD REQUIREMENT — never symmetric, never (A+A^T)/2):
  - "col" (default): out-normalization  A-tilde[t,s] = A[t,s] / (sum_t A[t,s] + eps)
        a regulator distributes its shift over its targets; column-substochastic => rho(A-tilde) <= 1.
  - "in"          : in-normalization by target in-strength  A-tilde[t,s] = A[t,s] / (sum_s A[t,s] + eps)

Spectral-radius guard: rho(A-tilde) is bounded <= 1 by construction for both modes; we still measure it
(Perron-Frobenius max col/row-sum bound + a best-effort eigs estimate) and scale by 1/rho if rho > 1+tol.

Derived arms built in-memory from the FUNGI edge list:
  - "reverse"   : flip every edge direction (A -> A^T) — diagnostic; if direction matters this should hurt.
  - "labelperm" : permute node labels (keep degree/density, destroy biological identity alignment).

CPU-only, sparse throughout.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

EPS = 1e-8


def load_edges(path: str):
    """Read an edge list (with header). TSV or .parquet. Detects column names case-insensitively:
    source/regulator/tf for the source, target for the target, weight/importance for the weight.
    Returns arrays src, tgt (str), w (float64)."""
    if str(path).lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    sc = cols.get("source", cols.get("regulator", cols.get("tf")))
    tc = cols.get("target")
    wc = cols.get("weight", cols.get("importance", cols.get("score")))
    if sc is None or tc is None or wc is None:
        raise KeyError(f"could not detect source/target/weight columns in {list(df.columns)}")
    s = df[sc].astype(str).values
    t = df[tc].astype(str).values
    w = df[wc].astype(np.float64).values
    return s, t, w


def _raw_adjacency(src, tgt, w, gene_to_idx, N):
    """Build raw directed weighted A with A[t,s]=w(s->t). Endpoints outside the panel are dropped (logged)."""
    si = np.array([gene_to_idx.get(g, -1) for g in src])
    ti = np.array([gene_to_idx.get(g, -1) for g in tgt])
    keep = (si >= 0) & (ti >= 0)
    dropped = int((~keep).sum())
    si, ti, wv = si[keep], ti[keep], w[keep]
    A = sp.coo_matrix((wv, (ti, si)), shape=(N, N)).tocsr()  # row=target, col=source
    A.sum_duplicates()
    return A, dropped


def normalize(A: sp.csr_matrix, mode: str = "col"):
    """Direction-preserving normalization. mode='col' (out) or 'in'. Returns CSR A-tilde."""
    A = A.tocsr().astype(np.float64)
    if mode == "col":
        denom = np.asarray(A.sum(axis=0)).ravel() + EPS       # per-source (column) out-strength
        scale = sp.diags(1.0 / denom)
        At = A @ scale                                          # scales columns
    elif mode == "in":
        denom = np.asarray(A.sum(axis=1)).ravel() + EPS        # per-target (row) in-strength
        scale = sp.diags(1.0 / denom)
        At = scale @ A                                          # scales rows
    else:
        raise ValueError(f"unknown norm_mode {mode!r}; use 'col' or 'in' (NEVER symmetric)")
    return At.tocsr()


def spectral_radius(At: sp.csr_matrix, want_eigs: bool = True):
    """Perron-Frobenius bound (min of max row/col sum for nonnegative A) + best-effort eigs estimate."""
    absA = At.copy()
    absA.data = np.abs(absA.data)
    max_col = float(np.asarray(absA.sum(axis=0)).ravel().max())
    max_row = float(np.asarray(absA.sum(axis=1)).ravel().max())
    bound = min(max_col, max_row)
    est = bound
    if want_eigs and At.shape[0] > 2:
        try:
            ev = spla.eigs(At.astype(np.float64), k=1, which="LM", maxiter=2000, tol=1e-4,
                           return_eigenvectors=False)
            est = float(np.abs(ev[0]))
        except Exception:
            est = bound
    return dict(bound=bound, eigs=est, max_col_sum=max_col, max_row_sum=max_row)


def out_neighbors(A_raw: sp.csr_matrix):
    """From raw A (A[t,s]=w(s->t)), return CSC so column s gives the out-neighbors (targets) of source s."""
    return A_raw.tocsc()


def _apply_derived(src, tgt, w, kind: str, N: int, gene_to_idx, seed: int = 0):
    """Return possibly-modified (src_idx, tgt_idx, w) for a derived arm, plus a permutation if any."""
    if kind == "reverse":
        return tgt, src, w  # swap direction (still symbols; mapped later)
    if kind == "labelperm":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(N)
        genes = list(gene_to_idx.keys())
        # map each gene symbol -> permuted gene symbol
        inv = {g: genes[perm[gene_to_idx[g]]] for g in genes}
        s2 = np.array([inv.get(g, g) for g in src])
        t2 = np.array([inv.get(g, g) for g in tgt])
        return s2, t2, w
    raise ValueError(f"unknown derived arm {kind!r}")


def load_graph(arm: str, cfg: dict, panel_genes, seed: int = 0, want_eigs: bool = True):
    """
    Load/normalize a graph arm. Returns dict:
      At        : CSR normalized adjacency (N x N)
      A_raw     : CSR raw adjacency (for out-neighbor / DNSA)
      rho       : spectral-radius diagnostics
      n_edges   : int
      out_deg   : per-node out-degree (n_out_neighbors as a source), length N
      arm, mode, dropped
    """
    root = cfg["paths"]["thesis_root"]
    gdir = os.path.join(root, cfg["paths"]["baseline_graphs_dir"]) if not os.path.isabs(
        cfg["paths"]["baseline_graphs_dir"]) else cfg["paths"]["baseline_graphs_dir"]
    N = len(panel_genes)
    gene_to_idx = {g: i for i, g in enumerate(panel_genes)}
    mode = cfg["model"]["norm_mode"]

    derived = arm in ("reverse", "labelperm")
    fname = cfg["graphs"].get("fungi" if derived else arm)
    if fname is None:
        raise KeyError(f"no graph file configured for arm {arm!r}")
    path = os.path.join(gdir, fname)
    src, tgt, w = load_edges(path)
    if derived:
        src, tgt, w = _apply_derived(src, tgt, w, arm, N, gene_to_idx, seed=seed)

    A_raw, dropped = _raw_adjacency(src, tgt, w, gene_to_idx, N)
    At = normalize(A_raw, mode=mode)

    rho = spectral_radius(At, want_eigs=want_eigs)
    if cfg["model"].get("spectral_guard", True) and rho["eigs"] > 1.0 + 1e-6:
        At = At.multiply(1.0 / rho["eigs"]).tocsr()
        rho["rescaled_by"] = 1.0 / rho["eigs"]

    A_csc = A_raw.tocsc()
    out_deg = np.diff(A_csc.indptr)  # nonzeros per column = out-neighbors per source
    return dict(arm=arm, At=At.tocsr(), A_raw=A_raw.tocsr(), A_csc=A_csc,
                rho=rho, n_edges=int(A_raw.nnz), out_deg=out_deg,
                mode=mode, dropped=int(dropped))


if __name__ == "__main__":
    # smoke: report rho + edge counts for every configured arm
    import yaml, json
    cfgp = os.path.join(os.path.dirname(__file__), "..", "configs", "obj_008.yaml")
    cfg = yaml.safe_load(open(os.path.abspath(cfgp)))
    root = cfg["paths"]["thesis_root"]
    cache = os.path.join(root, cfg["paths"]["cache_dir"])
    pb = np.load(os.path.join(cache, "pseudobulk.npz"), allow_pickle=True)
    panel_genes = list(pb["panel_genes"])
    for arm in list(cfg["graphs"].keys()) + ["reverse", "labelperm"]:
        g = load_graph(arm, cfg, panel_genes, seed=0)
        print(f"{arm:10s} edges={g['n_edges']:>7d} dropped={g['dropped']:>4d} "
              f"rho_bound={g['rho']['bound']:.4f} rho_eigs={g['rho']['eigs']:.4f} "
              f"max_outdeg={int(g['out_deg'].max())}")
