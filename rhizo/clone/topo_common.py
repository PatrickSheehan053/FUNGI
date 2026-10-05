"""
obj_009.1 — topo_common.py : shared GRAPH-DERIVED topology helpers for the V-EDGE / V-FILM extractors.

Everything here is a deterministic function of the swapped graph ALONE (graph-agnostic invariant #4): degrees,
strengths, directed PageRank, undirected Louvain communities, undirected local clustering, per-node
reciprocity, and per-edge reciprocity — computed identically for every arm. NO gene identity is used anywhere
(invariant #2). CPU-only; called in the cache-build phase (no GPU running -> respects the no-CPU∥GPU rule).

Convention (inherited): a directed edge is (s -> t); adjacency A[t,s]=w(s->t). "out" = as a source, "in" = as
a target. Reciprocal of edge (s,t) is edge (t,s).
"""
from __future__ import annotations
import numpy as np


def _zscore(x: np.ndarray) -> np.ndarray:
    """Population z-score with a std floor. Deterministic per graph (graph-agnostic)."""
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return x.astype(np.float32)
    mu = x.mean(); sd = x.std()
    return ((x - mu) / (sd + 1e-8)).astype(np.float32)


def degree_strength(src, tgt, w_raw, N):
    """Per-node out/in degree (edge COUNTS) and out/in strength (RAW weight sums)."""
    src = np.asarray(src, np.int64); tgt = np.asarray(tgt, np.int64)
    w_raw = np.asarray(w_raw, np.float64)
    outdeg = np.bincount(src, minlength=N).astype(np.float64) if src.size else np.zeros(N)
    indeg = np.bincount(tgt, minlength=N).astype(np.float64) if tgt.size else np.zeros(N)
    out_str = np.bincount(src, weights=w_raw, minlength=N) if src.size else np.zeros(N)
    in_str = np.bincount(tgt, weights=w_raw, minlength=N) if tgt.size else np.zeros(N)
    return dict(outdeg=outdeg, indeg=indeg, out_str=out_str, in_str=in_str)


def edge_pair_index(src, tgt, N):
    """Map each directed edge -> a set/dict for O(1) reciprocal lookup. Key = s*N + t."""
    src = np.asarray(src, np.int64); tgt = np.asarray(tgt, np.int64)
    if src.size == 0:
        return {}
    keys = src * np.int64(N) + tgt
    # last write wins on duplicates (there should be none after graphs.py dedup); index into the edge array
    return {int(k): i for i, k in enumerate(keys)}


def reciprocity_per_edge(src, tgt, w_outnorm, N):
    """For each edge (s,t): is_recip (1 if (t,s) exists), w_recip (w_outnorm of (t,s) or 0)."""
    src = np.asarray(src, np.int64); tgt = np.asarray(tgt, np.int64)
    E = src.size
    is_recip = np.zeros(E, np.float32); w_recip = np.zeros(E, np.float32)
    if E == 0:
        return is_recip, w_recip
    idx = edge_pair_index(src, tgt, N)
    rev_keys = tgt * np.int64(N) + src
    for e in range(E):
        j = idx.get(int(rev_keys[e]), -1)
        if j >= 0:
            is_recip[e] = 1.0
            w_recip[e] = float(w_outnorm[j])
    return is_recip, w_recip


def reciprocity_per_node(src, tgt, N):
    """Per-node reciprocity fraction = |out_nbrs ∩ in_nbrs| / |out_nbrs ∪ in_nbrs| (0 if isolated)."""
    src = np.asarray(src, np.int64); tgt = np.asarray(tgt, np.int64)
    out_nbrs = [set() for _ in range(N)]
    in_nbrs = [set() for _ in range(N)]
    for s, t in zip(src.tolist(), tgt.tolist()):
        out_nbrs[s].add(t); in_nbrs[t].add(s)
    frac = np.zeros(N, np.float64)
    for v in range(N):
        o = out_nbrs[v]; i = in_nbrs[v]
        if not o and not i:
            continue
        inter = len(o & i); union = len(o | i)
        frac[v] = inter / union if union else 0.0
    return frac


def _undirected_nx(src, tgt, w_raw, N):
    """Undirected projection as an nx.Graph with SUMMED reciprocal weights; all N nodes present."""
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(N))
    if len(src):
        agg = {}
        for s, t, w in zip(np.asarray(src).tolist(), np.asarray(tgt).tolist(), np.asarray(w_raw, np.float64).tolist()):
            if s == t:
                continue
            key = (s, t) if s < t else (t, s)
            agg[key] = agg.get(key, 0.0) + float(w)
        G.add_weighted_edges_from([(a, b, w) for (a, b), w in agg.items()])
    return G


def louvain_labels(src, tgt, w_raw, N, seed=0):
    """Undirected Louvain community label per node (isolated nodes -> singleton communities)."""
    from networkx.algorithms.community import louvain_communities
    G = _undirected_nx(src, tgt, w_raw, N)
    comms = louvain_communities(G, weight="weight", seed=seed)
    lab = np.full(N, -1, np.int64)
    for ci, nodes in enumerate(comms):
        for n in nodes:
            lab[n] = ci
    # any node the partition somehow missed -> its own community
    nxt = len(comms)
    for v in np.where(lab < 0)[0]:
        lab[v] = nxt; nxt += 1
    return lab


def clustering_undirected(src, tgt, w_raw, N):
    """Undirected local clustering coefficient per node in [0,1] (binary projection)."""
    import networkx as nx
    G = _undirected_nx(src, tgt, w_raw, N)
    cc = nx.clustering(G)  # binary (unweighted) clustering
    out = np.zeros(N, np.float64)
    for v, c in cc.items():
        out[v] = c
    return out


def pagerank_directed(src, tgt, w_raw, N, alpha=0.85):
    """Directed PageRank per node (raw-weighted). Uniform 1/N if the graph is empty."""
    if len(src) == 0:
        return np.full(N, 1.0 / N, np.float64)
    import networkx as nx
    G = nx.DiGraph(); G.add_nodes_from(range(N))
    agg = {}
    for s, t, w in zip(np.asarray(src).tolist(), np.asarray(tgt).tolist(), np.asarray(w_raw, np.float64).tolist()):
        agg[(s, t)] = agg.get((s, t), 0.0) + float(w)
    G.add_weighted_edges_from([(s, t, w) for (s, t), w in agg.items()])
    try:
        pr = nx.pagerank(G, alpha=alpha, weight="weight", max_iter=200, tol=1e-8)
    except nx.PowerIterationFailedConvergence:
        pr = nx.pagerank(G, alpha=alpha, weight="weight", max_iter=1000, tol=1e-6)
    out = np.zeros(N, np.float64)
    for v, p in pr.items():
        out[v] = p
    return out
