"""
motif_metrics.py -- obj_003.1 Tier 2: cisTarget motif-enrichment validator (`motif_prec`).

Perturbation- AND literature-independent orthogonal evidence: for each predicted top-K edge R->T
where R is a TF with a cisTarget motif, test (RcisTarget/ctxcore) whether R's own binding motif is
ENRICHED (NES >= threshold) over R's predicted target set, and whether T is a high-ranked
(leading-edge, rank <= rank_threshold) target of that enriched motif. Reads the aertslab hg38
gene-based ranking DB + the motif->TF annotation table; touches no SHROOM/FUNGI code.

Definition (mirrors causal_prec's denominator=K convention):
  motif_coverage@K = fraction of top-K edges whose Regulator is a TF present in the ranking DB with a
                     mapped motif (the honesty number -- how many edges are even motif-testable).
  motif_prec@K     = fraction of top-K edges that are motif-SUPPORTED: R's regulon (its top-K target
                     set) is motif-validated (best-NES over R's motifs >= nes_threshold) AND T is
                     ranked <= rank_threshold for that best motif.

Validated on real TP53 targets: best-motif NES = 9.01 (>> 3.0). See obj_003.1 SESSION_LOG.
"""
import numpy as np
import pandas as pd

_DB_CACHE = {}
_TF_CACHE = {}


def load_tf_motifs(motif2tf_path: str) -> dict:
    """{TF gene symbol -> set(motif_id)} from the aertslab motif2tf table."""
    if motif2tf_path in _TF_CACHE:
        return _TF_CACHE[motif2tf_path]
    m = pd.read_csv(motif2tf_path, sep="\t", low_memory=False)
    mid_col = "#motif_id" if "#motif_id" in m.columns else "motif_id"
    tf_col = "gene_name"
    tf_to_motifs = {}
    for mid, tf in zip(m[mid_col].astype(str), m[tf_col].astype(str)):
        if tf and tf != "nan":
            tf_to_motifs.setdefault(tf, set()).add(mid)
    _TF_CACHE[motif2tf_path] = tf_to_motifs
    return tf_to_motifs


def _get_db(ranking_db_path: str):
    if ranking_db_path not in _DB_CACHE:
        from ctxcore.rnkdb import FeatherRankingDatabase
        _DB_CACHE[ranking_db_path] = FeatherRankingDatabase(ranking_db_path, name="ranking")
    return _DB_CACHE[ranking_db_path]


def motif_topk(edges_df: pd.DataFrame, k: int, ranking_db_path: str, motif2tf_path: str,
               nes_threshold: float = 3.0, rank_threshold: int = 5000,
               auc_threshold: float = 0.05, min_targets: int = 5) -> dict:
    from ctxcore.genesig import GeneSignature
    from ctxcore.recovery import aucs as calc_aucs

    db = _get_db(ranking_db_path)
    tf_to_motifs = load_tf_motifs(motif2tf_path)
    db_genes = set(db.genes)
    scoreable_tfs = set(tf_to_motifs) & db_genes

    top = edges_df.head(k)
    reg = top["Regulator"].astype(str)
    cov_mask = reg.isin(scoreable_tfs)
    n_scoreable = int(cov_mask.sum())

    supported = 0
    n_validated = 0
    n_nes_tested_edges = 0
    sub = top[cov_mask].copy()
    sub["Regulator"] = sub["Regulator"].astype(str)
    sub["Target"] = sub["Target"].astype(str)
    for R, grp in sub.groupby("Regulator", observed=True):
        target_set = [t for t in grp["Target"] if t in db_genes]
        R_motifs = list(tf_to_motifs.get(R, ()))
        if len(target_set) < min_targets or not R_motifs:
            continue
        gs = GeneSignature(name=R, gene2weight={g: 1.0 for g in target_set})
        rnk = db.load(gs)
        R_motifs = [mid for mid in R_motifs if mid in rnk.index]
        if not R_motifs:
            continue
        weights = np.ones(rnk.shape[1], dtype=np.float64)
        auc = calc_aucs(rnk, db.total_genes, weights, auc_threshold)
        sd = float(auc.std())
        if sd == 0.0:
            continue
        nes = pd.Series((auc - auc.mean()) / sd, index=rnk.index)
        n_nes_tested_edges += len(target_set)
        best = nes[R_motifs].idxmax()
        if float(nes[best]) >= nes_threshold:
            n_validated += 1
            ranks_best = rnk.loc[best]  # genome-wide ranks of this R's target genes for the best motif
            supported += int((ranks_best <= rank_threshold).sum())

    return dict(k=k,
                motif_prec=supported / k if k else float("nan"),
                motif_coverage=n_scoreable / k if k else float("nan"),
                n_motif_supported=int(supported),
                n_scoreable_edges=n_scoreable,
                n_validated_regulons=int(n_validated),
                n_nes_tested_edges=int(n_nes_tested_edges))
