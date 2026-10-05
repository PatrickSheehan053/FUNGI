"""
databases_v2.py -- DatabaseLoaderV2 for obj_003_grn_bio_evaluator_v2.

Adapts clone/databases_obj001.py (validated across exp_003/exp_003b). Changes:

1. `panel_genes` is a required, positional argument on every loader and on
   `build_pooled()` -- no caller can evaluate a pair against genes outside
   the GRN's own node set (obj_003 design doc, "Panel gene filtering" section).
2. New TRRUST v2 loader (`load_trrust_pairs`).
3. `build_pooled()` returns the union AND a per-database breakdown dict, so
   per-DB precision (bio_metrics.py's `biological_topk_perdb`) is available
   without a second full reload -- exp_003b's per_database_evaluator.py had
   to load each database in isolation one at a time; this version gets the
   per-db sets for free as a byproduct of building the pool once.
4. CORUM is dropped entirely -- permanently unreachable (OAuth-gated;
   reconfirmed in exp_003b and every prior session) and absent from this
   object's own database table (Data and Dependencies section of the design
   doc lists STRING Network/Physical, ChIP-seq, CollecTRI, KnockTF, TRRUST;
   no CORUM).
5. ENCODE K562/H1-hESC ChIP-seq are stubs: return an empty set with a
   warning log when no source file is staged, per the design doc's explicit
   "do not block the RPE1 implementation" instruction.
6. `build_pbs()` ports exp_003b's validated PBS-set construction
   (Mann-Whitney U on held-out test cells, vectorized per-regulator) so
   `bio_prec_pbs` remains available as a labeled diagnostic (design doc
   Step 6, point 3), without re-deriving that logic from scratch.

string_network/string_physical/chipseq/collectri/knocktf loaders are ported
unchanged in logic (same URLs, same cache file names, same gold_dir/raw/
convention) from databases_obj001.py. Deliberate decision: the design doc's
own YAML example shows STRING v12.0 download URLs, but this object's
backward-compatibility tests (Test 1/2) require bio_prec_raw/stat_prec to
reproduce exp_003b's exact numbers, which were computed against the existing
v11.5 cache shared with obj_001. Re-downloading a newer STRING release here
would silently confound "did TRRUST change the score" with "did the STRING
version change." STRING stays at v11.5; a version bump is a separate,
deliberate decision for later, not a byproduct of this build.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import scipy.stats as stats

STRING_NETWORK_URL = "https://stringdb-static.org/download/protein.links.detailed.v11.5/9606.protein.links.detailed.v11.5.txt.gz"
STRING_PHYSICAL_URL = "https://stringdb-static.org/download/protein.physical.links.detailed.v11.5/9606.protein.physical.links.detailed.v11.5.txt.gz"
STRING_INFO_URL = "https://stringdb-static.org/download/protein.info.v11.5/9606.protein.info.v11.5.txt.gz"
CHIPSEQ_HEPG2_URL = "https://raw.githubusercontent.com/causalbench/causalbench/master/causalscbench/data_access/data/Hep_G2_ChipSeq.csv"
TRRUST_V2_URL = "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv"

KNOWN_DATABASES = ("string_network", "string_physical", "chipseq", "collectri",
                    "knocktf", "trrust_v2")


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _download(url: str, dest: Path, min_bytes: int = 1024) -> Path:
    if dest.exists() and dest.stat().st_size >= min_bytes:
        log(f"  cached: {dest.name} ({dest.stat().st_size:,} bytes)")
        return dest
    log(f"  downloading {url} -> {dest.name}")
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return dest


# ---------------------------------------------------------------------------
# Ported unchanged (logic) from databases_obj001.py
# ---------------------------------------------------------------------------

def load_string_pairs(gold_dir: Path, panel_genes: frozenset, physical: bool) -> set:
    raw_dir = gold_dir / "raw"
    info_path = _download(STRING_INFO_URL, raw_dir / "protein.info.txt.gz")
    links_url = STRING_PHYSICAL_URL if physical else STRING_NETWORK_URL
    links_name = "protein.physical.links.txt.gz" if physical else "protein.links.txt.gz"
    links_path = _download(links_url, raw_dir / links_name, min_bytes=1_000_000)

    info = pd.read_csv(info_path, sep="\t", compression="gzip")
    info_in_panel = info[info["preferred_name"].isin(panel_genes)]
    protein_to_symbol = dict(zip(info_in_panel["#string_protein_id"], info_in_panel["preferred_name"]))
    panel_protein_ids = set(protein_to_symbol)

    links = pd.read_csv(links_path, sep=" ", compression="gzip", usecols=["protein1", "protein2"])
    in_panel = links["protein1"].isin(panel_protein_ids) & links["protein2"].isin(panel_protein_ids)
    links = links.loc[in_panel]
    sym1 = links["protein1"].map(protein_to_symbol)
    sym2 = links["protein2"].map(protein_to_symbol)
    distinct = sym1 != sym2
    pairs = set(zip(sym1[distinct], sym2[distinct])) | set(zip(sym2[distinct], sym1[distinct]))
    kind = "STRING Physical" if physical else "STRING Network"
    log(f"  {kind}: {len(pairs):,} in-panel directed (symmetric) gene pairs "
        f"(panel proteins resolved: {len(panel_protein_ids):,}/{len(panel_genes):,})")
    return pairs


def load_chipseq_pairs(gold_dir: Path, panel_genes: frozenset, source: str = "hepg2_proxy",
                        file_override: str = None) -> set:
    if source != "hepg2_proxy":
        return load_encode_chipseq_stub(gold_dir, panel_genes, source, file_override)
    dest = _download(CHIPSEQ_HEPG2_URL, gold_dir / "raw" / "Hep_G2_ChipSeq.csv")
    df = pd.read_csv(dest)
    in_panel = df["source"].isin(panel_genes) & df["target"].isin(panel_genes)
    df = df.loc[in_panel & (df["source"] != df["target"])]
    pairs = set(zip(df["source"], df["target"]))
    log(f"  ChIP-seq (hepg2_proxy): {len(pairs):,} in-panel directed TF->target pairs")
    return pairs


def load_collectri_pairs(gold_dir: Path, panel_genes: frozenset) -> set:
    cache_path = gold_dir / "raw" / "collectri.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        log(f"  CollecTRI: loaded cached {len(df):,} interactions from {cache_path.name}")
    else:
        try:
            import decoupler as dc
            log("  CollecTRI: downloading via decoupler (dc.op.collectri)...")
            df = dc.op.collectri(organism="human", remove_complexes=False, license="academic")
        except Exception as exc:
            log(f"  CollecTRI download/import failed ({exc}); skipping CollecTRI.")
            return set()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        log(f"  CollecTRI: cached {len(df):,} interactions to {cache_path.name}")
    in_panel = df["source"].isin(panel_genes) & df["target"].isin(panel_genes)
    df = df.loc[in_panel & (df["source"] != df["target"])]
    pairs = set(zip(df["source"], df["target"]))
    log(f"  CollecTRI: {len(pairs):,} in-panel directed TF->target pairs")
    return pairs


def load_knocktf_pairs(gold_dir: Path, panel_genes: frozenset, biosample_filter: str,
                        logfc_threshold: float) -> set:
    safe_name = biosample_filter.replace(" ", "_").replace("/", "_")
    cache_path = gold_dir / "raw" / f"knocktf_{safe_name}.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        log(f"  KnockTF: loaded cached {len(df):,} TF-target rows from {cache_path.name}")
    else:
        try:
            import decoupler as dc
            log(f"  KnockTF: downloading via decoupler (dc.ds.knocktf), filtering to "
                f"Biosample.Name=={biosample_filter!r}...")
            adata = dc.ds.knocktf(thr_fc=None, verbose=False)
        except Exception as exc:
            log(f"  KnockTF download/import failed ({exc}); skipping KnockTF.")
            return set()
        sub = adata[adata.obs["Biosample.Name"] == biosample_filter]
        records = []
        var_names = np.asarray(sub.var_names)
        for i in range(sub.n_obs):
            tf = sub.obs["source"].iloc[i]
            row = np.asarray(sub.X[i]).ravel()
            mask = np.isfinite(row) & (np.abs(row) >= logfc_threshold)
            for gene, lfc in zip(var_names[mask], row[mask]):
                if gene != tf:
                    records.append({"source": tf, "target": gene, "logFC": float(lfc)})
        df = pd.DataFrame(records, columns=["source", "target", "logFC"])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        log(f"  KnockTF: cached {len(df):,} TF-target rows to {cache_path.name}")
    if len(df) == 0:
        return set()
    in_panel = df["source"].isin(panel_genes) & df["target"].isin(panel_genes)
    df = df.loc[in_panel & (df["source"] != df["target"])]
    pairs = set(zip(df["source"], df["target"]))
    log(f"  KnockTF ({biosample_filter}): {len(pairs):,} in-panel directed TF->target pairs")
    return pairs


# ---------------------------------------------------------------------------
# New loaders
# ---------------------------------------------------------------------------

def load_trrust_pairs(gold_dir: Path, panel_genes: frozenset) -> set:
    dest = gold_dir / "raw" / "trrust_rawdata.human.tsv"
    try:
        _download(TRRUST_V2_URL, dest, min_bytes=1000)
    except requests.RequestException as exc:
        log(f"  TRRUST v2 download failed ({exc}); skipping TRRUST.")
        return set()
    df = pd.read_csv(dest, sep="\t", names=["TF", "Target", "Type", "PMIDs"])
    in_panel = df["TF"].isin(panel_genes) & df["Target"].isin(panel_genes)
    sub = df.loc[in_panel & (df["TF"] != df["Target"])]
    pairs = set(zip(sub["TF"], sub["Target"]))
    log(f"  TRRUST v2: {len(pairs):,} in-panel directed TF->target pairs "
        f"({len(df):,} total curated pairs, {df['TF'].nunique()} unique TFs)")
    return pairs


def load_encode_chipseq_stub(gold_dir: Path, panel_genes: frozenset, cell_line: str,
                              file_override: str = None) -> set:
    """ENCODE K562/H1-hESC ChIP-seq: functional once a source file is staged
    at `file_override`; returns an empty set with a warning otherwise. The
    expected format matches the hepg2_proxy CSV (source,target columns)."""
    if file_override and Path(file_override).exists() and Path(file_override).stat().st_size > 0:
        df = pd.read_csv(file_override)
        in_panel = df["source"].isin(panel_genes) & df["target"].isin(panel_genes)
        df = df.loc[in_panel & (df["source"] != df["target"])]
        pairs = set(zip(df["source"], df["target"]))
        log(f"  ChIP-seq ({cell_line}): {len(pairs):,} in-panel directed TF->target pairs")
        return pairs
    log(f"  ChIP-seq ({cell_line}): no source file staged at {file_override!r} -- "
        f"returning empty set (stub; not yet populated for this cell line).")
    return set()


# ---------------------------------------------------------------------------
# Dispatch + orchestration
# ---------------------------------------------------------------------------

def _dispatch(name: str, gold_dir: Path, panel_genes: frozenset, spec: dict) -> set:
    spec = spec or {}
    if name == "string_network":
        return load_string_pairs(gold_dir, panel_genes, physical=False)
    if name == "string_physical":
        return load_string_pairs(gold_dir, panel_genes, physical=True)
    if name == "chipseq":
        return load_chipseq_pairs(gold_dir, panel_genes, source=spec.get("source", "hepg2_proxy"),
                                   file_override=spec.get("file"))
    if name == "collectri":
        return load_collectri_pairs(gold_dir, panel_genes)
    if name == "knocktf":
        return load_knocktf_pairs(gold_dir, panel_genes,
                                   biosample_filter=spec.get("biosample_filter", "hTERT-RPE1"),
                                   logfc_threshold=float(spec.get("logfc_threshold", 1.0)))
    if name == "trrust_v2":
        return load_trrust_pairs(gold_dir, panel_genes)
    log(f"  WARNING: no loader registered for database {name!r}; skipping.")
    return set()


class DatabaseLoaderV2:
    """Cell-type-agnostic. `panel_genes` is required and positional on
    `build_pooled` and `build_pbs` -- no keyword-default escape hatch that
    would let a caller accidentally evaluate against an unfiltered universe."""

    def __init__(self, cell_config_databases: dict, gold_dir: str):
        self.databases = cell_config_databases
        self.gold_dir = Path(gold_dir)

    def build_pooled(self, panel_genes: frozenset) -> dict:
        if not isinstance(panel_genes, (frozenset, set)):
            raise TypeError("panel_genes must be a frozenset/set of gene symbols")
        log(f"Building pooled biological gold standard, filtered to a "
            f"{len(panel_genes):,}-gene panel...")
        per_db = {}
        sources_in_panel = {}
        for name, spec in self.databases.items():
            if not (spec or {}).get("enabled", True):
                continue
            pairs = _dispatch(name, self.gold_dir, panel_genes, spec)
            per_db[name] = pairs
            sources_in_panel[name] = len(pairs)
        pooled = set()
        for pairs in per_db.values():
            pooled |= pairs
        log(f"Pooled set: {len(pooled):,} unique directed pairs "
            f"(per-source in-panel counts: {sources_in_panel})")
        return {"pooled": pooled, "per_db": per_db, "sources_in_panel": sources_in_panel}

    def build_causal_pool(self, per_db: dict, causal_databases, coexpression_databases) -> dict:
        """obj_003.1 causal-evidence panel (Tier 1). Pure set algebra over the
        `per_db` dict already produced by build_pooled() -- NO new loading/downloads.

        Unions the config's `causal_databases` (directed TF->target: CollecTRI,
        TRRUST v2, KnockTF, ChIP-seq) into one directed-causal gold set, and
        separately unions `coexpression_databases` (STRING network/physical) into a
        context-only co-expression set. The two are kept apart so MD-heavy graphs
        cannot be flattered by co-expression overlap when judged on causal evidence.
        Returns causal_pool, coexpr_pool, the per-database causal breakdown, and the
        set of distinct TF sources in the causal pool (the broad-TF coverage basis
        used by bio_metrics.causal_coverage)."""
        causal_pool, coexpr_pool = set(), set()
        for name in causal_databases:
            causal_pool |= per_db.get(name, set())
        for name in coexpression_databases:
            coexpr_pool |= per_db.get(name, set())
        causal_per_db = {n: per_db.get(n, set()) for n in causal_databases}
        causal_sources = {r for (r, _t) in causal_pool}
        log(f"Causal pool: {len(causal_pool):,} directed pairs from {list(causal_databases)} "
            f"({len(causal_sources):,} distinct TF sources); coexpr context pool: "
            f"{len(coexpr_pool):,} pairs from {list(coexpression_databases)}")
        return {"causal_pool": causal_pool, "coexpr_pool": coexpr_pool,
                "causal_per_db": causal_per_db, "causal_sources": causal_sources}

    def build_pbs(self, pooled: set, X: np.ndarray, gene_to_col: dict, pert_values: np.ndarray,
                  test_idx: np.ndarray, control_label: str, min_cells: int,
                  p_threshold: float = 0.05) -> tuple:
        """Ported from exp_003b's pbs_evaluator.py build_pbs_set -- GuanLab's
        PBS construction (Mann-Whitney U, p<0.05, on held-out test cells),
        vectorized per-regulator. Returns (pbs_set, stats_dict)."""
        pert_test = pert_values[test_idx]
        X_test = X[test_idx]
        ctrl_mask_test = pert_test == control_label
        X_ctrl_test = X_test[ctrl_mask_test]

        by_regulator = {}
        for a, b in pooled:
            by_regulator.setdefault(a, []).append(b)

        available_perturbations = set(np.unique(pert_test)) - {control_label}
        candidate_regulators = sorted(set(by_regulator) & available_perturbations)

        pbs_set = set()
        n_tested = 0
        n_passed = 0
        for a in candidate_regulators:
            knock_mask = pert_test == a
            n_knock = int(knock_mask.sum())
            targets = by_regulator[a]
            if n_knock < min_cells:
                continue
            X_knock_test = X_test[knock_mask]
            target_cols, valid_targets = [], []
            for b in targets:
                col = gene_to_col.get(b)
                if col is None:
                    continue
                target_cols.append(col)
                valid_targets.append(b)
            if not target_cols:
                continue
            target_cols = np.asarray(target_cols)
            obs_slice = X_ctrl_test[:, target_cols]
            int_slice = X_knock_test[:, target_cols]
            _, pvals = stats.mannwhitneyu(obs_slice, int_slice, axis=0)
            n_tested += len(valid_targets)
            passed_mask = pvals < p_threshold
            n_passed += int(passed_mask.sum())
            for b, passed in zip(valid_targets, passed_mask):
                if passed:
                    pbs_set.add((a, b))

        n_raw = len(pooled)
        fraction_evaluable = n_tested / n_raw if n_raw else 0.0
        fraction_passed = n_passed / n_tested if n_tested else 0.0
        log(f"PBS set: {len(pbs_set):,} pairs survive p<{p_threshold} out of "
            f"{n_tested:,} evaluable ({fraction_evaluable*100:.2f}% of {n_raw:,} raw pairs)")
        return pbs_set, dict(n_raw_gold_pairs=n_raw, n_evaluable_for_pbs=n_tested,
                              n_pbs=len(pbs_set), fraction_evaluable=round(fraction_evaluable, 6),
                              fraction_passed=round(fraction_passed, 6))
