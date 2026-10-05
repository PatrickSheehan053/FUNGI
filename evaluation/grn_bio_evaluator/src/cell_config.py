"""
cell_config.py -- YAML config loader and validator for obj_003_grn_bio_evaluator_v2.

Deviation from the obj_003 design doc's literal YAML schema: the design doc's
example YAML lists a `file:` path per database, implying gold-standard data
must be pre-staged by hand. Every database loader this project has ever
shipped (obj_001's databases.py, reused unmodified by exp_003b) instead
auto-downloads to a shared, cached `gold_dir` -- a pattern already proven
across three sessions and required for backward-compatibility (Test 1/2 in
the design doc demand bio_prec_raw/stat_prec match exp_003b's values, which
were produced against that exact cache). databases_v2.py keeps the
auto-download-to-gold_dir pattern; this config format reflects that: each
database has `enabled` (and, for sources with a real choice of variant, e.g.
chipseq's cell-line proxy) rather than a literal `file:` path. `gold_dir`
is a top-level field instead.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

ALLOWED_TOP_LEVEL = {
    "cell_line", "sc_input", "gold_dir", "pert_col", "control_label",
    "min_cells_per_perturbation", "split", "stat_metrics", "databases",
    "go_enrichment", "causal_panel",
}
ALLOWED_SPLIT = {"test_size", "random_state", "stratify_col"}
ALLOWED_STAT = {"p_threshold", "min_cells"}
ALLOWED_DB_KEYS = {"enabled", "source", "file", "info_file", "min_score", "note"}
ALLOWED_GO = {"organism", "sources", "min_regulon_size", "significance_threshold",
              "correction_method"}
# obj_003.1 causal-evidence panel (v2.1)
ALLOWED_CAUSAL = {"enabled", "causal_databases", "coexpression_databases", "motif"}
ALLOWED_MOTIF = {"enabled", "ranking_db", "motif2tf", "tss_window", "nes_threshold"}
DEFAULT_CAUSAL_DBS = ["collectri", "trrust_v2", "knocktf", "chipseq"]
DEFAULT_COEXPR_DBS = ["string_network", "string_physical"]
KNOWN_DATABASES = {"string_network", "string_physical", "chipseq", "collectri",
                    "knocktf", "trrust_v2"}


class ConfigError(ValueError):
    pass


@dataclass
class CellConfig:
    cell_line: str
    sc_input: Optional[str]
    gold_dir: str
    pert_col: str
    control_label: str
    min_cells: int
    split_test_size: float
    split_random_state: int
    stat_p_threshold: float
    stat_min_cells: int
    databases: dict = field(default_factory=dict)
    go_config: dict = field(default_factory=dict)
    causal_config: dict = field(default_factory=dict)  # obj_003.1 causal-evidence panel
    source_path: str = ""


def _check_unknown(d: dict, allowed: set, context: str) -> None:
    unknown = set(d) - allowed
    if unknown:
        raise ConfigError(f"{context}: unknown key(s) {sorted(unknown)} -- "
                           f"allowed keys are {sorted(allowed)}. This is almost "
                           f"always a typo; fix the YAML rather than silently "
                           f"ignoring it.")


def load_cell_config(path: str) -> CellConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    _check_unknown(raw, ALLOWED_TOP_LEVEL, "top level")

    sc_input = raw.get("sc_input")
    if sc_input is not None and not Path(sc_input).exists():
        raise ConfigError(f"sc_input={sc_input!r} does not exist on disk. "
                           f"Use sc_input: null for cell lines without staged "
                           f"single-cell data (stat metrics will be skipped).")

    gold_dir = raw.get("gold_dir")
    if not gold_dir:
        raise ConfigError("gold_dir is required (directory for cached/downloaded "
                           "gold-standard database files).")

    split = raw.get("split", {})
    _check_unknown(split, ALLOWED_SPLIT, "split")
    test_size = float(split.get("test_size", 0.2))
    if not (0.0 < test_size < 1.0):
        raise ConfigError(f"split.test_size={test_size} must be in (0, 1).")

    stat = raw.get("stat_metrics", {})
    _check_unknown(stat, ALLOWED_STAT, "stat_metrics")

    databases = raw.get("databases", {})
    unknown_dbs = set(databases) - KNOWN_DATABASES
    if unknown_dbs:
        raise ConfigError(f"databases: unknown database name(s) {sorted(unknown_dbs)} "
                           f"-- known databases are {sorted(KNOWN_DATABASES)}.")
    for name, spec in databases.items():
        _check_unknown(spec or {}, ALLOWED_DB_KEYS, f"databases.{name}")
        file_path = (spec or {}).get("file")
        if file_path and not Path(file_path).exists():
            raise ConfigError(f"databases.{name}.file={file_path!r} does not exist on "
                               f"disk. Either stage the file, remove the override (the "
                               f"loader will auto-download/cache to gold_dir instead), "
                               f"or set enabled: false.")

    go_cfg = raw.get("go_enrichment", {})
    _check_unknown(go_cfg, ALLOWED_GO, "go_enrichment")

    # obj_003.1 causal-evidence panel (v2.1). Backward-compatible: absent block ->
    # panel enabled with the default causal/coexpr DB partition (so re-scoring any
    # existing config gets the panel without editing the YAML).
    causal_raw = raw.get("causal_panel", {}) or {}
    _check_unknown(causal_raw, ALLOWED_CAUSAL, "causal_panel")
    motif_raw = causal_raw.get("motif", {}) or {}
    _check_unknown(motif_raw, ALLOWED_MOTIF, "causal_panel.motif")
    causal_dbs = list(causal_raw.get("causal_databases", DEFAULT_CAUSAL_DBS))
    coexpr_dbs = list(causal_raw.get("coexpression_databases", DEFAULT_COEXPR_DBS))
    bad_causal = (set(causal_dbs) | set(coexpr_dbs)) - KNOWN_DATABASES
    if bad_causal:
        raise ConfigError(f"causal_panel: unknown database name(s) {sorted(bad_causal)} "
                           f"-- known databases are {sorted(KNOWN_DATABASES)}.")
    causal_config = {
        "enabled": bool(causal_raw.get("enabled", True)),
        "causal_databases": causal_dbs,
        "coexpression_databases": coexpr_dbs,
        "motif": {
            "enabled": bool(motif_raw.get("enabled", False)),
            "ranking_db": motif_raw.get("ranking_db"),
            "motif2tf": motif_raw.get("motif2tf"),
            "tss_window": motif_raw.get("tss_window", "10kb"),
            "nes_threshold": float(motif_raw.get("nes_threshold", 3.0)),
        },
    }

    return CellConfig(
        cell_line=raw.get("cell_line", "?"),
        sc_input=sc_input,
        gold_dir=gold_dir,
        pert_col=raw.get("pert_col", "gene"),
        control_label=raw.get("control_label", "non-targeting"),
        min_cells=int(raw.get("min_cells_per_perturbation", 3)),
        split_test_size=test_size,
        split_random_state=int(split.get("random_state", 0)),
        stat_p_threshold=float(stat.get("p_threshold", 0.05)),
        stat_min_cells=int(stat.get("min_cells", 3)),
        databases=databases,
        go_config=go_cfg,
        causal_config=causal_config,
        source_path=str(path),
    )
