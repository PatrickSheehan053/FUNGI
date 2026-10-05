"""
obj_010 ANASTOMOSIS — metric panel.

NOTE (build deviation, documented in intermediate/BUILD_REPORT.md): the spec names this module `metrics.py`,
but obj_009.3's `metrics_v3f` (imported for the RSC + coexpr_resid axes) internally does a bare
`from metrics import ...` targeting obj_009's OWN `metrics.py` (v1). A module named `metrics` on obj_010's
`src/` path SHADOWS that and causes a circular import. To avoid modifying any obj_009 file (clone-only rule),
the panel lives in `panel.py` instead. Import it as:  `import panel`  (NOT `import metrics`).

This file is intentionally inert (docstring only) so it never registers a `metrics` module that would
re-introduce the collision. See panel.py for the real implementation.
"""
