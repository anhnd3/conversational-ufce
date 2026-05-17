#!/usr/bin/env python3
"""
UFCE Core Module P3 — UFCE-only optimized reproduction runner.

- Optimizes only UFCE1/UFCE2/UFCE3.
- Hard-disables DiCE, DiCE-UF, and AR paths.
- Uses dataset-tuned UFCE defaults with CLI override support.
"""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import os
import platform
import random
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ---- PYTHONPATH bootstrap (Python-native) ----
ROOT = os.path.abspath(os.getcwd())

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ---------------------------------------------

warnings.filterwarnings("ignore")
if plt is not None:
    plt.style.use("seaborn-whitegrid")
pd.set_option("display.max_columns", None)

import ufce
from ufce import UFCE
from ufce.core import cfmethods
from ufce.core import evaluations as eval_module
from ufce.core.data_processing import (
    classify_dataset_getModel,
    get_bank_user_constraints,
    get_grad_user_constraints,
    get_wine_user_constraints,
    get_bupa_user_constraints,
    get_movie_user_constraints,
)

ufc = UFCE()

# ----------------------------
# Author reference values (Table 7) — UFCE only
# ----------------------------
AUTHOR_TABLE7: Dict[str, Dict[str, Dict[str, float]]] = {
    "bank": {
        "UFCE1": {
            "Prox-Jac": 0.60,
            "Prox-Euc": 10.00,
            "Sparsity": 1.00,
            "Actionability": 14.00,
            "Plausibility": 14.00,
            "Feasibility": 14.00,
        },
        "UFCE2": {
            "Prox-Jac": 0.00,
            "Prox-Euc": 23.10,
            "Sparsity": 2.00,
            "Actionability": 30.00,
            "Plausibility": 30.00,
            "Feasibility": 30.00,
        },
        "UFCE3": {
            "Prox-Jac": 0.00,
            "Prox-Euc": 40.12,
            "Sparsity": 3.00,
            "Actionability": 44.00,
            "Plausibility": 43.00,
            "Feasibility": 43.00,
        },
    },
    "grad": {
        "UFCE1": {
            "Prox-Jac": 0.00,
            "Prox-Euc": 2.34,
            "Sparsity": 1.00,
            "Actionability": 8.00,
            "Plausibility": 8.00,
            "Feasibility": 8.00,
        },
        "UFCE2": {
            "Prox-Jac": 0.00,
            "Prox-Euc": 4.85,
            "Sparsity": 2.00,
            "Actionability": 13.00,
            "Plausibility": 13.00,
            "Feasibility": 13.00,
        },
        "UFCE3": {
            "Prox-Jac": 0.00,
            "Prox-Euc": 6.32,
            "Sparsity": 2.80,
            "Actionability": 13.00,
            "Plausibility": 13.00,
            "Feasibility": 13.00,
        },
    },
    "wine": {
        "UFCE1": {
            "Prox-Jac": float("nan"),
            "Prox-Euc": 14.90,
            "Sparsity": 1.00,
            "Actionability": 43.00,
            "Plausibility": 28.00,
            "Feasibility": 28.00,
        },
        "UFCE2": {
            "Prox-Jac": float("nan"),
            "Prox-Euc": 8.45,
            "Sparsity": 2.00,
            "Actionability": 50.00,
            "Plausibility": 41.00,
            "Feasibility": 41.00,
        },
        "UFCE3": {
            "Prox-Jac": float("nan"),
            "Prox-Euc": 21.95,
            "Sparsity": 3.00,
            "Actionability": 50.00,
            "Plausibility": 42.00,
            "Feasibility": 42.00,
        },
    },
    "bupa": {
        "UFCE1": {
            "Prox-Jac": float("nan"),
            "Prox-Euc": 10.00,
            "Sparsity": 1.00,
            "Actionability": 17.00,
            "Plausibility": 15.00,
            "Feasibility": 15.00,
        },
        "UFCE2": {
            "Prox-Jac": float("nan"),
            "Prox-Euc": 9.00,
            "Sparsity": 2.00,
            "Actionability": 15.00,
            "Plausibility": 15.00,
            "Feasibility": 15.00,
        },
        "UFCE3": {
            "Prox-Jac": float("nan"),
            "Prox-Euc": 17.10,
            "Sparsity": 2.90,
            "Actionability": 13.00,
            "Plausibility": 13.00,
            "Feasibility": 13.00,
        },
    },
    "movie": {
        "UFCE1": {
            "Prox-Jac": 0.00,
            "Prox-Euc": 20.00,
            "Sparsity": 1.00,
            "Actionability": 20.00,
            "Plausibility": 8.00,
            "Feasibility": 8.00,
        },
        "UFCE2": {
            "Prox-Jac": 0.00,
            "Prox-Euc": 32.00,
            "Sparsity": 2.00,
            "Actionability": 14.00,
            "Plausibility": 14.00,
            "Feasibility": 14.00,
        },
        "UFCE3": {
            "Prox-Jac": 0.00,
            "Prox-Euc": 43.00,
            "Sparsity": 3.00,
            "Actionability": 18.00,
            "Plausibility": 18.00,
            "Feasibility": 18.00,
        },
    },
}

TUNED_RUN2: Dict[str, Dict[str, int]] = {
    "bank": {"radius": 60, "n_neighbors": 200, "min_act": 1, "min_feas": 1, "ufce_flip_filter": 0},
    "bupa": {"radius": 70, "n_neighbors": 200, "min_act": 1, "min_feas": 1, "ufce_flip_filter": 0},
    "grad": {"radius": 500, "n_neighbors": 400, "min_act": 1, "min_feas": 1, "ufce_flip_filter": 0},
    "wine": {"radius": 7, "n_neighbors": 1000, "min_act": 1, "min_feas": 1, "ufce_flip_filter": 0},
    "movie": {"radius": 160, "n_neighbors": 100, "min_act": 1, "min_feas": 1, "ufce_flip_filter": 0},
}

FINAL_RUNTIME_CONFIG: Dict[str, Dict[str, int]] = {
    "bank": {"radius": 500, "n_neighbors": 1000, "min_act": 0, "min_feas": 0, "ufce_flip_filter": 0},
    "bupa": {"radius": 70, "n_neighbors": 200, "min_act": 1, "min_feas": 1, "ufce_flip_filter": 0},
    "grad": {"radius": 500, "n_neighbors": 400, "min_act": 1, "min_feas": 1, "ufce_flip_filter": 0},
    "wine": {"radius": 7, "n_neighbors": 1000, "min_act": 1, "min_feas": 1, "ufce_flip_filter": 0},
    "movie": {"radius": 80, "n_neighbors": 50, "min_act": 1, "min_feas": 1, "ufce_flip_filter": 0},
}

FINAL_BLINDSPOT_BUNDLE = {
    "bank": {
        "uf_mode": "scaled_up_150",
        "step_mode": "local_reproduction",
        "f2change_mode": "author_public",
    },
    "bupa": {
        "uf_mode": "neutral_all_1",
        "step_mode": "local_reproduction",
        "f2change_mode": "author_public",
    },
    "grad": {
        "uf_mode": "neutral_all_1",
        "step_mode": "local_reproduction",
        "f2change_mode": "author_public",
    },
    "wine": {
        "uf_mode": "neutral_all_1",
        "step_mode": "local_reproduction",
        "f2change_mode": "author_public",
    },
    "movie": {
        "uf_mode": "neutral_all_1",
        "step_mode": "local_reproduction",
        "f2change_mode": "author_public",
    },
}

ALL_DATASETS = ["bank", "bupa", "grad", "wine", "movie"]

TABLE7_AUTHOR_PUBLIC_BUNDLE = {
    dataset: {
        "uf_mode": "author_public",
        "step_mode": "author_public",
        "f2change_mode": "author_public",
    }
    for dataset in ALL_DATASETS
}

AUTHOR_PUBLIC_STEP_CONFIG: Dict[str, Dict[str, float]] = {
    "bank": {
        "Income": 1,
        "Family": 1,
        "CCAvg": 0.1,
        "Education": 1,
        "Mortgage": 1,
        "SecuritiesAccount": 1,
        "CDAccount": 1,
        "Online": 1,
        "CreditCard": 1,
    },
    "bupa": {"Mcv": 1, "Alkphos": 1, "Sgpt": 1, "Sgot": 1, "Gammagt": 1, "Drinks": 1},
    "grad": {
        "GRE Score": 1,
        "TOEFL Score": 1,
        "University Rating": 1,
        "SOP": 1,
        "LOR": 1,
        "CGPA": 0.1,
        "Research": 1,
    },
    "wine": {
        "fixed acidity": 0.5,
        "volatile acidity": 0.10,
        "citric acid": 0.1,
        "residual sugar": 0.5,
        "free sulfur dioxide": 1.0,
        "total sulfur dioxide": 1.0,
        "density": 0.1,
        "pH": 0.5,
        "alcohol": 0.5,
    },
    "movie": {
        "Production_expense": 3,
        "Num_multiplex": 3,
        "Multiplex_coverage": 0.2,
        "Movie_length": 5,
        "Lead_Actor_Rating": 1.0,
        "Lead_Actress_rating": 1.0,
        "Director_rating": 1.0,
        "Producer_rating": 1.0,
        "Genre": 1,
        "Collection": 500,
        "Budget": 3000,
    },
}

AUTHOR_PUBLIC_STEP_SOURCE = {
    "bank": "ufce/core/data_processing.py::get_bank_user_constraints.step",
    "bupa": "ufce/core/data_processing.py::get_bupa_user_constraints.step",
    "grad": "ufce/core/data_processing.py::get_grad_user_constraints.step",
    "wine": "ufce/core/data_processing.py::get_wine_user_constraints.step",
    "movie": "author experiment-layer movie step; Budget=3000 from ufce/core_author/experiments.py and final Table 7 reproduction scripts",
}

AUTHOR_PUBLIC_UF_SOURCE = {
    "bank": "ufce/core/data_processing.py::get_bank_user_constraints.uf",
    "bupa": "ufce/core/data_processing.py::get_bupa_user_constraints.uf",
    "grad": "ufce/core/data_processing.py::get_grad_user_constraints.uf",
    "wine": "ufce/core/data_processing.py::get_wine_user_constraints.uf",
    "movie": "ufce/core/data_processing.py::get_movie_user_constraints.uf",
}

AUTHOR_PUBLIC_F2CHANGE_SOURCE = {
    "bank": "ufce/core/data_processing.py::get_bank_user_constraints.f2change",
    "bupa": "ufce/core/data_processing.py::get_bupa_user_constraints.f2change",
    "grad": "ufce/core/data_processing.py::get_grad_user_constraints.f2change",
    "wine": "ufce/core/data_processing.py::get_wine_user_constraints.f2change",
    "movie": "ufce/core/data_processing.py::get_movie_user_constraints.f2change",
}

METRICS = ["Prox-Jac", "Prox-Euc", "Sparsity", "Actionability", "Plausibility", "Feasibility"]
METHODS = ["UFCE1", "UFCE2", "UFCE3"]
DIAGNOSTIC_DEFAULT_ROOT = os.path.join("outputs", "part1_table7_deviation_diagnostics")
RELATIVE_DELTA_EPSILON = 1.0
RECONSTRUCTION_ATOL = 1e-9
APF_METRICS = {"Actionability", "Plausibility", "Feasibility"}
APF_DENOMINATOR_TYPES = {
    "all_queries",
    "queries_with_raw_output",
    "queries_with_selected_candidate",
    "valid_published_candidates",
}
APF_DENOMINATOR_REQUIRED_FIELDS = [
    "apf_metric_name",
    "apf_pass",
    "apf_denominator_type",
    "apf_eligible_query",
    "apf_has_candidate",
    "apf_has_selected_candidate",
    "apf_counted_in_metric",
    "apf_fold_numerator",
    "apf_fold_denominator",
    "apf_fail_reason",
]
CANDIDATE_ROLE_REQUIRED_FIELDS = [
    "selected_candidate_by_ufce",
    "selected_candidate_used_for_metric",
    "selected_candidate_used_for_force_flip",
    "metric_candidate_type",
    "metric_candidate_id",
    "metric_candidate_selection_stage",
]
METRIC_FAMILY_BY_METRIC = {
    "Prox-Euc": "Distance contract",
    "Prox-Jac": "Changed-feature-set",
    "Sparsity": "Changed-feature-set",
    "Actionability": "APF contract",
    "Plausibility": "APF contract",
    "Feasibility": "APF contract",
}
TRACE_TYPES_BY_FAMILY = {
    "Distance contract": "per_query_metric_trace,feature_contribution_trace,candidate_selection_trace",
    "Changed-feature-set": "per_query_metric_trace,feature_contribution_trace,candidate_selection_trace",
    "APF contract": "apf_component_trace,per_query_metric_trace,candidate_selection_trace",
}
LOCKED_CONFIG_SOURCE = "FINAL_RUNTIME_CONFIG + TABLE7_AUTHOR_PUBLIC_BUNDLE in scripts/final/part1/01b_reproduce_ufce_only.py"
HYPER_TUNING_SOURCE = "docs/thesis/part1/2026-03-02_ufce_hypertuning_and_movie_proxeuc_report.md and scripts/final/part1/02_tune_final_parameters.py"
HYPER_TUNING_RUN_ID = "run2_plus_final_freeze_documented_20260425"
HYPER_TUNING_SELECTION_CRITERION = "valid_config first, score_max_penalized ascending, score_mean ascending; final_freeze runtime reused with table7_author_public bundle"
LOCKED_CONFIG_SELECTED_AT = "2026-04-25"
LOCKED_CONFIG_CLAIM_BOUNDARY = (
    "Locked runtime values were selected before the final Table 7 diagnostic run and then reused unchanged "
    "with the table7_author_public bundle across datasets/variants. Remaining deviations are analyzed, not tuned away."
)


@dataclass
class FoldResult:
    fold_name: str
    means: Dict[str, Dict[str, float]]
    times: Dict[str, float]
    diagnostics: Optional[Dict[str, List[Dict[str, object]]]] = None


@dataclass
class BundleResolution:
    requested_bundle_mode: str
    effective_bundle_mode: str
    bundle_cfg: Dict[str, str]
    uf: Dict[str, Any]
    f2change: List[str]
    step: Dict[str, float]
    effective_uf_source: str
    effective_f2change_source: str
    effective_step_source: str
    fallback_used: bool
    fallback_reason: str
    not_main_table7: bool


def safe_version(pkg_name: str) -> str:
    try:
        from importlib.metadata import version

        return version(pkg_name)
    except Exception:
        return "unknown"


def print_env_versions() -> None:
    pkgs = [
        ("python", sys.version.split()[0]),
        ("platform", platform.platform()),
        ("numpy", safe_version("numpy")),
        ("pandas", safe_version("pandas")),
        ("scipy", safe_version("scipy")),
        ("scikit-learn", safe_version("scikit-learn")),
        ("matplotlib", safe_version("matplotlib")),
        ("ufce", getattr(ufce, "__version__", "unknown")),
    ]
    print("\n=== Environment / Library Versions ===")
    for k, v in pkgs:
        print(f"- {k:12s}: {v}")
    print("======================================\n")


def get_dataset_constraints(dataset: str, datasetdf: pd.DataFrame):
    if dataset == "bank":
        return get_bank_user_constraints(datasetdf)
    if dataset == "grad":
        return get_grad_user_constraints(datasetdf)
    if dataset == "wine":
        return get_wine_user_constraints(datasetdf)
    if dataset == "bupa":
        return get_bupa_user_constraints(datasetdf)
    if dataset == "movie":
        return get_movie_user_constraints(datasetdf)
    raise ValueError(f"Unsupported dataset: {dataset}")


def get_step_config(dataset: str) -> Dict[str, float]:
    if dataset == "bank":
        return {
            "Income": 1,
            "Family": 1,
            "CCAvg": 0.1,
            "Education": 1,
            "Mortgage": 1,
            "SecuritiesAccount": 1,
            "CDAccount": 1,
            "Online": 1,
            "CreditCard": 1,
        }
    if dataset == "grad":
        return {
            "GRE Score": 1,
            "TOEFL Score": 1,
            "University Rating": 1,
            "SOP": 1,
            "LOR": 1,
            "CGPA": 0.1,
            "Research": 1,
        }
    if dataset == "wine":
        return {
            "fixed acidity": 0.5,
            "volatile acidity": 0.10,
            "citric acid": 0.1,
            "residual sugar": 0.5,
            "free sulfur dioxide": 1.0,
            "total sulfur dioxide": 1.0,
            "density": 0.1,
            "pH": 0.5,
            "alcohol": 0.5,
        }
    if dataset == "bupa":
        return {"Mcv": 1, "Alkphos": 1, "Sgpt": 1, "Sgot": 1, "Gammagt": 1, "Drinks": 1}
    if dataset == "movie":
        return {
            "Production_expense": 3,
            "Num_multiplex": 3,
            "Multiplex_coverage": 0.2,
            "Movie_length": 5,
            "Lead_Actor_Rating": 1.0,
            "Lead_Actress_rating": 1.0,
            "Director_rating": 1.0,
            "Producer_rating": 1.0,
            "Genre": 1,
            "Collection": 500,
            "Budget": 3000,
        }
    return {}


def map_nested_numeric(obj, fn):
    if isinstance(obj, dict):
        return {k: map_nested_numeric(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [map_nested_numeric(v, fn) for v in obj]
    if isinstance(obj, tuple):
        return tuple(map_nested_numeric(v, fn) for v in obj)
    if isinstance(obj, (int, float, np.integer, np.floating)):
        return fn(float(obj))
    return obj


def apply_uf_mode(author_uf: Dict, mode: str) -> Dict:
    if mode == "author_public":
        return copy.deepcopy(author_uf)
    if mode == "neutral_all_1":
        return map_nested_numeric(author_uf, lambda _x: 1.0)
    if mode == "scaled_up_150":
        return map_nested_numeric(author_uf, lambda x: x * 1.5)
    raise ValueError(f"Unsupported uf_mode: {mode}")


def apply_f2change_mode(author_f2change: List[str], mode: str) -> List[str]:
    if mode == "author_public":
        return list(author_f2change)
    raise ValueError(f"Unsupported f2change_mode for final reproduction: {mode}")


def get_author_public_step_config(dataset: str) -> Dict[str, float]:
    if dataset not in AUTHOR_PUBLIC_STEP_CONFIG:
        raise ValueError(f"Author/public step is unavailable for dataset={dataset}.")
    return copy.deepcopy(AUTHOR_PUBLIC_STEP_CONFIG[dataset])


def resolve_step_config(
    *,
    dataset: str,
    step_mode: str,
    f2change: Sequence[str],
) -> Tuple[Dict[str, float], str, bool, str]:
    if step_mode == "local_reproduction":
        step = get_step_config(dataset)
        source = "scripts/final/part1/01b_reproduce_ufce_only.py::get_step_config local_reproduction"
        fallback_used = False
        fallback_reason = ""
    elif step_mode == "author_public":
        step = get_author_public_step_config(dataset)
        source = AUTHOR_PUBLIC_STEP_SOURCE.get(dataset, "author_public_step_source_missing")
        fallback_used = False
        fallback_reason = ""
    else:
        raise ValueError(f"Unsupported step_mode: {step_mode}")

    missing = [feature for feature in f2change if feature not in step]
    if missing:
        missing_csv = ",".join(str(feature) for feature in missing)
        raise ValueError(
            f"step_source=public_preset_unresolved fallback_used=true "
            f"fallback_reason=missing_step_keys:{missing_csv}"
        )
    return step, source, fallback_used, fallback_reason


def _canonical_bundle_mode(bundle_mode: str) -> str:
    if bundle_mode == "author_public":
        return "table7_author_public"
    return str(bundle_mode)


def validate_main_table7_bundle(dataset: str, effective_bundle_mode: str, bundle_cfg: Dict[str, str]) -> None:
    if effective_bundle_mode == "final_blindspot_best":
        raise ValueError("Main Table 7 reproduction cannot use bundle_mode=final_blindspot_best.")
    uf_mode = str(bundle_cfg.get("uf_mode", ""))
    if dataset == "bank" and uf_mode == "scaled_up_150":
        raise ValueError("Main Table 7 reproduction cannot use Bank uf_mode=scaled_up_150.")
    if uf_mode == "neutral_all_1":
        raise ValueError("Main Table 7 reproduction cannot use uf_mode=neutral_all_1.")


def resolve_bundle_config(
    *,
    dataset: str,
    args,
    author_uf: Dict[str, Any],
    author_f2change: Sequence[str],
) -> BundleResolution:
    requested_bundle_mode = str(args.bundle_mode)
    effective_bundle_mode = _canonical_bundle_mode(requested_bundle_mode)
    if effective_bundle_mode == "table7_author_public":
        bundle_cfg = copy.deepcopy(TABLE7_AUTHOR_PUBLIC_BUNDLE[dataset])
    elif effective_bundle_mode == "final_blindspot_best":
        bundle_cfg = copy.deepcopy(FINAL_BLINDSPOT_BUNDLE[dataset])
    else:
        raise ValueError(f"Unsupported bundle_mode: {requested_bundle_mode}")

    not_main_table7 = bool(effective_bundle_mode == "final_blindspot_best")
    if not not_main_table7:
        validate_main_table7_bundle(dataset, effective_bundle_mode, bundle_cfg)

    uf = apply_uf_mode(author_uf, bundle_cfg["uf_mode"])
    f2change = apply_f2change_mode(list(author_f2change), bundle_cfg["f2change_mode"])
    step, step_source, fallback_used, fallback_reason = resolve_step_config(
        dataset=dataset,
        step_mode=bundle_cfg["step_mode"],
        f2change=f2change,
    )

    if bundle_cfg["uf_mode"] == "author_public":
        uf_source = AUTHOR_PUBLIC_UF_SOURCE.get(dataset, "author_public_uf_source_missing")
    else:
        uf_source = f"blindspot_diagnostic:{bundle_cfg['uf_mode']}"
    if bundle_cfg["f2change_mode"] == "author_public":
        f2change_source = AUTHOR_PUBLIC_F2CHANGE_SOURCE.get(dataset, "author_public_f2change_source_missing")
    else:
        f2change_source = f"blindspot_diagnostic:{bundle_cfg['f2change_mode']}"

    missing_sources = [
        name
        for name, value in {
            "effective_uf_source": uf_source,
            "effective_f2change_source": f2change_source,
            "effective_step_source": step_source,
        }.items()
        if not value or str(value).endswith("_missing")
    ]
    if missing_sources and not fallback_used:
        raise ValueError("Author/public source resolution failed: " + ",".join(missing_sources))

    return BundleResolution(
        requested_bundle_mode=requested_bundle_mode,
        effective_bundle_mode=effective_bundle_mode,
        bundle_cfg=bundle_cfg,
        uf=uf,
        f2change=f2change,
        step=step,
        effective_uf_source=uf_source,
        effective_f2change_source=f2change_source,
        effective_step_source=step_source,
        fallback_used=bool(fallback_used),
        fallback_reason=str(fallback_reason),
        not_main_table7=not_main_table7,
    )


def print_feature_relations(X: pd.DataFrame, features: List[str], mi_pairs: List) -> None:
    print("=== Feature Relations ===")
    print(f"- Top-5 MI feature pairs: {mi_pairs[:5]}")
    Xf = X[features].copy()
    Xnum = Xf.select_dtypes(include=[np.number])
    if Xnum.shape[1] == 0:
        print("- No numeric columns found for Pearson correlation.")
        print("=========================\n")
        return

    corr = Xnum.corr()
    abs_corr = corr.abs()
    np.fill_diagonal(abs_corr.values, 0.0)
    stacked = abs_corr.stack().sort_values(ascending=False)
    top = stacked.head(10)

    print("- Top-10 |Pearson corr| pairs (numeric-only):")
    printed = 0
    for (a, b), v in top.items():
        if a < b:
            print(f"  {a} <-> {b}: {v:.2f}")
            printed += 1
        if printed >= 10:
            break
    print("=========================\n")


def resolve_effective_cfg(dataset: str, args) -> Dict[str, int]:
    config_source = FINAL_RUNTIME_CONFIG if args.runtime_profile == "final_freeze" else TUNED_RUN2

    if dataset not in config_source:
        allowed = ", ".join(sorted(config_source.keys()))
        raise ValueError(f"Dataset '{dataset}' missing from runtime profile. Allowed: {allowed}")

    tuned = config_source[dataset]
    return {
        "radius": int(args.radius) if args.radius is not None else int(tuned["radius"]),
        "n_neighbors": int(args.n_neighbors) if args.n_neighbors is not None else int(tuned["n_neighbors"]),
        "min_act": int(args.min_act) if args.min_act is not None else int(tuned["min_act"]),
        "min_feas": int(args.min_feas) if args.min_feas is not None else int(tuned["min_feas"]),
        "ufce_flip_filter": (
            int(args.ufce_flip_filter) if args.ufce_flip_filter is not None else int(tuned["ufce_flip_filter"])
        ),
    }


def effective_runtime_fields(dataset: str, args) -> Dict[str, object]:
    cfg = resolve_effective_cfg(dataset, args)
    return {
        "effective_radius": int(cfg["radius"]),
        "effective_n_neighbors": int(cfg["n_neighbors"]),
        "effective_min_act": int(cfg["min_act"]),
        "effective_min_feas": int(cfg["min_feas"]),
        "effective_ufce_flip_filter": int(cfg["ufce_flip_filter"]),
    }


def effective_config_record(
    *,
    dataset: str,
    cfg: Dict[str, int],
    bundle: BundleResolution,
    runtime_profile: str,
) -> Dict[str, object]:
    return {
        "dataset": dataset,
        "runtime_profile": runtime_profile,
        "requested_bundle_mode": bundle.requested_bundle_mode,
        "effective_bundle_mode": bundle.effective_bundle_mode,
        "effective_radius": int(cfg["radius"]),
        "effective_n_neighbors": int(cfg["n_neighbors"]),
        "effective_min_act": int(cfg["min_act"]),
        "effective_min_feas": int(cfg["min_feas"]),
        "effective_ufce_flip_filter": int(cfg["ufce_flip_filter"]),
        "effective_uf_source": bundle.effective_uf_source,
        "effective_f2change_source": bundle.effective_f2change_source,
        "effective_step_source": bundle.effective_step_source,
        "fallback_used": bool(bundle.fallback_used),
        "fallback_reason": bundle.fallback_reason,
        "not_main_table7": bool(bundle.not_main_table7),
    }


def build_effective_manifest_config(args) -> Dict[str, Dict[str, object]]:
    datasets = ALL_DATASETS if getattr(args, "dataset", "bank") == "all" else [str(args.dataset)]
    out: Dict[str, Dict[str, object]] = {}
    effective_bundle_mode = _canonical_bundle_mode(str(args.bundle_mode))
    for dataset in datasets:
        record = effective_runtime_fields(dataset, args)
        if effective_bundle_mode == "final_blindspot_best":
            uf_source = f"blindspot_diagnostic:{FINAL_BLINDSPOT_BUNDLE[dataset]['uf_mode']}"
            f2change_source = f"blindspot_diagnostic:{FINAL_BLINDSPOT_BUNDLE[dataset]['f2change_mode']}"
            step_source = "blindspot_diagnostic:local_reproduction_step"
        else:
            uf_source = AUTHOR_PUBLIC_UF_SOURCE.get(dataset, "author_public_uf_source_missing")
            f2change_source = AUTHOR_PUBLIC_F2CHANGE_SOURCE.get(dataset, "author_public_f2change_source_missing")
            step_source = AUTHOR_PUBLIC_STEP_SOURCE.get(dataset, "author_public_step_source_missing")
        record.update(
            {
                "requested_bundle_mode": str(args.bundle_mode),
                "effective_bundle_mode": effective_bundle_mode,
                "not_main_table7": bool(effective_bundle_mode == "final_blindspot_best"),
                "effective_uf_source": uf_source,
                "effective_f2change_source": f2change_source,
                "effective_step_source": step_source,
                "fallback_used": False,
                "fallback_reason": "",
            }
        )
        out[dataset] = record
    return out


def _has_mad_scaler(scaler: Optional[Dict]) -> bool:
    return isinstance(scaler, dict) and "medians" in scaler and "mads" in scaler


def _mad_transform(df: pd.DataFrame, scaler: Dict, numf: List[str]) -> pd.DataFrame:
    out = df.copy()
    cols = [c for c in numf if c in out.columns]
    if len(cols) == 0:
        return out
    med = scaler["medians"].reindex(cols)
    mad = scaler["mads"].reindex(cols).replace(0, 1.0)
    out.loc[:, cols] = (out.loc[:, cols] - med) / mad
    return out


def _mad_inverse_inplace(df: pd.DataFrame, scaler: Dict, numf: List[str]) -> None:
    cols = [c for c in numf if c in df.columns]
    if len(cols) == 0:
        return
    med = scaler["medians"].reindex(cols)
    mad = scaler["mads"].reindex(cols).replace(0, 1.0)
    df.loc[:, cols] = (df.loc[:, cols] * mad) + med


def build_movie_distance_scaler(
    datasetdf: pd.DataFrame,
    features: List[str],
    numf: List[str],
    outcome_label: str,
) -> Dict[str, object]:
    """
    Build movie-only affine scaler:
      scaled = (raw - min) / ((max - min) / 100)
    Constant columns are pinned to 0 in scaled space.
    """
    cols = [c for c in numf if c in features and c in datasetdf.columns and c != outcome_label]
    if len(cols) == 0:
        raise ValueError("Movie distance scaler could not find numeric columns to scale.")

    base = datasetdf.loc[:, features].copy()
    mins = base[cols].min()
    maxs = base[cols].max()
    ranges = maxs - mins
    constant_cols = [c for c in cols if float(ranges[c]) == 0.0]

    mads = (ranges / 100.0).replace(0.0, 1.0)
    medians = mins
    return {
        "kind": "movie_minmax_0_100",
        "scale_cols": cols,
        "medians": medians.astype(float),
        "mads": mads.astype(float),
        "mins": mins.astype(float),
        "maxs": maxs.astype(float),
        "constant_cols": constant_cols,
    }


def apply_distance_scaler(df: pd.DataFrame, scaler: Optional[Dict[str, object]]) -> pd.DataFrame:
    if scaler is None:
        return df.copy()
    out = df.copy()
    scale_cols = scaler.get("scale_cols", None)
    if scale_cols is None:
        scale_cols = list(getattr(scaler.get("medians", pd.Series(dtype=float)), "index", []))
    cols = [c for c in scale_cols if c in out.columns]
    if len(cols) == 0:
        return out
    out.loc[:, cols] = (out.loc[:, cols] - scaler["medians"].reindex(cols)) / scaler["mads"].reindex(cols)
    constant_cols = [c for c in scaler.get("constant_cols", []) if c in cols]
    if len(constant_cols) != 0:
        out.loc[:, constant_cols] = 0.0
    return out


def _active_ufce_instance() -> UFCE:
    if getattr(cfmethods, "ufc", None) is not None:
        return cfmethods.ufc
    if getattr(eval_module, "ufc", None) is not None:
        return eval_module.ufc
    return ufc


def _scalar(v) -> float:
    arr = np.asarray(v).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return float(arr[0])


def _mean_or_nan(values: List[float]) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=float)))


def _shared_numeric_features(numf: List[str], *dfs: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in numf:
        if all(df is not None and c in df.columns for df in dfs):
            cols.append(c)
    return cols


def _row_l2_distance(lhs: pd.DataFrame, rhs: pd.DataFrame, cols: List[str]) -> float:
    if lhs is None or rhs is None or lhs.empty or rhs.empty or len(cols) == 0:
        return float("nan")
    left = lhs.iloc[0][cols].to_numpy(dtype=float)
    right = rhs.iloc[0][cols].to_numpy(dtype=float)
    return float(np.linalg.norm(left - right))


def _fmt_dbg_float(val: float) -> str:
    try:
        fv = float(val)
    except Exception:
        return "nan"
    if not np.isfinite(fv):
        return "nan"
    return f"{fv:.6f}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return [_json_safe(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return _json_safe(value.to_dict())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.ndarray,)):
        return _json_safe(value.tolist())
    return value


def _finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _write_json(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False, sort_keys=True)


def _write_jsonl(path: str, rows: Iterable[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")


def _sha256_file(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _git_value(args: Sequence[str]) -> str:
    try:
        import subprocess

        out = subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return "unknown"


def _arg_present(*names: str) -> bool:
    for arg in sys.argv[1:]:
        for name in names:
            if arg == name or arg.startswith(name + "="):
                return True
    return False


def normalize_mi_feature_pairs(mi_pairs: Iterable[Iterable[Any]], feature_order: Sequence[str]) -> List[List[str]]:
    """Stabilize UFCE MI pairs by preserving the dataset feature order inside each pair."""
    positions = {str(feature): idx for idx, feature in enumerate(feature_order)}
    fallback = len(positions)
    normalized: List[List[str]] = []
    for pair in mi_pairs:
        items = [str(feature) for feature in list(pair)]
        normalized.append(sorted(items, key=lambda feature: (positions.get(feature, fallback), feature)))
    return normalized


def _candidate_id(dataset: str, fold_name: str, query_pos: int, method: str, role: str, rank: int = 0) -> str:
    return f"{dataset}|{fold_name}|q{int(query_pos)}|{method}|{role}|{int(rank)}"


def _candidate_records(
    df: Any,
    *,
    dataset: str,
    fold_name: str,
    query_pos: int,
    method: str,
    role: str,
    top_k: int,
) -> List[Dict[str, object]]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    records: List[Dict[str, object]] = []
    for rank, row in enumerate(df.head(int(top_k)).to_dict(orient="records")):
        records.append(
            {
                "candidate_id": _candidate_id(dataset, fold_name, query_pos, method, role, rank),
                "candidate_rank": int(rank),
                "candidate_role": role,
                "candidate_profile_raw_json": _json_safe(row),
            }
        )
    return records


def _predict_label_and_probability(model: Any, row_df: pd.DataFrame) -> Tuple[Optional[int], Optional[float]]:
    if not isinstance(row_df, pd.DataFrame) or row_df.empty:
        return None, None
    try:
        pred_arr = np.asarray(model.predict(row_df)).reshape(-1)
        pred = int(pred_arr[0]) if pred_arr.size else None
    except Exception:
        pred = None
    prob = None
    try:
        probs = np.asarray(model.predict_proba(row_df)).reshape(1, -1)
        if probs.size:
            prob = float(np.max(probs[0]))
    except Exception:
        prob = None
    return pred, prob


def _changed_feature_details(
    factual: pd.DataFrame,
    candidate: pd.DataFrame,
    features: Sequence[str],
    *,
    atol: float = 1e-5,
) -> Tuple[List[str], Dict[str, Dict[str, object]]]:
    changed: List[str] = []
    details: Dict[str, Dict[str, object]] = {}
    if not isinstance(factual, pd.DataFrame) or factual.empty or not isinstance(candidate, pd.DataFrame) or candidate.empty:
        return changed, details
    left = factual.iloc[0]
    right = candidate.iloc[0]
    for feature in features:
        if feature not in left.index or feature not in right.index:
            continue
        before = left[feature]
        after = right[feature]
        try:
            raw_delta = float(after) - float(before)
            changed_flag = not np.isclose(float(before), float(after), atol=atol, rtol=0.0)
        except Exception:
            raw_delta = None
            changed_flag = str(before) != str(after)
        if changed_flag:
            changed.append(str(feature))
            details[str(feature)] = {
                "from": _json_safe(before),
                "to": _json_safe(after),
                "raw_delta": raw_delta,
            }
    return changed, details


def _feature_contributions(
    factual: pd.DataFrame,
    candidate: pd.DataFrame,
    numf: Sequence[str],
    distance_scaler: Optional[Dict[str, object]],
) -> Tuple[Optional[float], Dict[str, Dict[str, object]], List[str], str]:
    if not isinstance(factual, pd.DataFrame) or factual.empty or not isinstance(candidate, pd.DataFrame) or candidate.empty:
        return None, {}, [], "none"
    cols = [c for c in numf if c in factual.columns and c in candidate.columns]
    if not cols:
        return None, {}, [], "none"
    factual_dist = apply_distance_scaler(factual.loc[:, cols], distance_scaler) if distance_scaler is not None else factual.loc[:, cols]
    candidate_dist = apply_distance_scaler(candidate.loc[:, cols], distance_scaler) if distance_scaler is not None else candidate.loc[:, cols]
    contributions: Dict[str, Dict[str, object]] = {}
    total_sq = 0.0
    for col in cols:
        before = factual[col].iloc[0]
        after = candidate[col].iloc[0]
        try:
            raw_delta = float(after) - float(before)
            normalized_delta = float(candidate_dist[col].iloc[0]) - float(factual_dist[col].iloc[0])
            squared = float(normalized_delta * normalized_delta)
            total_sq += squared
            contribution = float(abs(normalized_delta))
        except Exception:
            raw_delta = None
            normalized_delta = None
            squared = None
            contribution = None
        contributions[str(col)] = {
            "from": _json_safe(before),
            "to": _json_safe(after),
            "raw_delta": raw_delta,
            "normalized_delta": normalized_delta,
            "squared_contribution": squared,
            "euc_contribution": contribution,
        }
    return float(np.sqrt(total_sq)), contributions, cols, "movie_minmax_0_100" if distance_scaler is not None else "raw"


def _safe_actionability_pair(
    active_ufc: UFCE,
    factual: pd.DataFrame,
    candidate: pd.DataFrame,
    features: Sequence[str],
    f2change: Sequence[str],
    uf: Dict[str, float],
) -> Tuple[bool, str, int, int, List[str], List[str]]:
    if not isinstance(candidate, pd.DataFrame) or candidate.empty:
        return False, "no_selected_candidate", 0, 0, [], []
    changed, _ = _changed_feature_details(factual, candidate, features, atol=getattr(active_ufc, "atol", 1e-5))
    actionable_changed = [f for f in changed if f in f2change]
    non_actionable = [f for f in changed if f not in f2change]
    try:
        cfs, _flag, _ids, _temp = active_ufc.actionability(
            candidate.copy(),
            factual.copy(),
            list(features),
            list(f2change),
            0,
            uf,
            method="other",
        )
        passed = bool(len(cfs) > 0)
    except Exception as exc:
        return False, f"actionability_exception:{type(exc).__name__}", len(actionable_changed), len(changed), actionable_changed, non_actionable
    if passed:
        return True, "", len(actionable_changed), len(changed), actionable_changed, non_actionable
    if len(actionable_changed) < int(getattr(active_ufc, "min_actionable_other", 0)):
        return False, "actionability_threshold_fail", len(actionable_changed), len(changed), actionable_changed, non_actionable
    return False, "actionability_limit_fail", len(actionable_changed), len(changed), actionable_changed, non_actionable


def _safe_plausibility_pair(
    active_ufc: UFCE,
    method: str,
    factual: pd.DataFrame,
    candidate: pd.DataFrame,
    xtrain: pd.DataFrame,
) -> Tuple[bool, str, Dict[str, object]]:
    if not isinstance(candidate, pd.DataFrame) or candidate.empty:
        return False, "no_selected_candidate", {}
    try:
        details = active_ufc.implausibility(
            candidate.copy(),
            factual.copy(),
            xtrain.copy(),
            len(candidate),
            0,
            method_name=method,
            return_details=True,
        )
        passed = int(details.get("count", 0)) > 0
        reason = "" if passed else "lof_outlier"
        return passed, reason, dict(details)
    except Exception as exc:
        return False, f"plausibility_exception:{type(exc).__name__}", {}


def _safe_feasibility_pair(
    active_ufc: UFCE,
    factual: pd.DataFrame,
    candidate: pd.DataFrame,
    xtrain: pd.DataFrame,
    features: Sequence[str],
    f2change: Sequence[str],
    bb_model: Any,
    desired_outcome: float,
    uf: Dict[str, float],
) -> Tuple[bool, str, Dict[str, object]]:
    if not isinstance(candidate, pd.DataFrame) or candidate.empty:
        return False, "no_selected_candidate", {}
    try:
        out = active_ufc.feasibility(
            factual.copy(),
            candidate.copy(),
            xtrain.copy(),
            list(features),
            list(f2change),
            bb_model,
            desired_outcome,
            uf,
            0,
            method="other",
            return_details=True,
        )
        details = out[2] if isinstance(out, tuple) and len(out) >= 3 else {}
        passed = int(details.get("count", 0)) > 0
        reason = "" if passed else "feasibility_fail"
        pair_details = details.get("pair_details", []) if isinstance(details, dict) else []
        if pair_details and isinstance(pair_details[0], dict):
            reason = "" if passed else str(pair_details[0].get("reason", reason))
        return passed, reason, dict(details) if isinstance(details, dict) else {}
    except Exception as exc:
        return False, f"feasibility_exception:{type(exc).__name__}", {}


def _relative_delta(author_value: Any, abs_delta: Any) -> Optional[float]:
    author = _finite_float(author_value)
    delta = _finite_float(abs_delta)
    if author is None or delta is None:
        return None
    return float(delta / max(abs(author), RELATIVE_DELTA_EPSILON))


def build_table7_delta_rows(dataset: str, mean_df: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    author = AUTHOR_TABLE7.get(dataset, {})
    for method in METHODS:
        for metric in METRICS:
            author_value = author.get(method, {}).get(metric, float("nan"))
            reproduced_value = mean_df.loc[metric, method] if metric in mean_df.index and method in mean_df.columns else float("nan")
            author_f = _finite_float(author_value)
            reproduced_f = _finite_float(reproduced_value)
            available = author_f is not None and reproduced_f is not None
            missing_reason = ""
            delta = None
            abs_delta = None
            relative_delta = None
            selection_score = None
            if available:
                delta = float(reproduced_f - author_f)
                abs_delta = abs(delta)
                relative_delta = _relative_delta(author_f, abs_delta)
                selection_score = relative_delta
            else:
                if author_f is None:
                    missing_reason = "author_value_missing_or_not_applicable"
                elif reproduced_f is None:
                    missing_reason = "reproduced_value_missing_or_not_applicable"
            rows.append(
                {
                    "dataset": dataset,
                    "ufce_variant": method,
                    "metric_name": metric,
                    "author_value": author_f,
                    "reproduced_value": reproduced_f,
                    "delta": delta,
                    "abs_delta": abs_delta,
                    "relative_delta": relative_delta,
                    "selection_score": selection_score,
                    "metric_available": bool(available),
                    "missing_reason": missing_reason,
                }
            )
    return rows


def build_metric_delta_summary(delta_rows: Sequence[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(delta_rows)
    if df.empty:
        return pd.DataFrame()
    available = df[df["metric_available"] == True].copy()
    if available.empty:
        return pd.DataFrame(columns=["dataset", "metric_name", "metric_family", "max_abs_delta", "mean_abs_delta", "max_relative_delta", "mean_relative_delta", "affected_variants"])
    available["metric_family"] = available["metric_name"].map(METRIC_FAMILY_BY_METRIC)
    grouped = available.groupby(["dataset", "metric_name", "metric_family"], dropna=False)
    return grouped.agg(
        max_abs_delta=("abs_delta", "max"),
        mean_abs_delta=("abs_delta", "mean"),
        max_relative_delta=("relative_delta", "max"),
        mean_relative_delta=("relative_delta", "mean"),
        affected_variants=("ufce_variant", lambda s: int(s.nunique())),
    ).reset_index()


def select_top_metric_targets(delta_rows: Sequence[Dict[str, object]], top_n: int) -> List[Dict[str, object]]:
    df = pd.DataFrame(delta_rows)
    if df.empty:
        return []
    df = df[df["metric_available"] == True].copy()
    if df.empty:
        return []
    df["metric_family"] = df["metric_name"].map(METRIC_FAMILY_BY_METRIC)
    out_rows: List[Dict[str, object]] = []
    for dataset, ds_df in df.groupby("dataset", sort=False):
        family_rows: List[Dict[str, object]] = []
        for family, fam_df in ds_df.groupby("metric_family", sort=False):
            fam_df = fam_df.copy()
            fam_df["selection_score_component"] = pd.to_numeric(fam_df["selection_score"], errors="coerce")
            finite = fam_df[np.isfinite(fam_df["selection_score_component"].to_numpy(dtype=float))]
            if finite.empty:
                continue
            max_score = float(finite["selection_score_component"].max())
            mean_score = float(finite["selection_score_component"].mean())
            affected_variants = int(finite.loc[finite["abs_delta"].astype(float) > 0.0, "ufce_variant"].nunique())
            family_score = float(max_score + 0.5 * mean_score + 0.2 * affected_variants)
            primary = finite.sort_values(
                ["selection_score_component", "abs_delta"],
                ascending=[False, False],
            ).iloc[0]
            family_rows.append(
                {
                    "dataset": dataset,
                    "metric_family": family,
                    "metrics_in_family": ",".join(sorted(finite["metric_name"].unique())),
                    "primary_metric": str(primary["metric_name"]),
                    "abs_delta": float(primary["abs_delta"]),
                    "relative_delta": float(primary["relative_delta"]),
                    "max_abs_delta": float(finite["abs_delta"].max()),
                    "mean_abs_delta": float(finite["abs_delta"].mean()),
                    "max_relative_delta": float(finite["relative_delta"].max()),
                    "mean_relative_delta": float(finite["relative_delta"].mean()),
                    "selection_score": family_score,
                    "affected_variants": affected_variants,
                    "primary_variant": str(primary["ufce_variant"]),
                    "diagnostic_hypothesis": _diagnostic_hypothesis(str(family)),
                    "required_trace_types": TRACE_TYPES_BY_FAMILY.get(str(family), ""),
                }
            )
        family_rows.sort(key=lambda row: float(row["selection_score"]), reverse=True)
        for rank, row in enumerate(family_rows[: int(top_n)], start=1):
            row = dict(row)
            row["rank"] = int(rank)
            out_rows.append(row)
    return out_rows


def _diagnostic_hypothesis(metric_family: str) -> str:
    if metric_family == "Distance contract":
        return "Observed contributors likely involve distance-space/scaler contract or selected candidate geometry."
    if metric_family == "Changed-feature-set":
        return "Observed contributors likely involve changed-feature detection, tolerance, or representative-candidate selection."
    if metric_family == "APF contract":
        return "Observed contributors likely involve APF numerator/denominator semantics or pass/fail component contract."
    return "Observed contributors require trace inspection."


def _safe_implausibility_count(
    active_ufc: UFCE,
    method_name: str,
    cfdf: pd.DataFrame,
    testdf: pd.DataFrame,
    xtrain: pd.DataFrame,
) -> float:
    try:
        val = active_ufc.implausibility(
            cfdf.copy(),
            testdf.copy(),
            xtrain.copy(),
            len(cfdf),
            0,
            method_name=method_name,
        )
        if isinstance(val, dict):
            return float(int(val.get("count", 0)))
        return float(val)
    except Exception:
        return 0.0


def _safe_feasibility_count(
    active_ufc: UFCE,
    cfdf: pd.DataFrame,
    testdf: pd.DataFrame,
    xtrain: pd.DataFrame,
    features: List[str],
    f2change: List[str],
    bb_model,
    desired_outcome: float,
    uf: Dict[str, float],
) -> float:
    try:
        out = active_ufc.feasibility(
            testdf.copy(),
            cfdf.copy(),
            xtrain.copy(),
            features,
            f2change,
            bb_model,
            desired_outcome,
            uf,
            0,
            method="other",
        )
        if out is None:
            return 0.0
        if isinstance(out, (int, float)):
            return float(out)
        if isinstance(out, tuple) and len(out) >= 1:
            return float(out[0])
        return 0.0
    except Exception:
        return 0.0


def evaluate_ufce_only_metrics(
    *,
    onecfs: pd.DataFrame,
    onetest: pd.DataFrame,
    twocfs: pd.DataFrame,
    twotest: pd.DataFrame,
    threecfs: pd.DataFrame,
    threetest: pd.DataFrame,
    catf: List[str],
    numf: List[str],
    features: List[str],
    f2change: List[str],
    uf: Dict[str, float],
    xtrain: pd.DataFrame,
    bb_model,
    desired_outcome: float,
    contprox_distance_scaler: Optional[Dict[str, object]] = None,
) -> Dict[str, Dict[str, float]]:
    active_ufc = _active_ufce_instance()
    method_triplets = [
        ("UFCE1", onecfs, onetest),
        ("UFCE2", twocfs, twotest),
        ("UFCE3", threecfs, threetest),
    ]
    means: Dict[str, Dict[str, float]] = {m: {} for m in METHODS}

    for method, cfdf, testdf in method_triplets:
        # Prox-Jac
        if len(catf) == 0:
            means[method]["Prox-Jac"] = float("nan")
        else:
            vals: List[float] = []
            n = min(len(cfdf), len(testdf))
            for i in range(n):
                dist = active_ufc.categorical_distance(testdf[i : i + 1], cfdf[i : i + 1], catf, metric="jaccard", agg=None)
                vals.append(_scalar(dist))
            means[method]["Prox-Jac"] = _mean_or_nan(vals)

        # Prox-Euc
        vals = []
        n = min(len(cfdf), len(testdf))
        cf_dist_df = apply_distance_scaler(cfdf, contprox_distance_scaler) if contprox_distance_scaler is not None else cfdf
        test_dist_df = apply_distance_scaler(testdf, contprox_distance_scaler) if contprox_distance_scaler is not None else testdf
        for i in range(n):
            dist = active_ufc.continuous_distance(
                test_dist_df[i : i + 1],
                cf_dist_df[i : i + 1],
                numf,
                metric="euclidean",
                agg=None,
            )
            vals.append(_scalar(dist))
        means[method]["Prox-Euc"] = _mean_or_nan(vals)

        # Sparsity
        sparsity_d, _ = active_ufc.sparsity_count(cfdf.copy(), testdf.copy(), numf, numf)
        sv = np.asarray(list(sparsity_d.values()), dtype=float) if len(sparsity_d) > 0 else np.asarray([])
        means[method]["Sparsity"] = float(np.mean(sv)) if sv.size > 0 else float("nan")

        # Actionability
        try:
            act_cfs, _flag, _ids, _temp = active_ufc.actionability(
                cfdf.copy(),
                testdf.copy(),
                features,
                f2change,
                0,
                uf,
                method="other",
            )
            means[method]["Actionability"] = float(len(act_cfs))
        except Exception:
            means[method]["Actionability"] = 0.0

        means[method]["Plausibility"] = _safe_implausibility_count(active_ufc, method, cfdf, testdf, xtrain)
        means[method]["Feasibility"] = _safe_feasibility_count(
            active_ufc,
            cfdf,
            testdf,
            xtrain,
            features,
            f2change,
            bb_model,
            desired_outcome,
            uf,
        )

    return means


def build_fold_diagnostics(
    *,
    dataset: str,
    fold_name: str,
    fold_index: int,
    fold_df: pd.DataFrame,
    fold_df_ufce: pd.DataFrame,
    xtrain: pd.DataFrame,
    features: List[str],
    catf: List[str],
    numf: List[str],
    uf: Dict[str, float],
    f2change: List[str],
    protectf: List[str],
    bb_model: Any,
    desired_outcome: float,
    cfg: Dict[str, int],
    bundle_cfg: Dict[str, str],
    bundle_meta: Dict[str, object],
    step: Dict[str, float],
    no_cf: int,
    movie_distance_scaler: Optional[Dict[str, object]],
    method_payloads: Dict[str, Dict[str, Any]],
    top_k: int,
) -> Dict[str, List[Dict[str, object]]]:
    active_ufc = _active_ufce_instance()
    records: Dict[str, List[Dict[str, object]]] = {
        "input_config": [],
        "candidate_generation": [],
        "candidate_validation": [],
        "candidate_selection": [],
        "metric_trace": [],
        "feature_contribution": [],
        "apf_component": [],
    }

    feature_mapping = {
        "feature_order": list(features),
        "continuous_features": list(numf),
        "categorical_features": list(catf),
        "binary_features": list(catf),
        "actionable_features": list(f2change),
        "immutable_features": [f for f in features if f not in f2change],
    }

    for method, payload in method_payloads.items():
        cfdf = payload["cfdf"].reset_index(drop=True)
        testdf = payload["testdf"].reset_index(drop=True)
        found_idx = [int(v) for v in payload["found_idx"]]
        trace_rows = list(payload.get("trace_rows", []))
        pair_pos_by_query = {int(src_pos): int(pair_pos) for pair_pos, src_pos in enumerate(found_idx)}
        apf_rows_this_method: List[Dict[str, object]] = []

        for trace_row in trace_rows:
            query_pos = int(trace_row.get("instance_pos", 0))
            query_id = f"{fold_name}#q{query_pos}"
            factual_raw = fold_df.iloc[[query_pos]][features].reset_index(drop=True)
            factual_model_input = fold_df_ufce.iloc[[query_pos]][features].reset_index(drop=True)
            original_prediction, original_probability = _predict_label_and_probability(bb_model, factual_model_input)

            pair_pos = pair_pos_by_query.get(query_pos)
            has_selected = pair_pos is not None and pair_pos < len(cfdf)
            metric_candidate = cfdf.iloc[[pair_pos]].reset_index(drop=True) if has_selected else pd.DataFrame(columns=features)
            metric_factual = testdf.iloc[[pair_pos]].reset_index(drop=True) if has_selected and pair_pos < len(testdf) else factual_raw

            generated_df = trace_row.get("generated_candidates_df")
            flip_df = trace_row.get("label_flip_candidates_df")
            ufce_selected_df = trace_row.get("selected_candidates_df")
            if not isinstance(generated_df, pd.DataFrame):
                generated_df = pd.DataFrame(columns=features)
            if not isinstance(flip_df, pd.DataFrame):
                flip_df = pd.DataFrame(columns=features)
            if not isinstance(ufce_selected_df, pd.DataFrame):
                ufce_selected_df = pd.DataFrame(columns=features)

            selected_by_ufce_id = (
                _candidate_id(dataset, fold_name, query_pos, method, "ufce_selected", 0)
                if not ufce_selected_df.empty
                else None
            )
            metric_candidate_id = (
                _candidate_id(dataset, fold_name, query_pos, method, "metric_candidate", 0)
                if has_selected
                else None
            )
            force_flip_candidate_id = (
                _candidate_id(dataset, fold_name, query_pos, method, "force_flip_candidate", 0)
                if has_selected
                else None
            )

            records["input_config"].append(
                {
                    "dataset": dataset,
                    "fold_id": fold_name,
                    "fold_index": int(fold_index),
                    "query_id": query_id,
                    "query_pos": int(query_pos),
                    "ufce_variant": method,
                    "locked_config_id": "final_freeze_raw",
                    "model_name": type(bb_model).__name__,
                    "seed": 42,
                    "split_id": "classify_dataset_getModel_author_pipeline",
                    "radius": int(cfg["radius"]),
                    "n_neighbors": int(cfg["n_neighbors"]),
                    "step": step,
                    "uf": uf,
                    "f2change": list(f2change),
                    "min_act": int(cfg["min_act"]),
                    "min_feas": int(cfg["min_feas"]),
                    "contprox_metric": "euclidean",
                    "atol": float(getattr(active_ufc, "atol", 1e-5)),
                    "bundle": dict(bundle_cfg),
                    "effective_radius": int(cfg["radius"]),
                    "effective_n_neighbors": int(cfg["n_neighbors"]),
                    "effective_min_act": int(cfg["min_act"]),
                    "effective_min_feas": int(cfg["min_feas"]),
                    "effective_uf_source": bundle_meta.get("effective_uf_source"),
                    "effective_f2change_source": bundle_meta.get("effective_f2change_source"),
                    "effective_step_source": bundle_meta.get("effective_step_source"),
                    "effective_bundle_mode": bundle_meta.get("effective_bundle_mode"),
                    "fallback_used": bool(bundle_meta.get("fallback_used", False)),
                    "fallback_reason": str(bundle_meta.get("fallback_reason", "")),
                    "not_main_table7": bool(bundle_meta.get("not_main_table7", False)),
                    **feature_mapping,
                    "original_profile_raw_json": factual_raw.iloc[0].to_dict(),
                    "original_model_input_json": factual_model_input.iloc[0].to_dict(),
                    "original_prediction": original_prediction,
                    "original_probability": original_probability,
                    "desired_label": int(desired_outcome),
                }
            )

            raw_count = int(len(generated_df))
            flip_count = int(len(flip_df))
            selected_count = int(len(ufce_selected_df))
            empty_reason = ""
            if raw_count == 0:
                empty_reason = "no_raw_candidate_generated"
            elif selected_count == 0:
                empty_reason = "no_selected_candidate_after_public_contract"
            records["candidate_generation"].append(
                {
                    "dataset": dataset,
                    "ufce_variant": method,
                    "fold_id": fold_name,
                    "query_id": query_id,
                    "query_pos": int(query_pos),
                    "candidate_generation_status": "ok" if raw_count > 0 else "empty",
                    "raw_candidate_count": raw_count,
                    "candidate_count_before_filter": raw_count,
                    "candidate_count_after_basic_filter": flip_count,
                    "selected_candidate_by_ufce_id": selected_by_ufce_id,
                    "candidate_empty_reason": empty_reason,
                    "top_k_raw_candidates_json": _candidate_records(
                        generated_df,
                        dataset=dataset,
                        fold_name=fold_name,
                        query_pos=query_pos,
                        method=method,
                        role="raw_candidate",
                        top_k=top_k,
                    ),
                    "top_k_flip_candidates_json": _candidate_records(
                        flip_df,
                        dataset=dataset,
                        fold_name=fold_name,
                        query_pos=query_pos,
                        method=method,
                        role="flip_valid_candidate",
                        top_k=top_k,
                    ),
                    "candidate_pool_summary_json": {
                        "raw_candidate_count": raw_count,
                        "flip_candidate_count": flip_count,
                        "selected_candidate_count": selected_count,
                        "search_meta": _json_safe(trace_row.get("search_meta", {})),
                        "search_parameters": _json_safe(trace_row.get("search_parameters", {})),
                        "source_path": str(trace_row.get("source_path", "")),
                    },
                }
            )

            validation_candidates: List[Tuple[str, str, pd.DataFrame]] = []
            if has_selected:
                validation_candidates.append(("metric_candidate", metric_candidate_id or "", metric_candidate))
                validation_candidates.append(("force_flip_candidate", force_flip_candidate_id or "", metric_candidate))
            for rec in _candidate_records(
                generated_df,
                dataset=dataset,
                fold_name=fold_name,
                query_pos=query_pos,
                method=method,
                role="raw_candidate",
                top_k=top_k,
            ):
                rank = int(rec["candidate_rank"])
                validation_candidates.append(
                    ("raw_candidate", str(rec["candidate_id"]), generated_df.iloc[[rank]].reset_index(drop=True))
                )
            seen_validation_ids = set()
            for candidate_type, candidate_id, candidate_df in validation_candidates:
                if candidate_id in seen_validation_ids or not candidate_id:
                    continue
                seen_validation_ids.add(candidate_id)
                pred, prob = _predict_label_and_probability(bb_model, candidate_df)
                records["candidate_validation"].append(
                    {
                        "dataset": dataset,
                        "ufce_variant": method,
                        "fold_id": fold_name,
                        "query_id": query_id,
                        "query_pos": int(query_pos),
                        "candidate_id": candidate_id,
                        "candidate_type": candidate_type,
                        "candidate_profile_raw_json": candidate_df.iloc[0].to_dict() if not candidate_df.empty else {},
                        "candidate_model_input_json": candidate_df.iloc[0].to_dict() if not candidate_df.empty else {},
                        "candidate_prediction": pred,
                        "candidate_probability": prob,
                        "flip_valid": bool(pred == int(desired_outcome)) if pred is not None else False,
                        "desired_label_satisfied": bool(pred == int(desired_outcome)) if pred is not None else False,
                        "metric_candidate_type": "raw_candidate",
                        "metric_candidate_id": metric_candidate_id,
                        "metric_candidate_selection_stage": "ufce_returned_output",
                    }
                )

            changed_features, changed_details = _changed_feature_details(metric_factual, metric_candidate, features)
            prox_euc, contribution_json, dist_cols, normalizer_name = _feature_contributions(
                metric_factual,
                metric_candidate,
                numf,
                movie_distance_scaler if dataset == "movie" else None,
            )
            prox_jac = None
            if has_selected and len(catf) > 0:
                try:
                    prox_jac = _scalar(active_ufc.categorical_distance(metric_factual, metric_candidate, catf, metric="jaccard", agg=None))
                except Exception:
                    prox_jac = None
            sparsity_value = None
            if has_selected:
                try:
                    sparsity_d, _ = active_ufc.sparsity_count(metric_candidate.copy(), metric_factual.copy(), numf, numf)
                    vals = list(sparsity_d.values())
                    sparsity_value = float(vals[0]) if vals else None
                except Exception:
                    sparsity_value = None

            action_pass, action_reason, action_num, action_den, actionable_changed, non_actionable_changed = _safe_actionability_pair(
                active_ufc,
                metric_factual,
                metric_candidate,
                features,
                f2change,
                uf,
            )
            plaus_pass, plaus_reason, plaus_details = _safe_plausibility_pair(
                active_ufc,
                method,
                metric_factual,
                metric_candidate,
                xtrain,
            )
            feas_pass, feas_reason, feas_details = _safe_feasibility_pair(
                active_ufc,
                metric_factual,
                metric_candidate,
                xtrain,
                features,
                f2change,
                bb_model,
                desired_outcome,
                uf,
            )

            records["candidate_selection"].append(
                {
                    "dataset": dataset,
                    "ufce_variant": method,
                    "fold_id": fold_name,
                    "query_id": query_id,
                    "query_pos": int(query_pos),
                    "selection_rule": "public_ufce_returned_output_then_nearest_if_multiple",
                    "selected_candidate_by_ufce": selected_by_ufce_id,
                    "selected_candidate_by_ufce_id": selected_by_ufce_id,
                    "selected_candidate_used_for_metric": metric_candidate_id,
                    "selected_candidate_used_for_force_flip": force_flip_candidate_id,
                    "selected_candidate_id": selected_by_ufce_id,
                    "selected_candidate_index": 0 if has_selected else None,
                    "selected_candidate_rank_by_prox_euc": 1 if has_selected else None,
                    "selected_candidate_rank_by_sparsity": None,
                    "selected_candidate_rank_by_actionability": None,
                    "selected_candidate_rank_by_feasibility": None,
                    "best_by_prox_euc_candidate_id": metric_candidate_id,
                    "best_by_sparsity_candidate_id": None,
                    "best_by_actionability_candidate_id": None,
                    "best_by_feasibility_candidate_id": None,
                    "metric_candidate_type": "raw_candidate",
                    "metric_candidate_id": metric_candidate_id,
                    "metric_candidate_selection_stage": "ufce_returned_output",
                    "metric_candidate_differs_from_ufce_selected": False,
                    "metric_candidate_explanation": "Table 7 metrics use the public UFCE returned output; force-flip validation is logged separately.",
                    "selected_candidate_metrics_json": {
                        "Prox-Jac": prox_jac,
                        "Prox-Euc": prox_euc,
                        "Sparsity": sparsity_value,
                        "Actionability_pass": action_pass,
                        "Plausibility_pass": plaus_pass,
                        "Feasibility_pass": feas_pass,
                    },
                    "best_candidate_metrics_json": {},
                }
            )

            records["metric_trace"].append(
                {
                    "dataset": dataset,
                    "ufce_variant": method,
                    "fold_id": fold_name,
                    "fold_index": int(fold_index),
                    "query_id": query_id,
                    "query_pos": int(query_pos),
                    "metric_candidate_type": "raw_candidate",
                    "metric_candidate_id": metric_candidate_id,
                    "metric_candidate_selection_stage": "ufce_returned_output",
                    "has_selected_candidate": bool(has_selected),
                    "prox_euc_final_value": prox_euc,
                    "prox_euc_contract": "euclidean over numf; movie uses movie_minmax_0_100 distance space",
                    "prox_jac_value": prox_jac,
                    "sparsity_value": sparsity_value,
                    "changed_features_json": changed_features,
                    "changed_feature_count": int(len(changed_features)),
                    "change_detection_atol": float(getattr(active_ufc, "atol", 1e-5)),
                    "actionability_pass": bool(action_pass),
                    "plausibility_pass": bool(plaus_pass),
                    "feasibility_pass": bool(feas_pass),
                    "actionability_fail_reason": action_reason,
                    "plausibility_fail_reason": plaus_reason,
                    "feasibility_fail_reason": feas_reason,
                    "metric_source_contract": "public_table7_reproduction",
                }
            )

            if has_selected:
                records["feature_contribution"].append(
                    {
                        "dataset": dataset,
                        "ufce_variant": method,
                        "fold_id": fold_name,
                        "query_id": query_id,
                        "query_pos": int(query_pos),
                        "metric_candidate_id": metric_candidate_id,
                        "prox_euc_final_value": prox_euc,
                        "prox_euc_contract": "euclidean",
                        "distance_feature_set": list(dist_cols),
                        "continuous_feature_set_used": list(dist_cols),
                        "raw_delta_by_feature_json": changed_details,
                        "normalized_delta_by_feature_json": {
                            f: details.get("normalized_delta") for f, details in contribution_json.items()
                        },
                        "per_feature_squared_contribution_json": {
                            f: details.get("squared_contribution") for f, details in contribution_json.items()
                        },
                        "per_feature_euc_contribution_json": {
                            f: details.get("euc_contribution") for f, details in contribution_json.items()
                        },
                        "feature_contribution_json": contribution_json,
                        "scaler_or_normalizer_used": normalizer_name,
                        "inverse_transform_applied": False,
                    }
                )

            apf_specs = [
                ("Actionability", action_pass, action_reason, {"actionability_numerator": action_num, "actionability_denominator": action_den}),
                ("Plausibility", plaus_pass, plaus_reason, plaus_details),
                ("Feasibility", feas_pass, feas_reason, feas_details),
            ]
            for metric_name, passed, fail_reason, details in apf_specs:
                apf_rows_this_method.append(
                    {
                        "dataset": dataset,
                        "ufce_variant": method,
                        "fold_id": fold_name,
                        "fold_index": int(fold_index),
                        "query_id": query_id,
                        "query_pos": int(query_pos),
                        "metric_candidate_id": metric_candidate_id,
                        "apf_metric_name": metric_name,
                        "apf_pass": bool(passed),
                        "apf_denominator_type": "queries_with_selected_candidate",
                        "apf_denominator_contract": "current Table 7 public-code contract counts selected UFCE outputs, not all original queries",
                        "apf_eligible_query": bool(has_selected),
                        "apf_has_candidate": bool(raw_count > 0),
                        "apf_has_selected_candidate": bool(has_selected),
                        "apf_counted_in_metric": bool(has_selected),
                        "apf_fold_numerator": None,
                        "apf_fold_denominator": None,
                        "apf_fail_reason": "" if passed else fail_reason,
                        "actionable_features_json": list(f2change),
                        "changed_features_json": changed_features,
                        "non_actionable_changed_features_json": non_actionable_changed,
                        "actionability_numerator": action_num,
                        "actionability_denominator": action_den,
                        "actionability_value": bool(action_pass),
                        "plausibility_checker_name": "LocalOutlierFactor",
                        "n_neighbors": int(cfg["n_neighbors"]),
                        "plausibility_score": _json_safe(details),
                        "plausibility_pass": bool(plaus_pass),
                        "feasibility_checker_components_json": {
                            "actionability_pass": bool(action_pass),
                            "plausibility_pass": bool(plaus_pass),
                            "constraint_pass": bool(feas_pass),
                        },
                        "min_act": int(cfg["min_act"]),
                        "min_feas": int(cfg["min_feas"]),
                        "constraint_pass": bool(feas_pass),
                        "feasibility_pass": bool(feas_pass),
                        "feasibility_value": bool(feas_pass),
                        "feasibility_fail_reason": feas_reason,
                    }
                )

        for metric_name in APF_METRICS:
            denom = int(sum(1 for row in apf_rows_this_method if row["apf_metric_name"] == metric_name and row["apf_counted_in_metric"]))
            numerator = int(sum(1 for row in apf_rows_this_method if row["apf_metric_name"] == metric_name and row["apf_counted_in_metric"] and row["apf_pass"]))
            for row in apf_rows_this_method:
                if row["apf_metric_name"] == metric_name:
                    row["apf_fold_numerator"] = numerator
                    row["apf_fold_denominator"] = denom
        records["apf_component"].extend(apf_rows_this_method)

    return records


def run_one_fold(
    *,
    dataset: str,
    fold_df: pd.DataFrame,
    x_all: pd.DataFrame,
    xtest: pd.DataFrame,
    xtrain: pd.DataFrame,
    data_lab1: pd.DataFrame,
    features: List[str],
    catf: List[str],
    numf: List[str],
    uf: Dict[str, float],
    f2change: List[str],
    protectf: List[str],
    bb_model,
    desired_outcome: float,
    mi_fp: List,
    no_cf: int,
    step: Dict[str, float],
    fold_name: str,
    scaler: Optional[Dict],
    flip_filter_enabled: bool,
    movie_distance_scaler: Optional[Dict[str, object]],
    fold_index: int,
    debug: int,
    prox_euc_contract_debug: int,
    contract_debug_fold: int,
    contract_debug_pos: int,
    contract_debug_method: str,
    diagnostics_enabled: bool = False,
    diagnostics_top_k: int = 5,
    cfg: Optional[Dict[str, int]] = None,
    bundle_cfg: Optional[Dict[str, str]] = None,
    bundle_meta: Optional[Dict[str, object]] = None,
) -> FoldResult:
    deterministic_seed = int(hashlib.sha256(f"{dataset}:{fold_name}:42".encode("utf-8")).hexdigest()[:8], 16)
    random.seed(deterministic_seed)
    np.random.seed(deterministic_seed % (2**32 - 1))
    use_mad = _has_mad_scaler(scaler)

    # Movie: keep UFCE logic in raw space, but use scaled distance space for neighbor search.
    if dataset == "movie" and movie_distance_scaler is not None:
        fold_df_ufce = fold_df[features].copy()
        data_lab1_ufce = data_lab1[features].copy()
        fold_df_dist = apply_distance_scaler(fold_df_ufce, movie_distance_scaler)
        data_lab1_dist = apply_distance_scaler(data_lab1_ufce, movie_distance_scaler)
        distance_scaler = movie_distance_scaler
    elif use_mad:
        # Non-movie MAD path when scaler is provided by pipeline.
        fold_df_ufce = _mad_transform(fold_df[features].copy(), scaler, numf)
        data_lab1_ufce = _mad_transform(data_lab1[features].copy(), scaler, numf)
        fold_df_dist = None
        data_lab1_dist = None
        distance_scaler = None
    else:
        fold_df_ufce = fold_df[features].copy()
        data_lab1_ufce = data_lab1[features].copy()
        fold_df_dist = None
        data_lab1_dist = None
        distance_scaler = None

    dbg_fold = fold_df_dist if fold_df_dist is not None else fold_df_ufce
    dbg_lab1 = data_lab1_dist if data_lab1_dist is not None else data_lab1_ufce
    if not dbg_fold.empty and not dbg_lab1.empty:
        x0 = dbg_fold[features].iloc[0:1].values
        lab1 = dbg_lab1[features].values
        dists = np.linalg.norm(lab1 - x0, axis=1)
        print(f"[DBG] Scaled Distances: min={dists.min():.2f}, mean={dists.mean():.2f}, max={dists.max():.2f}")

    if diagnostics_enabled:
        onecfs, t1, idx1, trace1 = cfmethods.sfexp(
            x_all,
            data_lab1_ufce,
            fold_df_ufce[:],
            uf,
            step,
            f2change,
            numf,
            catf,
            bb_model,
            desired_outcome,
            no_cf,
            features,
            flip_filter_enabled=bool(flip_filter_enabled),
            distance_data_lab1=data_lab1_dist,
            distance_X_test=fold_df_dist,
            distance_scaler=distance_scaler,
            return_trace=True,
        )
        twocfs, t2, idx2, trace2 = cfmethods.dfexp(
            x_all,
            data_lab1_ufce,
            fold_df_ufce[:],
            uf,
            mi_fp[:5],
            numf,
            catf,
            f2change,
            protectf,
            bb_model,
            desired_outcome,
            no_cf,
            features,
            flip_filter_enabled=bool(flip_filter_enabled),
            distance_data_lab1=data_lab1_dist,
            distance_X_test=fold_df_dist,
            distance_scaler=distance_scaler,
            return_trace=True,
        )
        threecfs, t3, idx3, trace3 = cfmethods.tfexp(
            x_all,
            data_lab1_ufce,
            fold_df_ufce[:],
            uf,
            mi_fp[:5],
            numf,
            catf,
            f2change,
            protectf,
            bb_model,
            desired_outcome,
            no_cf,
            features,
            flip_filter_enabled=bool(flip_filter_enabled),
            distance_data_lab1=data_lab1_dist,
            distance_X_test=fold_df_dist,
            distance_scaler=distance_scaler,
            return_trace=True,
        )
    else:
        trace1, trace2, trace3 = [], [], []
        onecfs, t1, idx1 = cfmethods.sfexp(
            x_all,
            data_lab1_ufce,
            fold_df_ufce[:],
            uf,
            step,
            f2change,
            numf,
            catf,
            bb_model,
            desired_outcome,
            no_cf,
            features,
            flip_filter_enabled=bool(flip_filter_enabled),
            distance_data_lab1=data_lab1_dist,
            distance_X_test=fold_df_dist,
            distance_scaler=distance_scaler,
        )
        twocfs, t2, idx2 = cfmethods.dfexp(
            x_all,
            data_lab1_ufce,
            fold_df_ufce[:],
            uf,
            mi_fp[:5],
            numf,
            catf,
            f2change,
            protectf,
            bb_model,
            desired_outcome,
            no_cf,
            features,
            flip_filter_enabled=bool(flip_filter_enabled),
            distance_data_lab1=data_lab1_dist,
            distance_X_test=fold_df_dist,
            distance_scaler=distance_scaler,
        )
        threecfs, t3, idx3 = cfmethods.tfexp(
            x_all,
            data_lab1_ufce,
            fold_df_ufce[:],
            uf,
            mi_fp[:5],
            numf,
            catf,
            f2change,
            protectf,
            bb_model,
            desired_outcome,
            no_cf,
            features,
            flip_filter_enabled=bool(flip_filter_enabled),
            distance_data_lab1=data_lab1_dist,
            distance_X_test=fold_df_dist,
            distance_scaler=distance_scaler,
        )

    if use_mad and dataset != "movie":
        for cf_df in [onecfs, twocfs, threecfs]:
            if cf_df is not None and not cf_df.empty:
                _mad_inverse_inplace(cf_df, scaler, numf)

    if int(debug) == 1:
        n_test = int(len(fold_df))
        one_total, two_total, three_total = int(len(onecfs)), int(len(twocfs)), int(len(threecfs))
        one_solved, two_solved, three_solved = int(len(idx1)), int(len(idx2)), int(len(idx3))
        one_mean = float(one_total / max(one_solved, 1))
        two_mean = float(two_total / max(two_solved, 1))
        three_mean = float(three_total / max(three_solved, 1))
        print(
            "[DBG][COVERAGE] "
            f"fold={fold_index} n_test={n_test} "
            f"UFCE1 solved={one_solved} mean_cf={one_mean:.3f} total_cf_rows={one_total} | "
            f"UFCE2 solved={two_solved} mean_cf={two_mean:.3f} total_cf_rows={two_total} | "
            f"UFCE3 solved={three_solved} mean_cf={three_mean:.3f} total_cf_rows={three_total}"
        )

    if int(prox_euc_contract_debug) == 1 and int(fold_index) == int(contract_debug_fold):
        method_to_payload = {
            "UFCE1": (onecfs.reset_index(drop=True), [int(v) for v in idx1]),
            "UFCE2": (twocfs.reset_index(drop=True), [int(v) for v in idx2]),
            "UFCE3": (threecfs.reset_index(drop=True), [int(v) for v in idx3]),
        }
        method = str(contract_debug_method)
        debug_pos = int(contract_debug_pos)
        cfdf_method, idx_method = method_to_payload.get(method, (pd.DataFrame(), []))

        status = "ok"
        raw_l2 = float("nan")
        z_l2 = float("nan")
        mm_l2 = float("nan")

        if debug_pos < 0 or debug_pos >= len(fold_df):
            status = "no_cf_for_instance"
        else:
            factual = fold_df.iloc[[debug_pos]][features].reset_index(drop=True)
            pair_pos = next((i for i, src_pos in enumerate(idx_method) if int(src_pos) == debug_pos), None)
            if pair_pos is None or pair_pos >= len(cfdf_method):
                status = "no_cf_for_instance"
            else:
                cf_row = cfdf_method.iloc[[pair_pos]].reset_index(drop=True)
                dist_cols = _shared_numeric_features(numf, xtrain, factual, cf_row)
                raw_l2 = _row_l2_distance(factual, cf_row, dist_cols)
                if len(dist_cols) > 0:
                    try:
                        z_scaler = StandardScaler().fit(xtrain.loc[:, dist_cols])
                        factual_z = pd.DataFrame(z_scaler.transform(factual.loc[:, dist_cols]), columns=dist_cols)
                        cf_z = pd.DataFrame(z_scaler.transform(cf_row.loc[:, dist_cols]), columns=dist_cols)
                        z_l2 = _row_l2_distance(factual_z, cf_z, dist_cols)
                    except Exception:
                        z_l2 = float("nan")
                    try:
                        mm_scaler = MinMaxScaler().fit(xtrain.loc[:, dist_cols])
                        factual_mm = pd.DataFrame(mm_scaler.transform(factual.loc[:, dist_cols]), columns=dist_cols)
                        cf_mm = pd.DataFrame(mm_scaler.transform(cf_row.loc[:, dist_cols]), columns=dist_cols)
                        mm_l2 = _row_l2_distance(factual_mm, cf_mm, dist_cols)
                    except Exception:
                        mm_l2 = float("nan")

        print(
            "[DBG][PROX-EUC-CONTRACT] "
            f"dataset={dataset} fold={fold_index} pos={debug_pos} "
            f"method={method} fit_scope=train status={status}"
        )
        print(
            f"raw_l2={_fmt_dbg_float(raw_l2)} "
            f"zscore_l2={_fmt_dbg_float(z_l2)} "
            f"minmax01_l2={_fmt_dbg_float(mm_l2)}"
        )

    onetest = fold_df.iloc[idx1][features].reset_index(drop=True)
    twotest = fold_df.iloc[idx2][features].reset_index(drop=True)
    threetest = fold_df.iloc[idx3][features].reset_index(drop=True)

    means = evaluate_ufce_only_metrics(
        onecfs=onecfs.reset_index(drop=True),
        onetest=onetest,
        twocfs=twocfs.reset_index(drop=True),
        twotest=twotest,
        threecfs=threecfs.reset_index(drop=True),
        threetest=threetest,
        catf=catf,
        numf=numf,
        features=features,
        f2change=f2change,
        uf=uf,
        xtrain=xtrain,
        bb_model=bb_model,
        desired_outcome=desired_outcome,
        contprox_distance_scaler=movie_distance_scaler if dataset == "movie" else None,
    )
    times = {"UFCE1": float(t1), "UFCE2": float(t2), "UFCE3": float(t3)}
    diagnostics = None
    if diagnostics_enabled:
        method_payloads = {
            "UFCE1": {
                "cfdf": onecfs.reset_index(drop=True),
                "testdf": onetest,
                "found_idx": [int(v) for v in idx1],
                "trace_rows": trace1,
            },
            "UFCE2": {
                "cfdf": twocfs.reset_index(drop=True),
                "testdf": twotest,
                "found_idx": [int(v) for v in idx2],
                "trace_rows": trace2,
            },
            "UFCE3": {
                "cfdf": threecfs.reset_index(drop=True),
                "testdf": threetest,
                "found_idx": [int(v) for v in idx3],
                "trace_rows": trace3,
            },
        }
        diagnostics = build_fold_diagnostics(
            dataset=dataset,
            fold_name=fold_name,
            fold_index=fold_index,
            fold_df=fold_df,
            fold_df_ufce=fold_df_ufce,
            xtrain=xtrain,
            features=features,
            catf=catf,
            numf=numf,
            uf=uf,
            f2change=f2change,
            protectf=protectf,
            bb_model=bb_model,
            desired_outcome=desired_outcome,
            cfg=cfg or {},
            bundle_cfg=bundle_cfg or {},
            bundle_meta=bundle_meta or {},
            step=step,
            no_cf=no_cf,
            movie_distance_scaler=movie_distance_scaler,
            method_payloads=method_payloads,
            top_k=diagnostics_top_k,
        )
    return FoldResult(fold_name=fold_name, means=means, times=times, diagnostics=diagnostics)


def aggregate_results(folds: List[FoldResult]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    rows = []
    time_rows = []
    for fr in folds:
        for method in METHODS:
            row = {"fold": fr.fold_name, "method": method}
            row.update(fr.means[method])
            rows.append(row)
        time_rows.append(fr.times)
    df = pd.DataFrame(rows)
    time_df = pd.DataFrame(time_rows)
    mean_df = df.groupby("method")[METRICS].mean().T
    std_df = df.groupby("method")[METRICS].std(ddof=1).T
    return mean_df[METHODS], std_df[METHODS], time_df[METHODS].mean()


def print_ours_table(mean_df: pd.DataFrame, std_df: pd.DataFrame, time_mean: pd.Series) -> None:
    print("=== Ours (Fold-mean aggregated) ===")
    print("Mean (across folds):")
    print(mean_df.to_string(float_format=lambda x: f"{x:0.6f}"))
    print("\nStd  (across folds):")
    print(std_df.to_string(float_format=lambda x: f"{x:0.6f}"))
    print("\nMean time (sec) across folds:")
    for m in METHODS:
        print(f"- {m}: {time_mean[m]:0.2f}s")
    print("==================================\n")


def print_author_delta(dataset: str, mean_df: pd.DataFrame) -> None:
    author = AUTHOR_TABLE7.get(dataset, None)
    if author is None:
        print(f"[WARN] No AUTHOR_TABLE7 values registered for dataset='{dataset}'. Skipping delta table.\n")
        return

    print("=== Authors vs Ours (Delta = Ours - Authors) ===")
    out_rows = []
    for method in METHODS:
        a = author.get(method, None)
        if a is None:
            continue
        for metric in METRICS:
            ours = float(mean_df.loc[metric, method])
            auth = float(a[metric])
            out_rows.append(
                {
                    "Method": method,
                    "Metric": metric,
                    "Authors": auth,
                    "Ours": ours,
                    "Delta(Ours-Authors)": ours - auth,
                }
            )
    out = pd.DataFrame(out_rows)
    print(out.to_string(index=False, float_format=lambda x: f"{x:0.6f}"))
    print("==============================================\n")


def plot_2x3_metrics(mean_df: pd.DataFrame, std_df: pd.DataFrame, out_path: str, title: str) -> None:
    if plt is None:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("matplotlib is not available in this environment; plot was skipped.\n")
        return
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    x = np.arange(len(METHODS))
    for ax, metric in zip(axes, METRICS):
        y = mean_df.loc[metric, METHODS].values
        e = std_df.loc[metric, METHODS].fillna(0.0).values
        ax.bar(x, y, yerr=e, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS)
        ax.set_title(metric)
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_for_dataset(dataset: str, args, run_id: str) -> Dict[str, object]:
    t0 = time.time()
    cfg = resolve_effective_cfg(dataset, args)
    print(
        "[CFG] "
        f"dataset={dataset} radius={cfg['radius']} n_neighbors={cfg['n_neighbors']} "
        f"min_act={cfg['min_act']} min_feas={cfg['min_feas']} "
        f"flip={cfg['ufce_flip_filter']} contprox_metric={args.contprox_metric} no_cf={args.no_cf}"
    )
    if int(args.debug) == 1:
        print(
            "[CFG][METRIC-CONTRACT] "
            f"Action/Plaus/Feas are COUNTS over generated CFs; max=no_cf={args.no_cf}"
        )

    data_path = os.path.join(args.data_dir, f"{dataset}.csv")
    datasetdf = pd.read_csv(data_path)
    out = classify_dataset_getModel(datasetdf, data_name=dataset)
    scaler = None
    if len(out) == 8:
        lr, lr_mean, lr_std, xtest, xtrain, x_all, _y, datasetdf = out
    elif len(out) == 9:
        lr, lr_mean, lr_std, xtest, xtrain, x_all, _y, datasetdf, scaler = out
    else:
        raise ValueError(f"Unexpected classify_dataset_getModel return length: {len(out)}")

    (
        features,
        catf,
        numf,
        uf,
        f2change,
        outcome_label,
        desired_outcome,
        _nbr_features,
        protectf,
        _data_lab0,
        data_lab1,
    ) = get_dataset_constraints(dataset, datasetdf)
    
    bundle = resolve_bundle_config(
        dataset=dataset,
        args=args,
        author_uf=uf,
        author_f2change=f2change,
    )
    bundle_cfg = bundle.bundle_cfg
    uf = bundle.uf
    f2change = bundle.f2change
    step = bundle.step
    effective_config = effective_config_record(
        dataset=dataset,
        cfg=cfg,
        bundle=bundle,
        runtime_profile=str(args.runtime_profile),
    )

    print(
        "[BUNDLE] "
        f"dataset={dataset} requested_bundle_mode={bundle.requested_bundle_mode} "
        f"effective_bundle_mode={bundle.effective_bundle_mode} "
        f"uf_mode={bundle_cfg['uf_mode']} "
        f"step_mode={bundle_cfg['step_mode']} "
        f"f2change_mode={bundle_cfg['f2change_mode']} "
        f"not_main_table7={bundle.not_main_table7}"
    )
    mi_fp = normalize_mi_feature_pairs(ufc.get_top_MI_features(x_all, features), features)
    movie_distance_scaler = None
    if dataset == "movie":
        movie_distance_scaler = build_movie_distance_scaler(
            datasetdf=datasetdf,
            features=features,
            numf=numf,
            outcome_label=outcome_label,
        )

    print(f"Dataset: {dataset}")
    print("Black-box: LogisticRegression (author pipeline)")
    print(f"CV accuracy: {lr_mean:.6f} +/- {lr_std:.6f}")
    print_feature_relations(x_all, features, mi_fp)

    cfmethods.initUFCE(
        radius=cfg["radius"],
        n_neighbors=cfg["n_neighbors"],
        contprox_metric=args.contprox_metric,
        min_act=cfg["min_act"],
        min_feas=cfg["min_feas"],
        atol=1e-5,
    )
    eval_module.ufc = cfmethods.ufc

    testfold_path = os.path.join(args.folds_dir, dataset, "totest")
    testfolds = sorted(glob.glob(os.path.join(testfold_path, "*.csv")))
    if not testfolds:
        raise FileNotFoundError(f"No test folds found in {testfold_path}")
    if args.fold_file:
        target_name = os.path.basename(str(args.fold_file))
        selected = [fp for fp in testfolds if os.path.basename(fp) == target_name]
        if not selected:
            raise FileNotFoundError(
                f"Requested fold_file='{target_name}' was not found in {testfold_path}"
            )
        testfolds = selected
    if args.max_folds > 0:
        testfolds = testfolds[: args.max_folds]

    print(f"Found {len(testfolds)} folds. Processing...")
    folds_results: List[FoldResult] = []

    for fold_idx, fp in enumerate(testfolds):
        fold_name = os.path.basename(fp)
        print(f"\n--- Fold {fold_idx}: {fold_name} ---")
        fold_df = pd.read_csv(fp)
        fr = run_one_fold(
            dataset=dataset,
            fold_df=fold_df,
            x_all=x_all,
            xtest=xtest,
            xtrain=xtrain,
            data_lab1=data_lab1,
            features=features,
            catf=catf,
            numf=numf,
            uf=uf,
            f2change=f2change,
            protectf=protectf,
            bb_model=lr,
            desired_outcome=desired_outcome,
            mi_fp=mi_fp,
            no_cf=args.no_cf,
            step=step,
            fold_name=fold_name,
            scaler=scaler,
            flip_filter_enabled=bool(cfg["ufce_flip_filter"]),
            movie_distance_scaler=movie_distance_scaler,
            fold_index=fold_idx,
            debug=args.debug,
            prox_euc_contract_debug=args.prox_euc_contract_debug,
            contract_debug_fold=args.contract_debug_fold,
            contract_debug_pos=args.contract_debug_pos,
            contract_debug_method=args.contract_debug_method,
            diagnostics_enabled=bool(getattr(args, "diagnostics", False)),
            diagnostics_top_k=int(getattr(args, "top_k_candidates", 5)),
            cfg=cfg,
            bundle_cfg=bundle_cfg,
            bundle_meta=effective_config,
        )
        print("Times (sec): " + ", ".join([f"{m}={fr.times[m]:0.2f}" for m in METHODS]))
        folds_results.append(fr)

    mean_df, std_df, time_mean = aggregate_results(folds_results)
    print_ours_table(mean_df, std_df, time_mean)
    print_author_delta(dataset, mean_df)

    os.makedirs(args.out_dir, exist_ok=True)
    summary_csv = os.path.join(args.out_dir, f"summary_{dataset}_{run_id}.csv")
    mean_df.to_csv(summary_csv)

    plot_path = os.path.join(args.out_dir, f"metrics_2x3_{dataset}_{run_id}.png")
    plot_2x3_metrics(
        mean_df=mean_df,
        std_df=std_df,
        out_path=plot_path,
        title=f"UFCE v3 Reproduction — {dataset.capitalize()}",
    )

    runtime_sec = float(time.time() - t0)
    print("\nExecution Complete.")
    print(f"- Summary CSV saved: {summary_csv}")
    print(f"- Metrics Plot saved: {plot_path}")
    print(f"Total runtime: {runtime_sec:0.2f}s")

    delta_rows = build_table7_delta_rows(dataset, mean_df)
    diagnostics_records: Dict[str, List[Dict[str, object]]] = {
        "input_config": [],
        "candidate_generation": [],
        "candidate_validation": [],
        "candidate_selection": [],
        "metric_trace": [],
        "feature_contribution": [],
        "apf_component": [],
    }
    if bool(getattr(args, "diagnostics", False)):
        for fr in folds_results:
            if not fr.diagnostics:
                continue
            for key in diagnostics_records:
                diagnostics_records[key].extend(fr.diagnostics.get(key, []))

    return {
        "dataset": dataset,
        "status": "ok",
        "runtime_sec": runtime_sec,
        "n_folds": int(len(testfolds)),
        "error": "",
        "summary_csv": summary_csv,
        "plot_path": plot_path,
        "delta_rows": delta_rows,
        "mean_records": [
            {
                "dataset": dataset,
                "ufce_variant": method,
                "metric_name": metric,
                "reproduced_value": _finite_float(mean_df.loc[metric, method]),
            }
            for method in METHODS
            for metric in METRICS
        ],
        "diagnostics_records": diagnostics_records,
        "locked_config": {
            "runtime_profile": args.runtime_profile,
            "requested_bundle_mode": bundle.requested_bundle_mode,
            "bundle_mode": bundle.effective_bundle_mode,
            "effective_bundle_mode": bundle.effective_bundle_mode,
            "ufce_flip_filter": int(cfg["ufce_flip_filter"]),
            "values": cfg,
            "bundle": bundle_cfg,
            "effective_config": effective_config,
            "not_main_table7": bool(bundle.not_main_table7),
        },
        "effective_config": effective_config,
        "feature_mapping": {
            "dataset": dataset,
            "feature_order": list(features),
            "continuous_features": list(numf),
            "categorical_features": list(catf),
            "binary_features": list(catf),
            "actionable_features": list(f2change),
            "immutable_features": [f for f in features if f not in f2change],
        },
    }


def write_run_manifest(args, run_id: str, out_dir: str, extra: Optional[Dict[str, object]] = None) -> str:
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/final/part1/01b_reproduce_ufce_only.py",
        "dataset": args.dataset,
        "runtime_profile": args.runtime_profile,
        "requested_bundle_mode": args.bundle_mode,
        "effective_bundle_mode": _canonical_bundle_mode(str(args.bundle_mode)),
        "not_main_table7": bool(_canonical_bundle_mode(str(args.bundle_mode)) == "final_blindspot_best"),
        "bundle_mode": args.bundle_mode,
        "no_cf": args.no_cf,
        "max_folds": args.max_folds,
        "fold_file": args.fold_file,
        "contprox_metric": args.contprox_metric,
        "ufce_flip_filter_override": args.ufce_flip_filter,
        "data_dir": args.data_dir,
        "folds_dir": args.folds_dir,
        "out_dir": out_dir,
        "final_runtime_config": FINAL_RUNTIME_CONFIG,
        "table7_author_public_bundle": TABLE7_AUTHOR_PUBLIC_BUNDLE,
        "final_blindspot_bundle": FINAL_BLINDSPOT_BUNDLE,
        "final_blindspot_bundle_usage": "separate blind-spot diagnostics only; not main Table 7 thesis reproduction",
        "effective_config_by_dataset": build_effective_manifest_config(args),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": safe_version("numpy"),
            "pandas": safe_version("pandas"),
            "scipy": safe_version("scipy"),
            "scikit-learn": safe_version("scikit-learn"),
            "matplotlib": safe_version("matplotlib"),
            "ufce": getattr(ufce, "__version__", "unknown"),
        },
        "extra": extra or {},
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"run_manifest_{run_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    return path


def _diagnostic_output_path(args, run_id: str) -> str:
    return os.path.join(args.diagnostic_out_dir, run_id)


def validate_diagnostic_args(args) -> None:
    if not bool(getattr(args, "diagnostics", False)):
        return
    explicit_runtime = _arg_present("--runtime_profile", "--runtime-profile")
    explicit_flip = _arg_present("--ufce_flip_filter", "--ufce-flip-filter")
    if explicit_runtime and args.runtime_profile != "final_freeze":
        raise ValueError("--diagnostics requires runtime_profile=final_freeze.")
    if explicit_flip and int(args.ufce_flip_filter) != 0:
        raise ValueError("--diagnostics requires ufce_flip_filter=0 for raw Table 7 reproduction.")
    for name in ["radius", "n_neighbors", "min_act", "min_feas"]:
        if getattr(args, name) is not None:
            raise ValueError(f"--diagnostics cannot override locked config field --{name}.")
    args.runtime_profile = "final_freeze"
    args.ufce_flip_filter = 0
    args.out_dir = _diagnostic_output_path(args, args.run_id)


def _collect_diagnostic_records(results: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    records: Dict[str, List[Dict[str, object]]] = {
        "input_config": [],
        "candidate_generation": [],
        "candidate_validation": [],
        "candidate_selection": [],
        "metric_trace": [],
        "feature_contribution": [],
        "apf_component": [],
    }
    for rec in results:
        ds_records = rec.get("diagnostics_records", {})
        if not isinstance(ds_records, dict):
            continue
        for key in records:
            rows = ds_records.get(key, [])
            if isinstance(rows, list):
                records[key].extend(rows)
    return records


def reconstruct_metrics_from_trace(
    *,
    delta_rows: Sequence[Dict[str, object]],
    metric_trace_rows: Sequence[Dict[str, object]],
    apf_trace_rows: Sequence[Dict[str, object]],
) -> pd.DataFrame:
    reported = pd.DataFrame(delta_rows)
    out_rows: List[Dict[str, object]] = []
    metric_df = pd.DataFrame(metric_trace_rows)
    apf_df = pd.DataFrame(apf_trace_rows)
    value_fields = {
        "Prox-Jac": "prox_jac_value",
        "Prox-Euc": "prox_euc_final_value",
        "Sparsity": "sparsity_value",
    }

    recomputed: Dict[Tuple[str, str, str], Optional[float]] = {}
    if not metric_df.empty:
        for metric_name, field_name in value_fields.items():
            if field_name not in metric_df.columns:
                continue
            tmp = metric_df.copy()
            tmp[field_name] = pd.to_numeric(tmp[field_name], errors="coerce")
            fold_means = (
                tmp.dropna(subset=[field_name])
                .groupby(["dataset", "ufce_variant", "fold_id"], dropna=False)[field_name]
                .mean()
                .reset_index()
            )
            if not fold_means.empty:
                means = fold_means.groupby(["dataset", "ufce_variant"], dropna=False)[field_name].mean().reset_index()
                for _, row in means.iterrows():
                    recomputed[(str(row["dataset"]), str(row["ufce_variant"]), metric_name)] = float(row[field_name])

    if not apf_df.empty:
        for metric_name in APF_METRICS:
            tmp = apf_df.loc[apf_df["apf_metric_name"] == metric_name].copy()
            if tmp.empty:
                continue
            tmp["apf_fold_numerator"] = pd.to_numeric(tmp["apf_fold_numerator"], errors="coerce")
            fold_counts = (
                tmp.dropna(subset=["apf_fold_numerator"])
                .groupby(["dataset", "ufce_variant", "fold_id"], dropna=False)["apf_fold_numerator"]
                .first()
                .reset_index()
            )
            if not fold_counts.empty:
                means = fold_counts.groupby(["dataset", "ufce_variant"], dropna=False)["apf_fold_numerator"].mean().reset_index()
                for _, row in means.iterrows():
                    recomputed[(str(row["dataset"]), str(row["ufce_variant"]), metric_name)] = float(row["apf_fold_numerator"])

    for _, row in reported.iterrows():
        dataset = str(row["dataset"])
        method = str(row["ufce_variant"])
        metric = str(row["metric_name"])
        reported_value = _finite_float(row.get("reproduced_value"))
        recomputed_value = recomputed.get((dataset, method, metric))
        if reported_value is None and recomputed_value is None:
            abs_diff = None
            ok = True
        elif reported_value is None or recomputed_value is None:
            abs_diff = None
            ok = False
        else:
            abs_diff = abs(float(recomputed_value) - float(reported_value))
            ok = bool(abs_diff <= RECONSTRUCTION_ATOL)
        out_rows.append(
            {
                "dataset": dataset,
                "ufce_variant": method,
                "metric_name": metric,
                "reported_reproduced_value": reported_value,
                "recomputed_from_trace_value": recomputed_value,
                "abs_diff": abs_diff,
                "validation_ok": ok,
            }
        )
    return pd.DataFrame(out_rows)


def select_representative_diagnostic_cases(
    *,
    targets: Sequence[Dict[str, object]],
    metric_trace_rows: Sequence[Dict[str, object]],
    apf_trace_rows: Sequence[Dict[str, object]],
    candidate_generation_rows: Sequence[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], str]:
    metric_df = pd.DataFrame(metric_trace_rows)
    apf_df = pd.DataFrame(apf_trace_rows)
    gen_df = pd.DataFrame(candidate_generation_rows)
    cases: List[Dict[str, object]] = []
    lines = [
        "# Representative Diagnostic Examples",
        "",
        "These are representative diagnostic examples, not per-query author-vs-reproduction deltas.",
        "",
    ]
    value_field_by_metric = {
        "Prox-Jac": "prox_jac_value",
        "Prox-Euc": "prox_euc_final_value",
        "Sparsity": "sparsity_value",
    }

    for target in targets:
        dataset = str(target["dataset"])
        method = str(target["primary_variant"])
        metric = str(target["primary_metric"])
        target_rank = int(target["rank"])
        selected_rows: List[Dict[str, object]] = []

        if metric in APF_METRICS and not apf_df.empty:
            tmp = apf_df[
                (apf_df["dataset"] == dataset)
                & (apf_df["ufce_variant"] == method)
                & (apf_df["apf_metric_name"] == metric)
            ].copy()
            if not tmp.empty:
                fail_rows = tmp[tmp["apf_pass"] == False]
                selected = fail_rows.head(1) if not fail_rows.empty else tmp.head(1)
                median = tmp.iloc[[len(tmp) // 2]]
                selected_rows.extend(selected.to_dict(orient="records"))
                if not median.empty:
                    selected_rows.extend(median.to_dict(orient="records"))
        elif metric in value_field_by_metric and not metric_df.empty:
            field_name = value_field_by_metric[metric]
            if field_name in metric_df.columns:
                tmp = metric_df[(metric_df["dataset"] == dataset) & (metric_df["ufce_variant"] == method)].copy()
                tmp[field_name] = pd.to_numeric(tmp[field_name], errors="coerce")
                tmp = tmp.dropna(subset=[field_name])
                if not tmp.empty:
                    selected_rows.append(tmp.sort_values(field_name, ascending=False).iloc[0].to_dict())
                    selected_rows.append(tmp.iloc[(tmp[field_name] - tmp[field_name].median()).abs().argsort()].iloc[0].to_dict())

        if not selected_rows and not gen_df.empty:
            fallback = gen_df[(gen_df["dataset"] == dataset) & (gen_df["ufce_variant"] == method)].head(1)
            selected_rows.extend(fallback.to_dict(orient="records"))

        unique_query_ids = set()
        kept = 0
        for row in selected_rows:
            query_id = str(row.get("query_id", ""))
            if query_id in unique_query_ids:
                continue
            unique_query_ids.add(query_id)
            kept += 1
            case_type = "Largest deviation diagnostic example" if kept == 1 else "Typical median diagnostic example"
            case = {
                "dataset": dataset,
                "target_rank": target_rank,
                "metric_family": target["metric_family"],
                "primary_metric": metric,
                "ufce_variant": method,
                "query_id": query_id,
                "case_type": case_type,
                "table_author_value": None,
                "table_reproduced_value": None,
                "table_delta": None,
                "trace_excerpt": _json_safe(row),
            }
            cases.append(case)
            lines.extend(
                [
                    f"## {dataset} / {method} / {metric} / {query_id}",
                    "",
                    f"Case type: {case_type}",
                    "",
                    "Author metric: table-level only",
                    "Reproduced metric: table-level only",
                    "Delta: table-level only",
                    "",
                    "Original profile: see per_query_input_config_trace.jsonl",
                    "Selected candidate: see candidate_selection_trace.jsonl",
                    "Changed features: see per_query_metric_trace.jsonl",
                    "Candidate pool: see per_query_candidate_generation_trace.jsonl",
                    "Selection rule: public UFCE returned output",
                    "Metric contribution: see feature_contribution_trace.jsonl and apf_component_trace.jsonl",
                    "APF components: see apf_component_trace.jsonl",
                    "Interpretation: representative diagnostic example for the table-level target, not a per-query author comparison.",
                    "",
                ]
            )
            if kept >= 2:
                break
    return cases, "\n".join(lines)


def build_deviation_notes(targets: Sequence[Dict[str, object]]) -> str:
    lines = ["# Dataset Metric Deviation Notes", ""]
    if not targets:
        lines.extend(["No ranked diagnostic targets were available.", ""])
        return "\n".join(lines)
    for dataset in sorted({str(row["dataset"]) for row in targets}):
        lines.append(f"## {dataset}")
        lines.append("")
        for target in [row for row in targets if str(row["dataset"]) == dataset]:
            lines.append(f"### Target {target['rank']}: {target['metric_family']}")
            lines.append("")
            lines.append("Observed deviation:")
            lines.append(
                f"- Primary metric `{target['primary_metric']}` on `{target['primary_variant']}`; "
                f"selection_score={float(target['selection_score']):.6g}, "
                f"max_abs_delta={float(target['max_abs_delta']):.6g}, "
                f"max_relative_delta={float(target['max_relative_delta']):.6g}."
            )
            lines.append("Trace evidence:")
            lines.append(f"- Required trace types: {target['required_trace_types']}.")
            lines.append("Likely contributor:")
            lines.append(f"- {target['diagnostic_hypothesis']}")
            lines.append("Claim boundary:")
            lines.append("- These traces show observed contributors and likely deviation sources; they do not prove the paper's unpublished computation path.")
            lines.append("")
    return "\n".join(lines)


def build_locked_config_manifest(results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    locked_by_dataset = {}
    feature_mappings = {}
    effective_by_dataset = {}
    for rec in results:
        dataset = str(rec.get("dataset", ""))
        if rec.get("status") != "ok" or not dataset:
            continue
        locked_by_dataset[dataset] = rec.get("locked_config", {})
        feature_mappings[dataset] = rec.get("feature_mapping", {})
        effective_by_dataset[dataset] = rec.get("effective_config", {})
    return {
        "locked_config_source": LOCKED_CONFIG_SOURCE,
        "hyper_tuning_source": HYPER_TUNING_SOURCE,
        "hyper_tuning_run_id": HYPER_TUNING_RUN_ID,
        "hyper_tuning_selection_criterion": HYPER_TUNING_SELECTION_CRITERION,
        "locked_config_selected_at": LOCKED_CONFIG_SELECTED_AT,
        "locked_config_values": locked_by_dataset,
        "effective_config_by_dataset": effective_by_dataset,
        "feature_order_by_dataset": feature_mappings,
        "locked_config_claim_boundary": LOCKED_CONFIG_CLAIM_BOUNDARY,
        "final_blindspot_bundle_usage": "separate blind-spot diagnostics only; not main Table 7 thesis reproduction",
    }


def _effective_field_map(results: Sequence[Dict[str, object]], field_name: str) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for rec in results:
        if rec.get("status") != "ok":
            continue
        dataset = str(rec.get("dataset", ""))
        effective = rec.get("effective_config", {})
        if dataset and isinstance(effective, dict):
            out[dataset] = effective.get(field_name)
    return out


def build_provenance(args, run_id: str, results: Sequence[Dict[str, object]], diagnostic_dir: str) -> Dict[str, object]:
    data_versions = {
        dataset: _sha256_file(os.path.join(args.data_dir, f"{dataset}.csv"))
        for dataset in ALL_DATASETS
    }
    model_versions = {
        dataset: _sha256_file(os.path.join("llm", "models", dataset, "model.joblib"))
        for dataset in ALL_DATASETS
    }
    locked_manifest = build_locked_config_manifest(results)
    return {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_status_short": _git_value(["status", "--short"]),
        "script_path": "scripts/final/part1/01b_reproduce_ufce_only.py",
        "diagnostic_dir": diagnostic_dir,
        "locked_config_source": LOCKED_CONFIG_SOURCE,
        "hyper_tuning_source": HYPER_TUNING_SOURCE,
        "hyper_tuning_run_id": HYPER_TUNING_RUN_ID,
        "hyper_tuning_selection_criterion": HYPER_TUNING_SELECTION_CRITERION,
        "locked_config_selected_at": LOCKED_CONFIG_SELECTED_AT,
        "locked_config_values": locked_manifest["locked_config_values"],
        "effective_config_by_dataset": locked_manifest["effective_config_by_dataset"],
        "effective_radius": _effective_field_map(results, "effective_radius"),
        "effective_n_neighbors": _effective_field_map(results, "effective_n_neighbors"),
        "effective_min_act": _effective_field_map(results, "effective_min_act"),
        "effective_min_feas": _effective_field_map(results, "effective_min_feas"),
        "effective_uf_source": _effective_field_map(results, "effective_uf_source"),
        "effective_f2change_source": _effective_field_map(results, "effective_f2change_source"),
        "effective_step_source": _effective_field_map(results, "effective_step_source"),
        "effective_bundle_mode": _effective_field_map(results, "effective_bundle_mode"),
        "fallback_used": _effective_field_map(results, "fallback_used"),
        "fallback_reason": _effective_field_map(results, "fallback_reason"),
        "not_main_table7": _effective_field_map(results, "not_main_table7"),
        "locked_config_claim_boundary": LOCKED_CONFIG_CLAIM_BOUNDARY,
        "final_blindspot_bundle_usage": "separate blind-spot diagnostics only; not main Table 7 thesis reproduction",
        "dataset_versions": data_versions,
        "model_versions": model_versions,
        "split_seed": 42,
        "author_table_reference_source": "AUTHOR_TABLE7 constant in scripts/final/part1/01b_reproduce_ufce_only.py",
        "metric_definitions": {
            "metrics": list(METRICS),
            "contract": "current UFCE-only Table 7 reproduction contract; APF metrics are fold-level counts over selected UFCE outputs",
            "relative_delta_epsilon": RELATIVE_DELTA_EPSILON,
        },
        "feature_order_by_dataset": locked_manifest["feature_order_by_dataset"],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": safe_version("numpy"),
            "pandas": safe_version("pandas"),
            "scikit-learn": safe_version("scikit-learn"),
        },
    }


def validate_apf_denominator_schema(apf_component_rows: Sequence[Dict[str, object]]) -> List[str]:
    errors: List[str] = []
    for idx, row in enumerate(apf_component_rows):
        missing = [field for field in APF_DENOMINATOR_REQUIRED_FIELDS if field not in row]
        if missing:
            errors.append(f"apf_denominator_schema_missing:{idx}:{','.join(missing)}")
            continue
        denominator_type = str(row.get("apf_denominator_type"))
        if denominator_type not in APF_DENOMINATOR_TYPES:
            errors.append(f"apf_denominator_type_unknown:{idx}:{denominator_type}")
        if row.get("apf_counted_in_metric") and row.get("apf_fold_denominator") is None:
            errors.append(f"apf_denominator_missing_for_counted_query:{idx}")
    return errors


def validate_candidate_role_schema(candidate_selection_rows: Sequence[Dict[str, object]]) -> List[str]:
    errors: List[str] = []
    for idx, row in enumerate(candidate_selection_rows):
        missing = [field for field in CANDIDATE_ROLE_REQUIRED_FIELDS if field not in row]
        if missing:
            errors.append(f"candidate_role_schema_missing:{idx}:{','.join(missing)}")
    return errors


def validate_diagnostic_artifacts(
    *,
    reconstruction_df: pd.DataFrame,
    top_targets: Sequence[Dict[str, object]],
    representative_cases: Sequence[Dict[str, object]],
    candidate_generation_rows: Sequence[Dict[str, object]],
    candidate_selection_rows: Sequence[Dict[str, object]],
    apf_component_rows: Sequence[Dict[str, object]] = (),
    provenance: Dict[str, object],
    fail_on_reconstruction_mismatch: bool,
) -> List[str]:
    errors: List[str] = []
    for key in [
        "locked_config_source",
        "hyper_tuning_source",
        "hyper_tuning_run_id",
        "hyper_tuning_selection_criterion",
        "locked_config_values",
        "locked_config_claim_boundary",
        "effective_config_by_dataset",
        "effective_radius",
        "effective_n_neighbors",
        "effective_min_act",
        "effective_min_feas",
        "effective_uf_source",
        "effective_f2change_source",
        "effective_step_source",
        "effective_bundle_mode",
        "fallback_used",
        "fallback_reason",
    ]:
        value = provenance.get(key)
        if value is None or value == "":
            errors.append(f"provenance_missing:{key}")

    errors.extend(validate_apf_denominator_schema(apf_component_rows))
    errors.extend(validate_candidate_role_schema(candidate_selection_rows))

    if fail_on_reconstruction_mismatch and not reconstruction_df.empty:
        bad = reconstruction_df[reconstruction_df["validation_ok"] != True]
        if not bad.empty:
            errors.append(f"metric_reconstruction_mismatch:{len(bad)}")

    gen_selected_ids = {
        str(row.get("selected_candidate_by_ufce_id"))
        for row in candidate_generation_rows
        if row.get("selected_candidate_by_ufce_id")
    }
    for row in candidate_selection_rows:
        selected_id = row.get("selected_candidate_by_ufce_id")
        if selected_id and str(selected_id) not in gen_selected_ids:
            errors.append(f"selected_candidate_missing_from_generation:{selected_id}")
        if row.get("metric_candidate_differs_from_ufce_selected") and not row.get("metric_candidate_explanation"):
            errors.append(f"metric_candidate_difference_unexplained:{row.get('query_id')}")

    cases_by_target = {(str(row.get("dataset")), int(row.get("target_rank", 0))) for row in representative_cases}
    for target in top_targets:
        key = (str(target.get("dataset")), int(target.get("rank", 0)))
        if key not in cases_by_target:
            errors.append(f"top_target_missing_representative_case:{key[0]}#{key[1]}")
    return errors


def write_diagnostic_artifacts(args, run_id: str, results: Sequence[Dict[str, object]]) -> str:
    diagnostic_dir = _diagnostic_output_path(args, run_id)
    os.makedirs(diagnostic_dir, exist_ok=True)

    delta_rows: List[Dict[str, object]] = []
    ok_results = [rec for rec in results if rec.get("status") == "ok"]
    for rec in ok_results:
        rows = rec.get("delta_rows", [])
        if isinstance(rows, list):
            delta_rows.extend(rows)

    long_df = pd.DataFrame(delta_rows)
    long_df.to_csv(os.path.join(diagnostic_dir, "table7_locked_reproduction_long.csv"), index=False)
    summary_df = build_metric_delta_summary(delta_rows)
    summary_df.to_csv(os.path.join(diagnostic_dir, "table7_metric_delta_summary.csv"), index=False)

    top_targets = select_top_metric_targets(delta_rows, int(args.top_targets_per_dataset))
    targets_df = pd.DataFrame(top_targets)
    targets_df.to_csv(os.path.join(diagnostic_dir, "table7_top2_metric_targets.csv"), index=False)

    records = _collect_diagnostic_records(ok_results)
    _write_jsonl(os.path.join(diagnostic_dir, "per_query_input_config_trace.jsonl"), records["input_config"])
    _write_jsonl(os.path.join(diagnostic_dir, "per_query_candidate_generation_trace.jsonl"), records["candidate_generation"])
    _write_jsonl(os.path.join(diagnostic_dir, "per_query_candidate_validation_trace.jsonl"), records["candidate_validation"])
    _write_jsonl(os.path.join(diagnostic_dir, "candidate_selection_trace.jsonl"), records["candidate_selection"])
    _write_jsonl(os.path.join(diagnostic_dir, "per_query_metric_trace.jsonl"), records["metric_trace"])
    _write_jsonl(os.path.join(diagnostic_dir, "feature_contribution_trace.jsonl"), records["feature_contribution"])
    _write_jsonl(os.path.join(diagnostic_dir, "apf_component_trace.jsonl"), records["apf_component"])

    reconstruction_df = reconstruct_metrics_from_trace(
        delta_rows=delta_rows,
        metric_trace_rows=records["metric_trace"],
        apf_trace_rows=records["apf_component"],
    )
    reconstruction_df.to_csv(os.path.join(diagnostic_dir, "metric_reconstruction_validation.csv"), index=False)

    representative_cases, representative_md = select_representative_diagnostic_cases(
        targets=top_targets,
        metric_trace_rows=records["metric_trace"],
        apf_trace_rows=records["apf_component"],
        candidate_generation_rows=records["candidate_generation"],
    )
    with open(os.path.join(diagnostic_dir, "representative_diagnostic_cases.md"), "w", encoding="utf-8") as f:
        f.write(representative_md)
    with open(os.path.join(diagnostic_dir, "representative_deviation_cases.md"), "w", encoding="utf-8") as f:
        f.write(representative_md)

    notes_md = build_deviation_notes(top_targets)
    with open(os.path.join(diagnostic_dir, "dataset_metric_deviation_notes.md"), "w", encoding="utf-8") as f:
        f.write(notes_md)

    locked_manifest = build_locked_config_manifest(ok_results)
    _write_json(os.path.join(diagnostic_dir, "locked_config_manifest.json"), locked_manifest)
    provenance = build_provenance(args, run_id, ok_results, diagnostic_dir)
    _write_json(os.path.join(diagnostic_dir, "provenance.json"), provenance)

    validation_errors = validate_diagnostic_artifacts(
        reconstruction_df=reconstruction_df,
        top_targets=top_targets,
        representative_cases=representative_cases,
        candidate_generation_rows=records["candidate_generation"],
        candidate_selection_rows=records["candidate_selection"],
        apf_component_rows=records["apf_component"],
        provenance=provenance,
        fail_on_reconstruction_mismatch=bool(args.fail_on_metric_reconstruction_mismatch),
    )

    report = {
        "run_id": run_id,
        "diagnostic_dir": diagnostic_dir,
        "datasets": [rec.get("dataset") for rec in ok_results],
        "status": "ok" if not validation_errors else "failed",
        "validation_errors": validation_errors,
        "artifact_counts": {key: len(value) for key, value in records.items()},
        "top_targets": top_targets,
        "representative_case_count": len(representative_cases),
    }
    _write_json(os.path.join(diagnostic_dir, "diagnostic_report.json"), report)
    with open(os.path.join(diagnostic_dir, "diagnostic_report.md"), "w", encoding="utf-8") as f:
        f.write("# Diagnostic Report\n\n")
        f.write(f"- run_id: {run_id}\n")
        f.write(f"- status: {report['status']}\n")
        f.write(f"- datasets: {', '.join(str(v) for v in report['datasets'])}\n")
        f.write(f"- representative diagnostic examples: {len(representative_cases)}\n")
        if validation_errors:
            f.write("\n## Validation Errors\n\n")
            for err in validation_errors:
                f.write(f"- {err}\n")
        f.write("\n## Claim Boundary\n\n")
        f.write(LOCKED_CONFIG_CLAIM_BOUNDARY + "\n")

    if validation_errors:
        raise RuntimeError("Diagnostic validation failed: " + "; ".join(validation_errors))
    return diagnostic_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime_profile",
        type=str,
        default="final_freeze",
        choices=["tuned_run2", "final_freeze"],
        help="Runtime config profile to use.",
    )
    parser.add_argument(
        "--bundle_mode",
        "--bundle-mode",
        dest="bundle_mode",
        type=str,
        default="table7_author_public",
        choices=["author_public", "table7_author_public", "final_blindspot_best"],
        help="UF/f2change/step bundle mode.",
    )
    parser.add_argument("--dataset", type=str, default="bank", choices=["bank", "grad", "wine", "bupa", "movie", "all"])
    parser.add_argument("--data_dir", type=str, default=os.path.join("ufce", "data"), help="Repo-relative path to data folder")
    parser.add_argument("--folds_dir", type=str, default=os.path.join("ufce", "data", "folds"), help="Repo-relative path to folds folder")
    parser.add_argument("--no_cf", type=int, default=10, help="Number of CFs requested per instance")
    parser.add_argument("--max_folds", type=int, default=0, help="0=all folds; otherwise limit folds")
    parser.add_argument(
        "--fold_file",
        type=str,
        default=None,
        help="Optional exact fold basename to run (e.g., testfold_0_pred_0.csv)",
    )
    parser.add_argument("--n_neighbors", type=int, default=None, help="Override tuned n_neighbors when provided")
    parser.add_argument("--radius", type=int, default=None, help="Override tuned radius when provided")
    parser.add_argument("--min_act", type=int, default=None, help="Override tuned min_act when provided")
    parser.add_argument("--min_feas", type=int, default=None, help="Override tuned min_feas when provided")
    parser.add_argument(
        "--ufce_flip_filter",
        type=int,
        default=0,
        choices=[0, 1],
        help="Override tuned UFCE flipping filter (0/1) when provided",
    )
    parser.add_argument("--debug", type=int, default=0, choices=[0, 1], help="Enable trust debug logs (0/1)")
    parser.add_argument(
        "--prox_euc_contract_debug",
        type=int,
        default=0,
        choices=[0, 1],
        help="Enable single-instance Prox-Euc contract debug block (0/1)",
    )
    parser.add_argument(
        "--contract_debug_fold",
        type=int,
        default=0,
        help="Fold index for Prox-Euc contract debug",
    )
    parser.add_argument(
        "--contract_debug_pos",
        type=int,
        default=0,
        help="Index position inside fold test list for Prox-Euc contract debug",
    )
    parser.add_argument(
        "--contract_debug_method",
        type=str,
        default="UFCE1",
        choices=METHODS,
        help="UFCE method for Prox-Euc contract debug (UFCE1/UFCE2/UFCE3)",
    )
    parser.add_argument("--contprox_metric", type=str, default="euclidean", help="ContProximity metric to use in UFCE")
    parser.add_argument(
        "--out_dir",
        "--out-dir",
        dest="out_dir",
        type=str,
        default=os.path.join("archive", "part1_old_runs", "repro_v3_out"),
        help="Output directory for plots/tables",
    )
    parser.add_argument("--diagnostics", action="store_true", help="Emit Table 7 deviation diagnostics sidecar artifacts.")
    parser.add_argument(
        "--diagnostic-out-dir",
        dest="diagnostic_out_dir",
        type=str,
        default=DIAGNOSTIC_DEFAULT_ROOT,
        help="Output root for diagnostics run directories.",
    )
    parser.add_argument("--top-k-candidates", dest="top_k_candidates", type=int, default=5)
    parser.add_argument("--top-targets-per-dataset", dest="top_targets_per_dataset", type=int, default=2)
    parser.add_argument("--emit-per-query-trace", action="store_true", default=True)
    parser.add_argument("--emit-representative-cases", action="store_true", default=True)
    parser.add_argument(
        "--fail-on-metric-reconstruction-mismatch",
        action="store_true",
        default=True,
        help="Fail diagnostics if trace reconstruction differs from reported reproduction values.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    print_env_versions()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.run_id = run_id
    validate_diagnostic_args(args)

    manifest_path = write_run_manifest(args, run_id, args.out_dir)
    print(f"- Run manifest saved: {manifest_path}")

    if args.dataset != "all":
        rec = run_for_dataset(args.dataset, args, run_id)
        if bool(args.diagnostics):
            diagnostic_dir = write_diagnostic_artifacts(args, run_id, [rec])
            print(f"- Diagnostics saved: {diagnostic_dir}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    batch_records: List[Dict[str, object]] = []
    batch_t0 = time.time()
    for dataset in ALL_DATASETS:
        ds_t0 = time.time()
        print(f"\n==================== BATCH DATASET: {dataset} ====================")
        try:
            rec = run_for_dataset(dataset, args, run_id)
        except Exception as exc:
            rec = {
                "dataset": dataset,
                "status": "failed",
                "runtime_sec": float(time.time() - ds_t0),
                "n_folds": 0,
                "error": f"{type(exc).__name__}: {exc}",
                "summary_csv": "",
                "plot_path": "",
            }
            print(f"[ERROR] dataset={dataset} failed with {type(exc).__name__}: {exc}")
        batch_records.append(rec)

    batch_df = pd.DataFrame(
        batch_records,
        columns=["dataset", "status", "runtime_sec", "n_folds", "error", "summary_csv", "plot_path"],
    )
    batch_csv = os.path.join(args.out_dir, f"batch_summary_{run_id}.csv")
    batch_df.to_csv(batch_csv, index=False)

    print("\n==================== BATCH SUMMARY ====================")
    print(batch_df.to_string(index=False))
    print(f"- Batch summary CSV saved: {batch_csv}")
    print(f"- Batch total runtime: {time.time() - batch_t0:0.2f}s")

    if bool(args.diagnostics):
        diagnostic_dir = write_diagnostic_artifacts(args, run_id, batch_records)
        print(f"- Diagnostics saved: {diagnostic_dir}")

    failed_count = int(np.sum(batch_df["status"].values == "failed"))
    if failed_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
