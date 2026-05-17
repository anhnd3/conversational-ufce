#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from scripts.archieve.reproduce_full_table7_result import (
    TUNED_RUN2,
    apply_distance_scaler,
    build_movie_distance_scaler,
    get_constraints,
    init_ufce_global,
    mad_inverse_inplace,
    mad_transform,
)
from ufce import UFCE
from ufce.core.cfmethods import sfexp, dfexp, tfexp
from ufce.core.data_processing import classify_dataset_getModel
from ufce.core.evaluations import (
    Actionability,
    Catproximity,
    Contproximity,
    Feasibility,
    Plausibility,
    Sparsity,
)


pd.set_option("display.max_columns", None)

METHODS = ["UFCE1", "UFCE2", "UFCE3"]
METRICS = ["prox_jac", "prox_euc", "sparsity", "actionability", "plausibility", "feasibility"]
DEFAULT_DATASETS = ["bank", "grad", "wine", "bupa", "movie"]
ALL_STAGES = ["provenance", "single", "joint", "strict"]
ANCHOR_PUBLIC_NAME = "author_public_bundle"
ANCHOR_AUTHOR_NAME = "author_preset_bundle"
SKIPPED_VARIANT_COLUMNS = [
    "dataset",
    "variant_name",
    "variant_family",
    "uf_mode",
    "step_mode",
    "f2change_mode",
    "flip_filter",
    "skip_reason",
    "missing_step_keys",
]
UF_MODE_NAMES = {
    "author_public",
    "neutral_all_1",
    "scaled_up_150",
    "scaled_down_50",
    "rank_inverted",
    "rank_permuted_seed123",
}
STEP_MODE_NAMES = {
    "local_reproduction",
    "author_preset",
    "visible_experiment",
    "finer_half",
    "coarser_double",
    "neutral_common_1",
}
F2CHANGE_MODE_NAMES = {
    "author_public",
    "all_features",
    "minus_top_1",
    "minus_top_2",
    "numeric_only",
    "categorical_only",
}

DATA_PROCESSING_SOURCE = ROOT / "ufce" / "core" / "data_processing.py"
CFMETHODS_SOURCE = ROOT / "ufce" / "core" / "cfmethods.py"
UFCE_SOURCE = ROOT / "ufce" / "core" / "ufce.py"
EXPERIMENTS_SOURCE = ROOT / "ufce" / "core" / "experiments.py"
REPRO_SOURCE = ROOT / "scripts" / "reproduce_full_table7_result.py"

CONSTRAINT_FUNCTIONS = {
    "bank": "get_bank_user_constraints",
    "grad": "get_grad_user_constraints",
    "wine": "get_wine_user_constraints",
    "bupa": "get_bupa_user_constraints",
    "movie": "get_movie_user_constraints",
}

METHOD_TO_TRACE_NAME = {
    "UFCE1": "single_feature",
    "UFCE2": "double_feature",
    "UFCE3": "triple_feature",
}


@dataclass(frozen=True)
class VariantSpec:
    variant_name: str
    variant_family: str
    uf_mode: str
    step_mode: str
    f2change_mode: str
    flip_filter: int
    uf: Dict[str, float]
    step: Dict[str, float]
    f2change: List[str]


@dataclass
class DatasetContext:
    dataset: str
    datasetdf: pd.DataFrame
    lr: Any
    lr_mean: float
    lr_std: float
    Xtest: pd.DataFrame
    Xtrain: pd.DataFrame
    X: pd.DataFrame
    features: List[str]
    catf: List[str]
    numf: List[str]
    author_uf: Dict[str, float]
    author_f2change: List[str]
    author_step: Dict[str, float]
    local_step: Dict[str, float]
    visible_step: Optional[Dict[str, float]]
    protectf: List[str]
    outcome_label: str
    desired_outcome: float
    data_lab1: pd.DataFrame
    mi_pairs_top5: List[List[str]]
    scaler_ar: StandardScaler
    ufce_mad_scaler: Optional[Dict[str, pd.Series]]
    movie_distance_scaler: Optional[Dict[str, object]]
    tuned_cfg: Dict[str, int]
    contprox_metric: str
    provenance: Dict[str, Any]


@dataclass
class ProgressTracker:
    total_units: int
    start_time: float
    completed_units: int = 0
    progress_bar: Optional[Any] = None

    def __post_init__(self) -> None:
        if tqdm is not None and (sys.stderr.isatty() or sys.stdout.isatty()):
            self.progress_bar = tqdm(
                total=int(self.total_units),
                desc="UFCE ablation",
                unit="variant",
                dynamic_ncols=True,
                mininterval=1.0,
                smoothing=0.1,
                leave=True,
            )

    def extend_total(self, delta_units: int, *, reason: str) -> None:
        if int(delta_units) <= 0:
            return
        self.total_units += int(delta_units)
        if self.progress_bar is not None:
            self.progress_bar.total = int(self.total_units)
            self.progress_bar.refresh()
        print(
            "[ABLATE][MASTER-TOTAL] "
            f"reason={reason} completed={self.completed_units} total={self.total_units}",
            flush=True,
        )

    def record_completion(
        self,
        *,
        dataset: str,
        variant_name: str,
        stage: str,
        last_seconds: float,
    ) -> None:
        self.completed_units += 1
        elapsed = time.perf_counter() - self.start_time
        avg_seconds = elapsed / self.completed_units if self.completed_units > 0 else 0.0
        remaining_units = max(int(self.total_units) - int(self.completed_units), 0)
        eta_seconds = avg_seconds * remaining_units
        percent_complete = (100.0 * self.completed_units / self.total_units) if self.total_units > 0 else 100.0
        if self.progress_bar is not None:
            self.progress_bar.update(1)
            self.progress_bar.set_postfix_str(
                f"{dataset}:{stage} eta={_format_duration(eta_seconds)}",
                refresh=False,
            )
        print(
            "[ABLATE][MASTER] "
            f"stage={stage} completed={self.completed_units}/{self.total_units} "
            f"pct={percent_complete:.1f}% elapsed={_format_duration(elapsed)} "
            f"eta={_format_duration(eta_seconds)} avg_per_variant={avg_seconds:.2f}s "
            f"last_variant={last_seconds:.2f}s dataset={dataset} variant={variant_name}",
            flush=True,
        )

    def close(self) -> None:
        if self.progress_bar is not None:
            self.progress_bar.close()


def _normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _resolve_output_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return ROOT / path


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_rate(numerator: float, denominator: float) -> float:
    if float(denominator) == 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def _format_duration(seconds: float) -> str:
    total_seconds = max(int(round(float(seconds))), 0)
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return _to_builtin(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _to_builtin(value.to_dict())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _json_dump(value: Any) -> str:
    return json.dumps(_to_builtin(value), ensure_ascii=False, sort_keys=True)


def _dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _dicts_exact_match(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    return left == right


def _dicts_normalized_match(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    left_norm = {_normalize_key(k): v for k, v in left.items()}
    right_norm = {_normalize_key(k): v for k, v in right.items()}
    return left_norm == right_norm


def _list_normalized_match(left: Sequence[str], right: Sequence[str]) -> bool:
    return [_normalize_key(v) for v in left] == [_normalize_key(v) for v in right]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_to_builtin(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_to_builtin(row), ensure_ascii=False) + "\n")


def _load_ast(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _find_function(module: ast.AST, function_name: str, class_name: Optional[str] = None) -> ast.FunctionDef:
    if class_name is None:
        for node in getattr(module, "body", []):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return node
    for node in getattr(module, "body", []):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return child
    raise ValueError(f"Function not found: {class_name + '.' if class_name else ''}{function_name}")


def _literal_eval_node(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def extract_layer_a_constraints() -> Dict[str, Dict[str, Any]]:
    module = _load_ast(DATA_PROCESSING_SOURCE)
    out: Dict[str, Dict[str, Any]] = {}
    for dataset, function_name in CONSTRAINT_FUNCTIONS.items():
        fn = _find_function(module, function_name)
        values: Dict[str, Any] = {}
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assign):
                continue
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            target = node.targets[0].id
            if target not in {"uf", "step", "f2change", "protectf"}:
                continue
            literal = _literal_eval_node(node.value)
            if literal is not None:
                values[target] = literal
        out[dataset] = values
    return out


def _extract_dataset_comparison(test: ast.AST) -> Optional[str]:
    if not isinstance(test, ast.Compare):
        return None
    if not isinstance(test.left, ast.Name) or test.left.id != "dataset":
        return None
    if len(test.ops) != 1 or len(test.comparators) != 1:
        return None
    if not isinstance(test.ops[0], ast.Eq):
        return None
    comparator = test.comparators[0]
    if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
        return comparator.value
    return None


def _extract_return_dict_from_body(body: Sequence[ast.stmt]) -> Optional[Dict[str, float]]:
    for stmt in body:
        if isinstance(stmt, ast.Return):
            literal = _literal_eval_node(stmt.value)
            if isinstance(literal, dict):
                return literal
    return None


def extract_layer_c_steps() -> Dict[str, Dict[str, float]]:
    module = _load_ast(REPRO_SOURCE)
    fn = _find_function(module, "get_step_config")
    out: Dict[str, Dict[str, float]] = {}

    def walk_if(node: ast.If) -> None:
        dataset = _extract_dataset_comparison(node.test)
        if dataset is not None:
            returned = _extract_return_dict_from_body(node.body)
            if isinstance(returned, dict):
                out[dataset] = returned
        for child in node.orelse:
            if isinstance(child, ast.If):
                walk_if(child)

    for stmt in fn.body:
        if isinstance(stmt, ast.If):
            walk_if(stmt)
    return out


def extract_visible_experiment_step() -> Optional[Dict[str, float]]:
    module = _load_ast(EXPERIMENTS_SOURCE)
    assignments: List[Tuple[int, Dict[str, float]]] = []
    for node in ast.walk(module):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if node.targets[0].id != "step":
            continue
        literal = _literal_eval_node(node.value)
        if isinstance(literal, dict):
            assignments.append((int(getattr(node, "lineno", 0)), literal))
    if not assignments:
        return None
    assignments.sort(key=lambda item: item[0])
    return assignments[-1][1]


def _count_loaded_name(fn: ast.FunctionDef, argument_name: str) -> int:
    count = 0
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == argument_name:
            count += 1
    return count


def build_role_scan() -> Dict[str, bool]:
    cfmethods_module = _load_ast(CFMETHODS_SOURCE)
    ufce_module = _load_ast(UFCE_SOURCE)
    sfexp_fn = _find_function(cfmethods_module, "sfexp")
    dfexp_fn = _find_function(cfmethods_module, "dfexp")
    tfexp_fn = _find_function(cfmethods_module, "tfexp")
    single_f_fn = _find_function(ufce_module, "Single_F", class_name="UFCE")
    double_f_fn = _find_function(ufce_module, "Double_F", class_name="UFCE")
    triple_f_fn = _find_function(ufce_module, "Triple_F", class_name="UFCE")
    actionability_fn = _find_function(ufce_module, "actionability", class_name="UFCE")
    feasibility_fn = _find_function(ufce_module, "feasibility", class_name="UFCE")

    return {
        "ufce1_generation_uses_step": _count_loaded_name(sfexp_fn, "step") > 0 and _count_loaded_name(single_f_fn, "step") > 0,
        "ufce1_generation_uses_f2change": _count_loaded_name(sfexp_fn, "f2change") > 0,
        "ufce2_generation_uses_f2change": _count_loaded_name(dfexp_fn, "features") > 0 and _count_loaded_name(double_f_fn, "features") > 0,
        "ufce3_generation_uses_f2change": _count_loaded_name(tfexp_fn, "feature2change") > 0 and _count_loaded_name(triple_f_fn, "features_2change") > 0,
        "actionability_uses_uf": _count_loaded_name(actionability_fn, "uf") > 0,
        "actionability_uses_f2change": _count_loaded_name(actionability_fn, "changeable_features") > 0,
        "feasibility_uses_uf": _count_loaded_name(feasibility_fn, "uf") > 0,
        "feasibility_uses_f2change": _count_loaded_name(feasibility_fn, "changeable_features") > 0,
    }


def extract_provenance_records(data_dir: str | Path, datasets: Sequence[str]) -> List[Dict[str, Any]]:
    data_dir = _resolve_output_path(str(data_dir))
    layer_a = extract_layer_a_constraints()
    layer_c = extract_layer_c_steps()
    layer_b_step = extract_visible_experiment_step()
    role_scan = build_role_scan()

    records: List[Dict[str, Any]] = []
    for dataset in datasets:
        datasetdf = pd.read_csv(data_dir / f"{dataset}.csv")
        (
            features,
            _catf,
            _numf,
            _uf,
            _f2change,
            _outcome_label,
            _desired_outcome,
            _nbr_features,
            _protectf,
            _data_lab0,
            _data_lab1,
        ) = get_constraints(dataset, datasetdf)
        feature_keys = {_normalize_key(feature) for feature in features}
        author_values = layer_a.get(dataset, {})
        author_step = dict(author_values.get("step", {}))
        local_step = dict(layer_c.get(dataset, {}))
        author_f2change = list(author_values.get("f2change", list(_f2change)))
        local_f2change = list(_f2change)
        layer_a_missing = [feature for feature in author_f2change if feature not in author_step]
        layer_c_missing = [feature for feature in local_f2change if feature not in local_step]
        layer_b_available = False
        visible_step: Optional[Dict[str, float]] = None
        notes: List[str] = []
        if isinstance(layer_b_step, dict):
            visible_keys = {_normalize_key(key) for key in layer_b_step.keys()}
            if visible_keys == feature_keys:
                layer_b_available = True
                visible_step = dict(layer_b_step)
            else:
                notes.append("visible_experiment_step_skipped_due_to_feature_mismatch")
        record = {
            "dataset": dataset,
            "layer_a_uf": _json_dump(author_values.get("uf", {})),
            "layer_a_step": _json_dump(author_step),
            "layer_a_f2change": _json_dump(author_f2change),
            "layer_a_protectf": _json_dump(author_values.get("protectf", [])),
            "layer_b_step": _json_dump(visible_step) if visible_step is not None else "",
            "layer_b_available": bool(layer_b_available),
            "layer_c_step": _json_dump(local_step),
            "layer_a_step_missing_for_f2change": _json_dump(layer_a_missing),
            "layer_c_step_missing_for_f2change": _json_dump(layer_c_missing),
            "layer_a_step_runnable_for_f2change": len(layer_a_missing) == 0,
            "layer_c_step_runnable_for_f2change": len(layer_c_missing) == 0,
            "exact_match": _dicts_exact_match(author_step, local_step),
            "normalized_match": _dicts_normalized_match(author_step, local_step),
            "layer_b_matches_dataset_exact": visible_step == local_step if visible_step is not None else False,
            "layer_b_matches_dataset_normalized": _dicts_normalized_match(visible_step, local_step) if visible_step is not None else False,
            "notes": ";".join(notes),
        }
        record.update(role_scan)
        records.append(record)
    return records


def _parse_dataset_list(raw: str) -> List[str]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    allowed = set(DEFAULT_DATASETS)
    invalid = [item for item in values if item not in allowed]
    if invalid:
        raise ValueError(f"Unsupported dataset(s): {invalid}. Allowed: {sorted(allowed)}")
    return values if values else list(DEFAULT_DATASETS)


def _parse_stage_list(raw: str) -> List[str]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values or "all" in values:
        return list(ALL_STAGES)
    invalid = [item for item in values if item not in ALL_STAGES]
    if invalid:
        raise ValueError(f"Unsupported stage(s): {invalid}. Allowed: {ALL_STAGES + ['all']}")
    expanded: List[str] = []
    if "strict" in values:
        for stage in ["single", "joint", "strict"]:
            if stage not in expanded:
                expanded.append(stage)
    for stage in values:
        if stage not in expanded:
            expanded.append(stage)
    return expanded


def _parse_variant_filter(raw: Optional[str]) -> set[str]:
    if raw is None:
        return set()
    return {item.strip() for item in str(raw).split(",") if item.strip()}


def build_dataset_context(
    dataset: str,
    data_dir: str | Path,
    provenance_map: Dict[str, Dict[str, Any]],
    *,
    contprox_metric: str,
) -> DatasetContext:
    data_dir = _resolve_output_path(str(data_dir))
    raw_df = pd.read_csv(data_dir / f"{dataset}.csv")
    out = classify_dataset_getModel(raw_df.copy(), data_name=dataset)
    if len(out) == 8:
        lr, lr_mean, lr_std, Xtest, Xtrain, X, _Y, datasetdf = out
        scaler_mad_from_func = None
    else:
        lr, lr_mean, lr_std, Xtest, Xtrain, X, _Y, datasetdf, scaler_mad_from_func = out

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
    ) = get_constraints(dataset, datasetdf)

    movie_distance_scaler = None
    if dataset == "movie":
        movie_distance_scaler = build_movie_distance_scaler(
            datasetdf=datasetdf,
            features=features,
            numf=numf,
            outcome_label=outcome_label,
        )

    ufc = UFCE()
    mi_pairs_top5 = ufc.get_top_MI_features(X, features)[:5]
    provenance = provenance_map[dataset]
    visible_step = None
    if provenance.get("layer_b_available"):
        visible_step = json.loads(str(provenance.get("layer_b_step", "")))

    return DatasetContext(
        dataset=dataset,
        datasetdf=datasetdf,
        lr=lr,
        lr_mean=float(lr_mean),
        lr_std=float(lr_std),
        Xtest=Xtest,
        Xtrain=Xtrain,
        X=X,
        features=list(features),
        catf=list(catf),
        numf=list(numf),
        author_uf=dict(uf),
        author_f2change=list(f2change),
        author_step=json.loads(str(provenance["layer_a_step"])),
        local_step=json.loads(str(provenance["layer_c_step"])),
        visible_step=visible_step,
        protectf=list(protectf),
        outcome_label=outcome_label,
        desired_outcome=float(desired_outcome),
        data_lab1=data_lab1,
        mi_pairs_top5=[list(pair) for pair in mi_pairs_top5],
        scaler_ar=StandardScaler().fit(Xtrain[:]),
        ufce_mad_scaler=scaler_mad_from_func,
        movie_distance_scaler=movie_distance_scaler,
        tuned_cfg=dict(TUNED_RUN2[dataset]),
        contprox_metric=str(contprox_metric),
        provenance=provenance,
    )


def _type_preserving_one(value: Any) -> float | int:
    try:
        value_float = float(value)
    except Exception:
        return 1
    if float(value_float).is_integer():
        return 1
    return 1.0


def build_uf_variants(author_uf: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    keys = list(author_uf.keys())
    sorted_by_value = sorted(keys, key=lambda key: (float(author_uf[key]), key))
    sorted_values = [author_uf[key] for key in sorted_by_value]
    inverted_values = list(reversed(sorted_values))
    rng = random.Random(123)
    permuted_values = list(sorted_values)
    rng.shuffle(permuted_values)
    return {
        "author_public": dict(author_uf),
        "neutral_all_1": {key: _type_preserving_one(author_uf[key]) for key in keys},
        "scaled_up_150": {key: float(author_uf[key]) * 1.5 for key in keys},
        "scaled_down_50": {key: float(author_uf[key]) * 0.5 for key in keys},
        "rank_inverted": {key: inverted_values[idx] for idx, key in enumerate(sorted_by_value)},
        "rank_permuted_seed123": {key: permuted_values[idx] for idx, key in enumerate(sorted_by_value)},
    }


def build_step_variants(context: DatasetContext) -> Dict[str, Dict[str, float]]:
    variants = {
        "local_reproduction": dict(context.local_step),
        "author_preset": dict(context.author_step),
        "finer_half": {key: float(value) * 0.5 for key, value in context.local_step.items()},
        "coarser_double": {key: float(value) * 2.0 for key, value in context.local_step.items()},
        "neutral_common_1": {key: _type_preserving_one(value) for key, value in context.local_step.items()},
    }
    if isinstance(context.visible_step, dict):
        variants["visible_experiment"] = dict(context.visible_step)
    return variants


def validate_variant_step_coverage(context: DatasetContext, variant: VariantSpec) -> List[str]:
    return [feature for feature in variant.f2change if feature not in variant.step]


def _build_skip_record(context: DatasetContext, variant: VariantSpec, missing_step_keys: Sequence[str]) -> Dict[str, Any]:
    missing = [str(feature) for feature in missing_step_keys]
    reason = "missing_step_keys:" + ",".join(missing) if missing else "unknown"
    return {
        "dataset": context.dataset,
        "variant_name": variant.variant_name,
        "variant_family": variant.variant_family,
        "uf_mode": variant.uf_mode,
        "step_mode": variant.step_mode,
        "f2change_mode": variant.f2change_mode,
        "flip_filter": int(variant.flip_filter),
        "skip_reason": reason,
        "missing_step_keys": ",".join(missing),
    }


def standardized_feature_ranking(context: DatasetContext, allowed_features: Sequence[str]) -> List[str]:
    coef = np.asarray(context.lr.coef_, dtype=float)
    if coef.ndim == 2:
        coef_vec = coef[0]
    else:
        coef_vec = coef.reshape(-1)
    ranking: List[Tuple[float, str]] = []
    for feature in allowed_features:
        if feature not in context.Xtrain.columns:
            continue
        col_idx = context.Xtrain.columns.get_loc(feature)
        score = abs(float(coef_vec[col_idx]) * float(context.Xtrain[feature].std()))
        ranking.append((score, feature))
    ranking.sort(key=lambda item: (-item[0], item[1]))
    return [feature for _, feature in ranking]


def build_f2change_variants(context: DatasetContext) -> Dict[str, List[str]]:
    ranked = standardized_feature_ranking(context, context.author_f2change)
    minus_top_1 = [feature for feature in context.author_f2change if feature not in set(ranked[:1])]
    minus_top_2 = [feature for feature in context.author_f2change if feature not in set(ranked[:2])]
    variants = {
        "author_public": list(context.author_f2change),
        "all_features": list(context.features),
        "minus_top_1": minus_top_1,
        "minus_top_2": minus_top_2,
        "numeric_only": [feature for feature in context.author_f2change if feature in context.numf],
        "categorical_only": [feature for feature in context.author_f2change if feature in context.catf],
    }
    if len(variants["categorical_only"]) == 0:
        variants.pop("categorical_only")
    return {name: _dedupe_preserve_order(values) for name, values in variants.items()}


def build_anchor_specs(context: DatasetContext) -> List[VariantSpec]:
    return [
        VariantSpec(
            variant_name=ANCHOR_PUBLIC_NAME,
            variant_family="anchor",
            uf_mode="author_public",
            step_mode="local_reproduction",
            f2change_mode="author_public",
            flip_filter=0,
            uf=dict(context.author_uf),
            step=dict(context.local_step),
            f2change=list(context.author_f2change),
        ),
        VariantSpec(
            variant_name=ANCHOR_AUTHOR_NAME,
            variant_family="anchor",
            uf_mode="author_public",
            step_mode="author_preset",
            f2change_mode="author_public",
            flip_filter=0,
            uf=dict(context.author_uf),
            step=dict(context.author_step),
            f2change=list(context.author_f2change),
        ),
    ]


def build_single_factor_specs(context: DatasetContext) -> List[VariantSpec]:
    specs: List[VariantSpec] = []
    uf_variants = build_uf_variants(context.author_uf)
    for uf_mode, uf_value in uf_variants.items():
        specs.append(
            VariantSpec(
                variant_name=f"uf__{uf_mode}",
                variant_family="uf",
                uf_mode=uf_mode,
                step_mode="local_reproduction",
                f2change_mode="author_public",
                flip_filter=0,
                uf=dict(uf_value),
                step=dict(context.local_step),
                f2change=list(context.author_f2change),
            )
        )
    step_variants = build_step_variants(context)
    for step_mode, step_value in step_variants.items():
        specs.append(
            VariantSpec(
                variant_name=f"step__{step_mode}",
                variant_family="step",
                uf_mode="author_public",
                step_mode=step_mode,
                f2change_mode="author_public",
                flip_filter=0,
                uf=dict(context.author_uf),
                step=dict(step_value),
                f2change=list(context.author_f2change),
            )
        )
    f2change_variants = build_f2change_variants(context)
    for f2change_mode, f2change_value in f2change_variants.items():
        specs.append(
            VariantSpec(
                variant_name=f"f2change__{f2change_mode}",
                variant_family="f2change",
                uf_mode="author_public",
                step_mode="local_reproduction",
                f2change_mode=f2change_mode,
                flip_filter=0,
                uf=dict(context.author_uf),
                step=dict(context.local_step),
                f2change=list(f2change_value),
            )
        )
    return specs


def build_joint_specs(context: DatasetContext) -> List[VariantSpec]:
    uf_variants = build_uf_variants(context.author_uf)
    step_variants = build_step_variants(context)
    f2change_variants = build_f2change_variants(context)
    combinations = [
        ("neutral_all_1", "local_reproduction", "author_public"),
        ("author_public", "neutral_common_1", "author_public"),
        ("author_public", "local_reproduction", "all_features"),
        ("neutral_all_1", "neutral_common_1", "author_public"),
        ("author_public", "coarser_double", "minus_top_1"),
    ]
    specs: List[VariantSpec] = []
    for uf_mode, step_mode, f2change_mode in combinations:
        specs.append(
            VariantSpec(
                variant_name=f"joint__{uf_mode}__{step_mode}__{f2change_mode}",
                variant_family="joint",
                uf_mode=uf_mode,
                step_mode=step_mode,
                f2change_mode=f2change_mode,
                flip_filter=0,
                uf=dict(uf_variants[uf_mode]),
                step=dict(step_variants[step_mode]),
                f2change=list(f2change_variants[f2change_mode]),
            )
        )
    return specs


def _variant_passes_filter(spec: VariantSpec, allowed: set[str]) -> bool:
    if not allowed or spec.variant_family == "anchor":
        return True
    if spec.variant_name in allowed:
        return True
    for token in allowed:
        if token in UF_MODE_NAMES:
            if spec.variant_family == "uf" and spec.uf_mode == token:
                return True
            continue
        if token in STEP_MODE_NAMES:
            if spec.variant_family == "step" and spec.step_mode == token:
                return True
            continue
        if token in F2CHANGE_MODE_NAMES:
            if spec.variant_family == "f2change" and spec.f2change_mode == token:
                return True
            continue
    return False


def _prepare_fold_search_frames(
    context: DatasetContext,
    fold_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict[str, object]]]:
    testset_features = fold_df.loc[:, context.features].copy()
    if context.dataset == "movie" and context.movie_distance_scaler is not None:
        data_lab1_ufce = context.data_lab1.loc[:, context.features].copy()
        testset_dist = apply_distance_scaler(testset_features, context.movie_distance_scaler)
        data_lab1_dist = apply_distance_scaler(data_lab1_ufce, context.movie_distance_scaler)
        return testset_features, data_lab1_ufce, testset_dist, data_lab1_dist, context.movie_distance_scaler
    if context.ufce_mad_scaler is not None:
        data_lab1_ufce = mad_transform(context.data_lab1.loc[:, context.features].copy(), context.ufce_mad_scaler, context.numf)
        testset_ufce = mad_transform(testset_features, context.ufce_mad_scaler, context.numf)
        return testset_ufce, data_lab1_ufce, None, None, None
    return testset_features, context.data_lab1.loc[:, context.features].copy(), None, None, None


def _inverse_trace_rows_inplace(trace_rows: List[dict], scaler: Dict[str, pd.Series], numf: Sequence[str]) -> None:
    for row in trace_rows:
        for key in ["generated_candidates_df", "label_flip_candidates_df", "selected_candidates_df"]:
            frame = row.get(key)
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                mad_inverse_inplace(frame, scaler, list(numf))


def _build_fallback_trace_rows(
    method_name: str,
    selected_df: pd.DataFrame,
    found_idx: Sequence[int],
    query_count: int,
    order: Sequence[str],
) -> List[dict]:
    idx_to_pos = {int(idx): pos for pos, idx in enumerate(found_idx)}
    trace_rows: List[dict] = []
    for query_pos in range(int(query_count)):
        if query_pos in idx_to_pos:
            selected = selected_df.iloc[idx_to_pos[query_pos] : idx_to_pos[query_pos] + 1].copy()
        else:
            selected = pd.DataFrame(columns=list(order))
        trace_rows.append(
            {
                "instance_pos": int(query_pos),
                "method": method_name,
                "generated_candidates_df": pd.DataFrame(columns=list(order)),
                "label_flip_candidates_df": pd.DataFrame(columns=list(order)),
                "selected_candidates_df": selected,
                "search_meta": {},
                "search_parameters": {},
                "source_path": METHOD_TO_TRACE_NAME[method_name],
            }
        )
    return trace_rows


def _call_with_trace_or_fallback(
    method_name: str,
    func: Any,
    *args: Any,
    order: Sequence[str],
    **kwargs: Any,
) -> Tuple[pd.DataFrame, float, List[int], Dict[str, Any], List[dict], bool]:
    try:
        selected_df, elapsed, found_idx, stats, trace_rows = func(
            *args,
            return_stats=True,
            return_trace=True,
            **kwargs,
        )
        return selected_df, float(elapsed), [int(idx) for idx in found_idx], dict(stats), list(trace_rows), True
    except TypeError:
        selected_df, elapsed, found_idx = func(*args, **kwargs)
        fallback_stats = {
            "n_instances": int(len(args[2])) if len(args) > 2 else 0,
            "n_candidates_raw_total": 0,
            "n_candidates_flip_total": int(len(selected_df)),
            "n_instances_with_flip_cf": int(len(found_idx)),
            "n_empty_after_filter": 0,
            "coverage": _safe_rate(len(found_idx), len(args[2])) if len(args) > 2 else 0.0,
        }
        fallback_trace_rows = _build_fallback_trace_rows(
            method_name,
            selected_df=selected_df,
            found_idx=found_idx,
            query_count=len(args[2]) if len(args) > 2 else 0,
            order=order,
        )
        return selected_df, float(elapsed), [int(idx) for idx in found_idx], fallback_stats, fallback_trace_rows, False


def _candidate_prediction(model: Any, candidate: Optional[pd.DataFrame], desired_outcome: float) -> Optional[int]:
    if not isinstance(candidate, pd.DataFrame) or candidate.empty:
        return None
    pred = np.asarray(model.predict(candidate)).reshape(-1)
    if pred.size == 0:
        return None
    return int(pred[0])


def _value_changed(left: Any, right: Any, atol: float = 1e-5) -> bool:
    try:
        return not np.isclose(float(left), float(right), atol=atol, rtol=0.0)
    except Exception:
        return str(left) != str(right)


def _summarize_trace_rows(
    *,
    context: DatasetContext,
    variant: VariantSpec,
    fold_name: str,
    fold_df: pd.DataFrame,
    method_name: str,
    trace_rows: Sequence[dict],
) -> List[dict]:
    fold_features = fold_df.loc[:, context.features].reset_index(drop=True).copy()
    summaries: List[dict] = []
    for trace_row in trace_rows:
        query_pos = int(trace_row.get("instance_pos", 0))
        factual = fold_features.iloc[query_pos : query_pos + 1].copy()
        generated = trace_row.get("generated_candidates_df")
        flip_candidates = trace_row.get("label_flip_candidates_df")
        selected = trace_row.get("selected_candidates_df")
        if not isinstance(generated, pd.DataFrame):
            generated = pd.DataFrame(columns=context.features)
        if not isinstance(flip_candidates, pd.DataFrame):
            flip_candidates = pd.DataFrame(columns=context.features)
        if not isinstance(selected, pd.DataFrame):
            selected = pd.DataFrame(columns=context.features)

        generated_count = int(len(generated))
        flip_count = int(len(flip_candidates))
        selected_count = int(len(selected))
        selected_prediction = _candidate_prediction(context.lr, selected.iloc[:1].copy(), context.desired_outcome)

        representative_source = None
        representative = pd.DataFrame(columns=context.features)
        if selected_count > 0:
            representative_source = "selected"
            representative = selected.iloc[:1].copy()
        elif generated_count > 0:
            representative_source = "generated"
            representative = generated.iloc[:1].copy()

        if selected_count > 0 and selected_prediction == int(context.desired_outcome):
            status = "valid"
        elif generated_count > 0:
            status = "invalid"
        else:
            status = "empty"

        changed_features: List[str] = []
        feature_deltas: Dict[str, float] = {}
        changed_numeric_count = 0
        changed_categorical_count = 0
        if not representative.empty:
            for feature in context.features:
                left = factual.iloc[0][feature]
                right = representative.iloc[0][feature]
                if _value_changed(left, right):
                    changed_features.append(feature)
                    try:
                        feature_deltas[feature] = float(right) - float(left)
                    except Exception:
                        feature_deltas[feature] = float("nan")
                    if feature in context.catf:
                        changed_categorical_count += 1
                    else:
                        changed_numeric_count += 1

        summaries.append(
            {
                "dataset": context.dataset,
                "fold_name": fold_name,
                "method": method_name,
                "variant_name": variant.variant_name,
                "variant_family": variant.variant_family,
                "uf_mode": variant.uf_mode,
                "step_mode": variant.step_mode,
                "f2change_mode": variant.f2change_mode,
                "flip_filter": int(variant.flip_filter),
                "query_id": f"{fold_name}:{query_pos}",
                "query_pos": int(query_pos),
                "status": status,
                "generated_candidate_count": generated_count,
                "flip_candidate_count": flip_count,
                "selected_candidate_count": selected_count,
                "selected_prediction": selected_prediction,
                "representative_source": representative_source or "",
                "changed_features": changed_features,
                "feature_deltas": feature_deltas,
                "changed_feature_count": int(len(changed_features)),
                "changed_numeric_count": int(changed_numeric_count),
                "changed_categorical_count": int(changed_categorical_count),
                "selected_candidate": selected.iloc[:1].to_dict(orient="records")[0] if selected_count > 0 else {},
                "representative_candidate": representative.iloc[:1].to_dict(orient="records")[0] if not representative.empty else {},
                "search_meta": _to_builtin(trace_row.get("search_meta", {})),
                "search_parameters": _to_builtin(trace_row.get("search_parameters", {})),
                "trace_source_path": str(trace_row.get("source_path", METHOD_TO_TRACE_NAME[method_name])),
            }
        )
    return summaries


def _dummy_frame(features: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(features))


def _movie_contprox_frame(frame: pd.DataFrame, context: DatasetContext) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame
    if context.dataset != "movie" or context.movie_distance_scaler is None:
        return frame
    if not all(feature in frame.columns for feature in context.features):
        return frame
    projected = frame.loc[:, context.features].copy()
    return apply_distance_scaler(projected, context.movie_distance_scaler)


def evaluate_ufce_only(
    *,
    context: DatasetContext,
    variant: VariantSpec,
    onecfs: pd.DataFrame,
    onetest: pd.DataFrame,
    twocfs: pd.DataFrame,
    twotest: pd.DataFrame,
    threecfs: pd.DataFrame,
    threetest: pd.DataFrame,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    dummy = _dummy_frame(context.features)

    cat_means, _cat_stds = Catproximity(
        onecfs,
        onetest,
        twocfs,
        twotest,
        threecfs,
        threetest,
        dummy,
        dummy,
        dummy,
        dummy,
        context.Xtest,
        context.catf,
    )
    if len(context.catf) == 0:
        cat_means = [np.nan] * 6

    cont_inputs = {
        "onecfs": _movie_contprox_frame(onecfs, context),
        "onetest": _movie_contprox_frame(onetest, context),
        "twocfs": _movie_contprox_frame(twocfs, context),
        "twotest": _movie_contprox_frame(twotest, context),
        "threecfs": _movie_contprox_frame(threecfs, context),
        "threetest": _movie_contprox_frame(threetest, context),
        "dummy": _movie_contprox_frame(dummy, context),
        "xtest": _movie_contprox_frame(context.Xtest, context),
    }
    cont_means, _cont_stds = Contproximity(
        cont_inputs["onecfs"],
        cont_inputs["onetest"],
        cont_inputs["twocfs"],
        cont_inputs["twotest"],
        cont_inputs["threecfs"],
        cont_inputs["threetest"],
        cont_inputs["dummy"],
        cont_inputs["dummy"],
        cont_inputs["dummy"],
        cont_inputs["dummy"],
        cont_inputs["xtest"],
        context.numf,
    )
    spar_means, _spar_stds = Sparsity(
        onecfs,
        onetest,
        twocfs,
        twotest,
        threecfs,
        threetest,
        dummy,
        dummy,
        dummy,
        dummy,
        context.Xtest,
        context.numf,
    )
    act_means, _act_stds, act_details = Actionability(
        onecfs,
        onetest,
        twocfs,
        twotest,
        threecfs,
        threetest,
        dummy,
        dummy,
        dummy,
        dummy,
        context.Xtest,
        context.features,
        variant.f2change,
        variant.uf,
        return_details=True,
    )
    plaus_means, _plaus_stds, plaus_details = Plausibility(
        onecfs,
        onetest,
        twocfs,
        twotest,
        threecfs,
        threetest,
        dummy,
        dummy,
        dummy,
        dummy,
        context.Xtest,
        context.Xtrain,
        return_details=True,
    )
    feas_means, _feas_stds, feas_details = Feasibility(
        onecfs,
        onetest,
        twocfs,
        twotest,
        threecfs,
        threetest,
        dummy,
        dummy,
        dummy,
        dummy,
        context.Xtest,
        context.Xtrain,
        context.features,
        variant.f2change,
        context.lr,
        context.desired_outcome,
        variant.uf,
        return_details=True,
    )

    per_method = {
        "UFCE1": {
            "prox_jac": float(cat_means[0]),
            "prox_euc": float(cont_means[0]),
            "sparsity": float(spar_means[0]),
            "actionability": float(act_means[0]),
            "plausibility": float(plaus_means[0]),
            "feasibility": float(feas_means[0]),
        },
        "UFCE2": {
            "prox_jac": float(cat_means[1]),
            "prox_euc": float(cont_means[1]),
            "sparsity": float(spar_means[1]),
            "actionability": float(act_means[1]),
            "plausibility": float(plaus_means[1]),
            "feasibility": float(feas_means[1]),
        },
        "UFCE3": {
            "prox_jac": float(cat_means[2]),
            "prox_euc": float(cont_means[2]),
            "sparsity": float(spar_means[2]),
            "actionability": float(act_means[2]),
            "plausibility": float(plaus_means[2]),
            "feasibility": float(feas_means[2]),
        },
    }
    details = {
        "actionability": _to_builtin(act_details),
        "plausibility": _to_builtin(plaus_details),
        "feasibility": _to_builtin(feas_details),
    }
    return per_method, details


def run_variant_fold(
    context: DatasetContext,
    variant: VariantSpec,
    fold_path: str | Path,
    *,
    no_cf: int,
) -> Dict[str, Any]:
    fold_path = Path(fold_path)
    init_ufce_global(
        radius=int(context.tuned_cfg["radius"]),
        n_neighbors=int(context.tuned_cfg["n_neighbors"]),
        contprox_metric=str(context.contprox_metric),
        min_act=int(context.tuned_cfg["min_act"]),
        min_feas=int(context.tuned_cfg["min_feas"]),
        atol=1e-5,
    )
    fold_df = pd.read_csv(fold_path)
    testset_ufce, data_lab1_ufce, testset_dist, data_lab1_dist, distance_scaler = _prepare_fold_search_frames(context, fold_df)

    onecfs, t_ufce1, idx1, stats1, trace1, native1 = _call_with_trace_or_fallback(
        "UFCE1",
        sfexp,
        context.X,
        data_lab1_ufce,
        testset_ufce[:],
        variant.uf,
        variant.step,
        variant.f2change,
        context.numf,
        context.catf,
        context.lr,
        context.desired_outcome,
        int(no_cf),
        context.features,
        order=context.features,
        flip_filter_enabled=bool(variant.flip_filter),
        distance_data_lab1=data_lab1_dist,
        distance_X_test=testset_dist,
        distance_scaler=distance_scaler,
        debug_ctx=None,
    )
    twocfs, t_ufce2, idx2, stats2, trace2, native2 = _call_with_trace_or_fallback(
        "UFCE2",
        dfexp,
        context.X,
        data_lab1_ufce,
        testset_ufce[:],
        variant.uf,
        context.mi_pairs_top5,
        context.numf,
        context.catf,
        variant.f2change,
        context.protectf,
        context.lr,
        context.desired_outcome,
        int(no_cf),
        context.features,
        order=context.features,
        flip_filter_enabled=bool(variant.flip_filter),
        distance_data_lab1=data_lab1_dist,
        distance_X_test=testset_dist,
        distance_scaler=distance_scaler,
    )
    threecfs, t_ufce3, idx3, stats3, trace3, native3 = _call_with_trace_or_fallback(
        "UFCE3",
        tfexp,
        context.X,
        data_lab1_ufce,
        testset_ufce[:],
        variant.uf,
        context.mi_pairs_top5,
        context.numf,
        context.catf,
        variant.f2change,
        context.protectf,
        context.lr,
        context.desired_outcome,
        int(no_cf),
        context.features,
        order=context.features,
        flip_filter_enabled=bool(variant.flip_filter),
        distance_data_lab1=data_lab1_dist,
        distance_X_test=testset_dist,
        distance_scaler=distance_scaler,
    )

    if context.ufce_mad_scaler is not None and context.dataset != "movie":
        for df in (onecfs, twocfs, threecfs):
            if isinstance(df, pd.DataFrame) and not df.empty:
                mad_inverse_inplace(df, context.ufce_mad_scaler, context.numf)
        _inverse_trace_rows_inplace(trace1, context.ufce_mad_scaler, context.numf)
        _inverse_trace_rows_inplace(trace2, context.ufce_mad_scaler, context.numf)
        _inverse_trace_rows_inplace(trace3, context.ufce_mad_scaler, context.numf)

    onetest = fold_df.loc[idx1].reset_index(drop=True)
    twotest = fold_df.loc[idx2].reset_index(drop=True)
    threetest = fold_df.loc[idx3].reset_index(drop=True)

    per_method_metrics, detail_payloads = evaluate_ufce_only(
        context=context,
        variant=variant,
        onecfs=onecfs,
        onetest=onetest,
        twocfs=twocfs,
        twotest=twotest,
        threecfs=threecfs,
        threetest=threetest,
    )

    summarized_trace_rows: List[dict] = []
    summarized_trace_rows.extend(
        _summarize_trace_rows(
            context=context,
            variant=variant,
            fold_name=fold_path.name,
            fold_df=fold_df,
            method_name="UFCE1",
            trace_rows=trace1,
        )
    )
    summarized_trace_rows.extend(
        _summarize_trace_rows(
            context=context,
            variant=variant,
            fold_name=fold_path.name,
            fold_df=fold_df,
            method_name="UFCE2",
            trace_rows=trace2,
        )
    )
    summarized_trace_rows.extend(
        _summarize_trace_rows(
            context=context,
            variant=variant,
            fold_name=fold_path.name,
            fold_df=fold_df,
            method_name="UFCE3",
            trace_rows=trace3,
        )
    )

    trace_df = pd.DataFrame(summarized_trace_rows)
    fold_rows: List[dict] = []
    method_stats_map = {
        "UFCE1": stats1,
        "UFCE2": stats2,
        "UFCE3": stats3,
    }
    native_trace_map = {
        "UFCE1": native1,
        "UFCE2": native2,
        "UFCE3": native3,
    }
    runtime_map = {
        "UFCE1": float(t_ufce1),
        "UFCE2": float(t_ufce2),
        "UFCE3": float(t_ufce3),
    }

    for method_name in METHODS:
        method_df = trace_df[trace_df["method"] == method_name].copy()
        n_queries = int(len(method_df))
        valid_count = int((method_df["status"] == "valid").sum())
        invalid_count = int((method_df["status"] == "invalid").sum())
        empty_count = int((method_df["status"] == "empty").sum())
        fold_rows.append(
            {
                "dataset": context.dataset,
                "fold_name": fold_path.name,
                "method": method_name,
                "variant_name": variant.variant_name,
                "variant_family": variant.variant_family,
                "uf_mode": variant.uf_mode,
                "step_mode": variant.step_mode,
                "f2change_mode": variant.f2change_mode,
                "flip_filter": int(variant.flip_filter),
                "n_queries": n_queries,
                "valid_count": valid_count,
                "invalid_count": invalid_count,
                "empty_count": empty_count,
                "valid_rate": _safe_rate(valid_count, n_queries),
                "invalid_rate": _safe_rate(invalid_count, n_queries),
                "empty_rate": _safe_rate(empty_count, n_queries),
                "avg_changed_features": float(method_df["changed_feature_count"].mean()) if n_queries else 0.0,
                "avg_numeric_changes": float(method_df["changed_numeric_count"].mean()) if n_queries else 0.0,
                "avg_categorical_changes": float(method_df["changed_categorical_count"].mean()) if n_queries else 0.0,
                "prox_jac": per_method_metrics[method_name]["prox_jac"],
                "prox_euc": per_method_metrics[method_name]["prox_euc"],
                "sparsity": per_method_metrics[method_name]["sparsity"],
                "actionability": per_method_metrics[method_name]["actionability"],
                "plausibility": per_method_metrics[method_name]["plausibility"],
                "feasibility": per_method_metrics[method_name]["feasibility"],
                "runtime_sec": runtime_map[method_name],
                "trace_mode_native": bool(native_trace_map[method_name]),
                "method_stats": _json_dump(method_stats_map[method_name]),
            }
        )

    return {
        "fold_rows": fold_rows,
        "trace_rows": summarized_trace_rows,
        "details": {
            "dataset": context.dataset,
            "fold_name": fold_path.name,
            "variant_name": variant.variant_name,
            "variant_family": variant.variant_family,
            "flip_filter": int(variant.flip_filter),
            "metrics": _to_builtin(per_method_metrics),
            "method_stats": _to_builtin(method_stats_map),
            "detail_payloads": detail_payloads,
            "native_trace": native_trace_map,
        },
    }


def run_variant_dataset(
    context: DatasetContext,
    variant: VariantSpec,
    *,
    fold_dir: str | Path,
    max_folds: Optional[int],
    no_cf: int,
) -> Dict[str, Any]:
    fold_dir = _resolve_output_path(str(fold_dir))
    dataset_fold_dir = fold_dir / context.dataset / "totest"
    fold_paths = sorted(glob.glob(str(dataset_fold_dir / "*.csv")))
    if max_folds is not None and int(max_folds) > 0:
        fold_paths = fold_paths[: int(max_folds)]
    if not fold_paths:
        raise FileNotFoundError(f"No folds found for dataset={context.dataset} under {dataset_fold_dir}")

    print(
        "[ABLATE][DATASET-RUN] "
        f"dataset={context.dataset} variant={variant.variant_name} family={variant.variant_family} "
        f"flip_filter={variant.flip_filter} folds={len(fold_paths)}",
        flush=True,
    )

    all_fold_rows: List[dict] = []
    all_trace_rows: List[dict] = []
    all_details: List[dict] = []
    for fold_index, fold_path in enumerate(fold_paths, start=1):
        print(
            "[ABLATE][FOLD] "
            f"dataset={context.dataset} variant={variant.variant_name} "
            f"fold={fold_index}/{len(fold_paths)} file={Path(fold_path).name}",
            flush=True,
        )
        fold_result = run_variant_fold(
            context,
            variant,
            fold_path,
            no_cf=int(no_cf),
        )
        all_fold_rows.extend(fold_result["fold_rows"])
        all_trace_rows.extend(fold_result["trace_rows"])
        all_details.append(fold_result["details"])
    print(
        "[ABLATE][DATASET-RUN-DONE] "
        f"dataset={context.dataset} variant={variant.variant_name} "
        f"fold_rows={len(all_fold_rows)} trace_rows={len(all_trace_rows)}",
        flush=True,
    )
    return {
        "fold_rows": all_fold_rows,
        "trace_rows": all_trace_rows,
        "details": all_details,
    }


def _metric_delta_columns(prefix: str) -> List[str]:
    return [
        f"delta_prox_jac_vs_{prefix}",
        f"delta_prox_euc_vs_{prefix}",
        f"delta_sparsity_vs_{prefix}",
        f"delta_actionability_vs_{prefix}",
        f"delta_plausibility_vs_{prefix}",
        f"delta_feasibility_vs_{prefix}",
    ]


def _find_baseline_row(
    summary_df: pd.DataFrame,
    *,
    dataset: str,
    method: str,
    variant_name: str,
    flip_filter: int,
) -> Optional[pd.Series]:
    same_mask = (
        (summary_df["dataset"] == dataset)
        & (summary_df["method"] == method)
        & (summary_df["variant_name"] == variant_name)
        & (summary_df["flip_filter"] == int(flip_filter))
    )
    if same_mask.any():
        return summary_df.loc[same_mask].iloc[0]
    base_mask = (
        (summary_df["dataset"] == dataset)
        & (summary_df["method"] == method)
        & (summary_df["variant_name"] == variant_name)
        & (summary_df["flip_filter"] == 0)
    )
    if base_mask.any():
        return summary_df.loc[base_mask].iloc[0]
    return None


def _js_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if left_arr.sum() <= 0.0 and right_arr.sum() <= 0.0:
        return 0.0
    if left_arr.sum() > 0.0:
        left_arr = left_arr / left_arr.sum()
    if right_arr.sum() > 0.0:
        right_arr = right_arr / right_arr.sum()
    midpoint = 0.5 * (left_arr + right_arr)

    def _kl_div(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0.0) & (b > 0.0)
        if not np.any(mask):
            return 0.0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl_div(left_arr, midpoint) + 0.5 * _kl_div(right_arr, midpoint)


def build_feature_change_profile(
    trace_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    feature_catalog: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    group_keys = [
        "dataset",
        "method",
        "variant_name",
        "variant_family",
        "uf_mode",
        "step_mode",
        "f2change_mode",
        "flip_filter",
    ]
    exploded_rows: List[dict] = []
    for row in trace_df.to_dict(orient="records"):
        feature_deltas = dict(row.get("feature_deltas", {}))
        for feature, delta in feature_deltas.items():
            exploded_rows.append(
                {
                    **{key: row[key] for key in group_keys},
                    "status": row["status"],
                    "feature": feature,
                    "abs_delta": abs(float(delta)) if pd.notna(delta) else float("nan"),
                }
            )

    exploded_df = pd.DataFrame(exploded_rows)
    summary_index = {
        tuple(row[key] for key in group_keys): row
        for row in summary_df.to_dict(orient="records")
    }

    profile_rows: List[dict] = []
    for key_tuple, summary_row in summary_index.items():
        dataset = summary_row["dataset"]
        feature_types = feature_catalog[dataset]
        subset = exploded_df
        if not exploded_df.empty:
            mask = np.ones(len(exploded_df), dtype=bool)
            for key, value in zip(group_keys, key_tuple):
                mask &= exploded_df[key].astype(object).values == value
            subset = exploded_df.loc[mask].copy()
        for feature, feature_type in feature_types.items():
            feature_subset = subset[subset["feature"] == feature].copy() if not subset.empty else pd.DataFrame()
            changed_count = int(len(feature_subset))
            valid_changed = int((feature_subset["status"] == "valid").sum()) if not feature_subset.empty else 0
            profile_rows.append(
                {
                    **{key: summary_row[key] for key in group_keys},
                    "feature": feature,
                    "feature_type": feature_type,
                    "changed_count": changed_count,
                    "changed_rate_valid": _safe_rate(valid_changed, summary_row["valid_count"]),
                    "changed_rate_non_empty": _safe_rate(changed_count, summary_row["valid_count"] + summary_row["invalid_count"]),
                    "avg_abs_delta": float(feature_subset["abs_delta"].mean()) if changed_count else 0.0,
                }
            )
    return pd.DataFrame(profile_rows)


def build_summary_frames(
    fold_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    feature_catalog: Dict[str, Dict[str, str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    group_keys = [
        "dataset",
        "method",
        "variant_name",
        "variant_family",
        "uf_mode",
        "step_mode",
        "f2change_mode",
        "flip_filter",
    ]

    if trace_df.empty:
        query_summary = pd.DataFrame(columns=group_keys + [
            "n_queries",
            "valid_count",
            "invalid_count",
            "empty_count",
            "valid_rate",
            "invalid_rate",
            "empty_rate",
            "avg_changed_features",
            "avg_numeric_changes",
            "avg_categorical_changes",
        ])
    else:
        query_summary = (
            trace_df.groupby(group_keys, dropna=False)
            .agg(
                n_queries=("query_id", "count"),
                valid_count=("status", lambda series: int((series == "valid").sum())),
                invalid_count=("status", lambda series: int((series == "invalid").sum())),
                empty_count=("status", lambda series: int((series == "empty").sum())),
                avg_changed_features=("changed_feature_count", "mean"),
                avg_numeric_changes=("changed_numeric_count", "mean"),
                avg_categorical_changes=("changed_categorical_count", "mean"),
            )
            .reset_index()
        )
        query_summary["valid_rate"] = query_summary.apply(lambda row: _safe_rate(row["valid_count"], row["n_queries"]), axis=1)
        query_summary["invalid_rate"] = query_summary.apply(lambda row: _safe_rate(row["invalid_count"], row["n_queries"]), axis=1)
        query_summary["empty_rate"] = query_summary.apply(lambda row: _safe_rate(row["empty_count"], row["n_queries"]), axis=1)

    metric_cols = [
        "prox_jac",
        "prox_euc",
        "sparsity",
        "actionability",
        "plausibility",
        "feasibility",
        "runtime_sec",
    ]
    if fold_df.empty:
        metric_summary = pd.DataFrame(columns=group_keys + metric_cols)
    else:
        metric_summary = (
            fold_df.groupby(group_keys, dropna=False)[metric_cols]
            .mean()
            .reset_index()
        )

    summary_df = metric_summary.merge(query_summary, on=group_keys, how="outer")
    if summary_df.empty:
        feature_profile_df = pd.DataFrame()
        return summary_df, feature_profile_df, {"feature_path": {}, "normalization": {}, "baselines": {}}

    fill_zero_cols = [
        "n_queries",
        "valid_count",
        "invalid_count",
        "empty_count",
        "valid_rate",
        "invalid_rate",
        "empty_rate",
        "avg_changed_features",
        "avg_numeric_changes",
        "avg_categorical_changes",
        "runtime_sec",
    ]
    for col in fill_zero_cols:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].fillna(0.0)

    feature_profile_df = build_feature_change_profile(trace_df, summary_df, feature_catalog)

    baseline_debug: Dict[str, Any] = {}
    for prefix, baseline_name in [("public_baseline", ANCHOR_PUBLIC_NAME), ("author_baseline", ANCHOR_AUTHOR_NAME)]:
        summary_df[f"{prefix}_available"] = 0
        for metric_col, delta_col in zip(
            METRICS,
            _metric_delta_columns(prefix),
        ):
            summary_df[delta_col] = np.nan
        summary_df[f"delta_valid_rate_vs_{prefix}"] = np.nan
        summary_df[f"delta_invalid_rate_vs_{prefix}"] = np.nan
        summary_df[f"delta_empty_rate_vs_{prefix}"] = np.nan
        for idx, row in summary_df.iterrows():
            baseline_row = _find_baseline_row(
                summary_df,
                dataset=str(row["dataset"]),
                method=str(row["method"]),
                variant_name=baseline_name,
                flip_filter=int(row["flip_filter"]),
            )
            if baseline_row is None:
                continue
            summary_df.at[idx, f"{prefix}_available"] = 1
            summary_df.at[idx, f"delta_valid_rate_vs_{prefix}"] = float(row["valid_rate"]) - float(baseline_row["valid_rate"])
            summary_df.at[idx, f"delta_invalid_rate_vs_{prefix}"] = float(row["invalid_rate"]) - float(baseline_row["invalid_rate"])
            summary_df.at[idx, f"delta_empty_rate_vs_{prefix}"] = float(row["empty_rate"]) - float(baseline_row["empty_rate"])
            for metric_col, delta_col in zip(METRICS, _metric_delta_columns(prefix)):
                summary_df.at[idx, delta_col] = float(row[metric_col]) - float(baseline_row[metric_col])
            baseline_debug.setdefault(str(row["dataset"]), {}).setdefault(str(row["method"]), {})[str(row["variant_name"])] = {
                "flip_filter": int(row["flip_filter"]),
                "baseline_name": baseline_name,
                "resolved_flip_filter": int(baseline_row["flip_filter"]),
            }

    feature_path_debug: Dict[str, Any] = {}
    summary_df["feature_path_jsd_vs_public_baseline"] = 0.0
    summary_df["feature_path_score_vs_public_baseline"] = 0.0
    for idx, row in summary_df.iterrows():
        dataset = str(row["dataset"])
        method = str(row["method"])
        variant_name = str(row["variant_name"])
        flip_filter = int(row["flip_filter"])
        baseline_row = _find_baseline_row(
            summary_df,
            dataset=dataset,
            method=method,
            variant_name=ANCHOR_PUBLIC_NAME,
            flip_filter=flip_filter,
        )
        if baseline_row is None:
            continue
        current_profile = feature_profile_df[
            (feature_profile_df["dataset"] == dataset)
            & (feature_profile_df["method"] == method)
            & (feature_profile_df["variant_name"] == variant_name)
            & (feature_profile_df["flip_filter"] == flip_filter)
        ].sort_values("feature")
        baseline_profile = feature_profile_df[
            (feature_profile_df["dataset"] == dataset)
            & (feature_profile_df["method"] == method)
            & (feature_profile_df["variant_name"] == str(baseline_row["variant_name"]))
            & (feature_profile_df["flip_filter"] == int(baseline_row["flip_filter"]))
        ].sort_values("feature")
        current_dist = current_profile["changed_count"].to_numpy(dtype=float) if not current_profile.empty else np.zeros(0, dtype=float)
        baseline_dist = baseline_profile["changed_count"].to_numpy(dtype=float) if not baseline_profile.empty else np.zeros(0, dtype=float)
        jsd = _js_divergence(current_dist, baseline_dist) if len(current_dist) == len(baseline_dist) else 0.0
        summary_df.at[idx, "feature_path_jsd_vs_public_baseline"] = jsd
        feature_path_debug.setdefault(dataset, {}).setdefault(method, {})[variant_name] = {
            "flip_filter": flip_filter,
            "jsd_vs_public_baseline": jsd,
        }

    normalization_debug: Dict[str, Any] = {}
    for prefix in ["public_baseline", "author_baseline"]:
        score_col = f"delta_metric_score_vs_{prefix}"
        summary_df[score_col] = 0.0
        delta_cols = _metric_delta_columns(prefix)
        for dataset, dataset_df in summary_df.groupby("dataset"):
            maxima = {}
            for delta_col in delta_cols:
                values = dataset_df[delta_col].abs().replace([np.inf, -np.inf], np.nan).dropna()
                maxima[delta_col] = float(values.max()) if not values.empty else 0.0
            normalization_debug.setdefault(str(dataset), {})[prefix] = maxima
            for idx in dataset_df.index:
                normalized_values: List[float] = []
                for delta_col in delta_cols:
                    denom = maxima[delta_col]
                    value = float(summary_df.at[idx, delta_col]) if pd.notna(summary_df.at[idx, delta_col]) else 0.0
                    normalized_values.append(abs(value) / denom if denom > 0 else 0.0)
                summary_df.at[idx, score_col] = float(np.mean(normalized_values)) if normalized_values else 0.0

    for dataset, dataset_df in summary_df.groupby("dataset"):
        jsd_max = float(dataset_df["feature_path_jsd_vs_public_baseline"].abs().max()) if not dataset_df.empty else 0.0
        avg_change_max = float(dataset_df["delta_valid_rate_vs_public_baseline"].abs().max()) if not dataset_df.empty else 0.0
        normalization_debug.setdefault(str(dataset), {})["feature_path"] = {
            "jsd_max": jsd_max,
            "avg_changed_features_delta_max": float(dataset_df["avg_changed_features"].abs().max()) if not dataset_df.empty else 0.0,
            "avg_numeric_changes_delta_max": float(dataset_df["avg_numeric_changes"].abs().max()) if not dataset_df.empty else 0.0,
            "avg_categorical_changes_delta_max": float(dataset_df["avg_categorical_changes"].abs().max()) if not dataset_df.empty else 0.0,
            "delta_valid_rate_max": avg_change_max,
        }
        for idx in dataset_df.index:
            baseline_row = _find_baseline_row(
                summary_df,
                dataset=str(summary_df.at[idx, "dataset"]),
                method=str(summary_df.at[idx, "method"]),
                variant_name=ANCHOR_PUBLIC_NAME,
                flip_filter=int(summary_df.at[idx, "flip_filter"]),
            )
            if baseline_row is None:
                continue
            denom_cfg = normalization_debug[str(dataset)]["feature_path"]
            jsd_component = abs(float(summary_df.at[idx, "feature_path_jsd_vs_public_baseline"])) / denom_cfg["jsd_max"] if denom_cfg["jsd_max"] > 0 else 0.0
            avg_component = abs(float(summary_df.at[idx, "avg_changed_features"]) - float(baseline_row["avg_changed_features"])) / denom_cfg["avg_changed_features_delta_max"] if denom_cfg["avg_changed_features_delta_max"] > 0 else 0.0
            numeric_component = abs(float(summary_df.at[idx, "avg_numeric_changes"]) - float(baseline_row["avg_numeric_changes"])) / denom_cfg["avg_numeric_changes_delta_max"] if denom_cfg["avg_numeric_changes_delta_max"] > 0 else 0.0
            categorical_component = abs(float(summary_df.at[idx, "avg_categorical_changes"]) - float(baseline_row["avg_categorical_changes"])) / denom_cfg["avg_categorical_changes_delta_max"] if denom_cfg["avg_categorical_changes_delta_max"] > 0 else 0.0
            summary_df.at[idx, "feature_path_score_vs_public_baseline"] = float(np.mean([jsd_component, avg_component, numeric_component, categorical_component]))

    summary_df = summary_df.sort_values(["dataset", "variant_family", "variant_name", "method", "flip_filter"]).reset_index(drop=True)
    sidecar = {
        "feature_path": feature_path_debug,
        "normalization": normalization_debug,
        "baselines": baseline_debug,
    }
    return summary_df, feature_profile_df, sidecar


def build_ranked_findings(summary_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[dict] = []
    debug: Dict[str, Any] = {}
    non_anchor = summary_df[
        (summary_df["flip_filter"] == 0)
        & (summary_df["variant_family"] != "anchor")
    ].copy()
    if non_anchor.empty:
        return pd.DataFrame(columns=[
            "dataset",
            "top_valid_family",
            "top_valid_variant",
            "top_valid_score",
            "top_empty_family",
            "top_empty_variant",
            "top_empty_score",
            "top_feature_path_family",
            "top_feature_path_variant",
            "top_feature_path_score",
            "top_metric_family",
            "top_metric_variant",
            "top_metric_score",
        ]), debug

    variant_scores = (
        non_anchor.groupby(["dataset", "variant_family", "variant_name"], dropna=False)
        .agg(
            valid_success_score=("delta_valid_rate_vs_public_baseline", lambda series: float(np.mean(np.abs(series)))),
            empty_rate_score=("delta_empty_rate_vs_public_baseline", lambda series: float(np.mean(np.abs(series)))),
            feature_path_score=("feature_path_score_vs_public_baseline", "mean"),
            metric_impact_score=("delta_metric_score_vs_public_baseline", "mean"),
        )
        .reset_index()
    )

    for dataset, dataset_df in variant_scores.groupby("dataset"):
        debug[str(dataset)] = dataset_df.to_dict(orient="records")
        out_row = {"dataset": dataset}
        for score_col, family_col, variant_col, score_out in [
            ("valid_success_score", "top_valid_family", "top_valid_variant", "top_valid_score"),
            ("empty_rate_score", "top_empty_family", "top_empty_variant", "top_empty_score"),
            ("feature_path_score", "top_feature_path_family", "top_feature_path_variant", "top_feature_path_score"),
            ("metric_impact_score", "top_metric_family", "top_metric_variant", "top_metric_score"),
        ]:
            ranked = dataset_df.sort_values(
                [score_col, "variant_family", "variant_name"],
                ascending=[False, True, True],
            ).reset_index(drop=True)
            top = ranked.iloc[0]
            out_row[family_col] = str(top["variant_family"])
            out_row[variant_col] = str(top["variant_name"])
            out_row[score_out] = float(top[score_col])
        rows.append(out_row)
    return pd.DataFrame(rows), debug


def select_strict_variant_names(summary_df: pd.DataFrame) -> Dict[str, List[str]]:
    strict_selection: Dict[str, List[str]] = {}
    base_df = summary_df[summary_df["flip_filter"] == 0].copy()
    for dataset, dataset_df in base_df.groupby("dataset"):
        candidate_df = dataset_df[
            (dataset_df["variant_name"] != ANCHOR_PUBLIC_NAME)
            & (dataset_df["variant_name"] != ANCHOR_AUTHOR_NAME)
        ].copy()
        if candidate_df.empty:
            strict_selection[str(dataset)] = [ANCHOR_PUBLIC_NAME]
            continue
        scored = (
            candidate_df.groupby("variant_name", dropna=False)
            .agg(
                base_score=("delta_valid_rate_vs_public_baseline", lambda series: float(np.mean(np.abs(series)))),
                invalid_score=("delta_invalid_rate_vs_public_baseline", lambda series: float(np.mean(np.abs(series)))),
                empty_score=("delta_empty_rate_vs_public_baseline", lambda series: float(np.mean(np.abs(series)))),
                feature_tiebreak=("feature_path_score_vs_public_baseline", "mean"),
            )
            .reset_index()
        )
        scored["selection_score"] = scored[["base_score", "invalid_score", "empty_score"]].mean(axis=1)
        ranked = scored.sort_values(
            ["selection_score", "feature_tiebreak", "variant_name"],
            ascending=[False, False, True],
        )
        top_two = ranked["variant_name"].head(2).tolist()
        strict_selection[str(dataset)] = [ANCHOR_PUBLIC_NAME] + top_two
    return strict_selection


def _build_feature_catalog(contexts: Dict[str, DatasetContext]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for dataset, context in contexts.items():
        feature_types = {
            feature: ("categorical" if feature in context.catf else "numeric")
            for feature in context.features
        }
        out[dataset] = feature_types
    return out


def _collect_specs_for_dataset(
    context: DatasetContext,
    *,
    stages: Sequence[str],
    variant_filter: set[str],
) -> List[VariantSpec]:
    specs: List[VariantSpec] = []
    specs.extend(build_anchor_specs(context))
    if "single" in stages:
        specs.extend(build_single_factor_specs(context))
    if "joint" in stages:
        specs.extend(build_joint_specs(context))

    deduped: Dict[Tuple[str, int], VariantSpec] = {}
    for spec in specs:
        if not _variant_passes_filter(spec, variant_filter):
            continue
        deduped[(spec.variant_name, spec.flip_filter)] = spec
    return list(deduped.values())


def _append_strict_specs(
    base_specs: Dict[str, Dict[str, VariantSpec]],
    strict_selection: Dict[str, List[str]],
) -> List[Tuple[str, VariantSpec]]:
    strict_specs: List[Tuple[str, VariantSpec]] = []
    for dataset, variant_names in strict_selection.items():
        for variant_name in variant_names:
            if dataset not in base_specs or variant_name not in base_specs[dataset]:
                continue
            base_spec = base_specs[dataset][variant_name]
            strict_specs.append(
                (
                    dataset,
                    VariantSpec(
                    variant_name=base_spec.variant_name,
                    variant_family=base_spec.variant_family,
                    uf_mode=base_spec.uf_mode,
                    step_mode=base_spec.step_mode,
                    f2change_mode=base_spec.f2change_mode,
                    flip_filter=1,
                    uf=dict(base_spec.uf),
                    step=dict(base_spec.step),
                    f2change=list(base_spec.f2change),
                    ),
                )
            )
    return strict_specs


def _write_master_partial_checkpoint(
    *,
    out_dir: Path,
    fold_rows: Sequence[dict],
    trace_rows: Sequence[dict],
    feature_catalog: Dict[str, Dict[str, str]],
) -> None:
    fold_df = pd.DataFrame(fold_rows)
    trace_df = pd.DataFrame(trace_rows)
    summary_df, _feature_profile_df, _summary_sidecar = build_summary_frames(fold_df, trace_df, feature_catalog)
    partial_path = out_dir / "master_summary_partial.csv"
    summary_df.to_csv(partial_path, index=False)
    print(
        "[ABLATE][PARTIAL-WRITE] "
        f"master_summary_partial={partial_path} rows={len(summary_df)}",
        flush=True,
    )


def _write_dataset_checkpoint_artifacts(
    *,
    out_dir: Path,
    dataset: str,
    summary_df: pd.DataFrame,
    feature_profile_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    partial: bool,
) -> None:
    dataset_out = _ensure_dir(out_dir / dataset)
    suffix = "_partial" if partial else ""
    dataset_summary = summary_df[summary_df["dataset"] == dataset].copy() if "dataset" in summary_df.columns else pd.DataFrame()
    dataset_features = (
        feature_profile_df[feature_profile_df["dataset"] == dataset].copy()
        if "dataset" in feature_profile_df.columns
        else pd.DataFrame()
    )
    dataset_folds = fold_df[fold_df["dataset"] == dataset].copy() if "dataset" in fold_df.columns else pd.DataFrame()
    dataset_traces = trace_df[trace_df["dataset"] == dataset].copy() if "dataset" in trace_df.columns else pd.DataFrame()

    dataset_summary[dataset_summary["variant_family"].isin(["uf", "step", "f2change"])].to_csv(
        dataset_out / f"single_factor{suffix}.csv",
        index=False,
    )
    dataset_summary[dataset_summary["variant_family"].isin(["joint", "anchor"])].to_csv(
        dataset_out / f"joint_ablation{suffix}.csv",
        index=False,
    )
    dataset_features.to_csv(dataset_out / f"feature_change_profile{suffix}.csv", index=False)
    dataset_folds.to_csv(dataset_out / f"fold_breakdown{suffix}.csv", index=False)
    _write_jsonl(dataset_out / f"query_trace{suffix}.jsonl", dataset_traces.to_dict(orient="records"))
    print(
        "[ABLATE][DATASET-WRITE] "
        f"dataset={dataset} out_dir={dataset_out} partial={int(partial)} "
        f"summary_rows={len(dataset_summary)} feature_rows={len(dataset_features)} "
        f"fold_rows={len(dataset_folds)} trace_rows={len(dataset_traces)}",
        flush=True,
    )


def _write_skipped_variants_csv(out_dir: Path, skipped_variants: Sequence[dict]) -> Path:
    skipped_path = out_dir / "skipped_variants.csv"
    skipped_df = pd.DataFrame(list(skipped_variants), columns=SKIPPED_VARIANT_COLUMNS)
    skipped_df.to_csv(skipped_path, index=False)
    print(
        "[ABLATE][SKIP-WRITE] "
        f"path={skipped_path} rows={len(skipped_df)}",
        flush=True,
    )
    return skipped_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UFCE blind-spot ablation harness.")
    parser.add_argument("--datasets", default="bank,grad,wine,bupa,movie")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--out_dir", "--out-dir", dest="out_dir", default="outputs/uf_ablation")
    parser.add_argument("--max_folds", type=int, default=None)
    parser.add_argument("--stages", default="all")
    parser.add_argument("--stage", default=None)
    parser.add_argument("--variant_filter", default=None)
    parser.add_argument("--data_dir", default=os.path.join("ufce", "data"))
    parser.add_argument("--fold_dir", default=os.path.join("ufce", "data", "folds"))
    parser.add_argument("--no_cf", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--contprox_metric", default="euclidean")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    args = parse_args(argv)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    if args.dataset is not None:
        args.datasets = str(args.dataset)
    if args.stage is not None:
        args.stages = str(args.stage)

    datasets = _parse_dataset_list(args.datasets)
    stages = _parse_stage_list(args.stages)
    variant_filter = _parse_variant_filter(args.variant_filter)

    out_dir = _ensure_dir(_resolve_output_path(args.out_dir))
    print(
        "[ABLATE][START] "
        f"datasets={datasets} stages={stages} out_dir={out_dir} "
        f"max_folds={args.max_folds} no_cf={args.no_cf} seed={args.seed}",
        flush=True,
    )
    provenance_records = extract_provenance_records(args.data_dir, datasets)
    provenance_df = pd.DataFrame(provenance_records)
    provenance_path = out_dir / "provenance.csv"
    provenance_df.to_csv(provenance_path, index=False)
    print(
        "[ABLATE][PROVENANCE] "
        f"rows={len(provenance_df)} path={provenance_path}",
        flush=True,
    )

    master_summary_path = out_dir / "master_summary.csv"
    master_sidecar_path = out_dir / "master_summary_sidecar.json"
    ranked_findings_path = out_dir / "master_ranked_findings.csv"
    skipped_variants_path = out_dir / "skipped_variants.csv"

    execute_variants = any(stage in stages for stage in ["single", "joint", "strict"])
    if not execute_variants:
        print(
            "[ABLATE][PROVENANCE-ONLY] "
            f"master_summary={master_summary_path} ranked_findings={ranked_findings_path}",
            flush=True,
        )
        empty_summary = pd.DataFrame()
        empty_ranked = pd.DataFrame(columns=[
            "dataset",
            "top_valid_family",
            "top_valid_variant",
            "top_valid_score",
            "top_empty_family",
            "top_empty_variant",
            "top_empty_score",
            "top_feature_path_family",
            "top_feature_path_variant",
            "top_feature_path_score",
            "top_metric_family",
            "top_metric_variant",
            "top_metric_score",
        ])
        empty_summary.to_csv(master_summary_path, index=False)
        empty_ranked.to_csv(ranked_findings_path, index=False)
        _write_skipped_variants_csv(out_dir, [])
        _write_json(
            master_sidecar_path,
            {"datasets": {}, "summary": {}, "ranking": {}, "strict_selection": {}, "skipped_variants": []},
        )
        return {
            "provenance_path": str(provenance_path),
            "master_summary_path": str(master_summary_path),
            "master_sidecar_path": str(master_sidecar_path),
            "ranked_findings_path": str(ranked_findings_path),
            "skipped_variants_path": str(skipped_variants_path),
            "datasets": datasets,
        }

    provenance_map = {str(row["dataset"]): row for row in provenance_records}
    contexts = {
        dataset: build_dataset_context(
            dataset,
            args.data_dir,
            provenance_map,
            contprox_metric=str(args.contprox_metric),
        )
        for dataset in datasets
    }
    print(
        "[ABLATE][CONTEXTS-READY] "
        f"datasets={list(contexts.keys())}",
        flush=True,
    )
    feature_catalog = _build_feature_catalog(contexts)
    planned_specs_by_dataset = {
        dataset: _collect_specs_for_dataset(
            contexts[dataset],
            stages=stages,
            variant_filter=variant_filter,
        )
        for dataset in datasets
    }
    progress_tracker = ProgressTracker(
        total_units=sum(len(specs) for specs in planned_specs_by_dataset.values()),
        start_time=t0,
    )
    print(
        "[ABLATE][MASTER-PLAN] "
        f"base_total_variants={progress_tracker.total_units}",
        flush=True,
    )

    all_fold_rows: List[dict] = []
    all_trace_rows: List[dict] = []
    skipped_variants: List[dict] = []
    _write_skipped_variants_csv(out_dir, skipped_variants)
    sidecar_details: Dict[str, Any] = {"datasets": {}, "skipped_variants": skipped_variants}
    base_specs_by_dataset: Dict[str, Dict[str, VariantSpec]] = {}

    for dataset in datasets:
        context = contexts[dataset]
        specs = planned_specs_by_dataset[dataset]
        print(
            "[ABLATE][DATASET] "
            f"dataset={dataset} variants={len(specs)} "
            f"families={sorted(set(spec.variant_family for spec in specs))}",
            flush=True,
        )
        base_specs_by_dataset[dataset] = {}
        dataset_sidecar = sidecar_details["datasets"].setdefault(dataset, {})
        dataset_sidecar["planned_variants"] = [spec.variant_name for spec in specs]
        dataset_sidecar["executed_variants"] = []
        dataset_sidecar["skipped_variants"] = []
        dataset_sidecar["fold_details"] = {}
        for variant_index, spec in enumerate(specs, start=1):
            t_variant = time.perf_counter()
            print(
                "[ABLATE][VARIANT-START] "
                f"dataset={dataset} index={variant_index}/{len(specs)} "
                f"name={spec.variant_name} uf={spec.uf_mode} step={spec.step_mode} "
                f"f2change={spec.f2change_mode} flip_filter={spec.flip_filter}",
                flush=True,
            )
            missing_step_keys = validate_variant_step_coverage(context, spec)
            if missing_step_keys:
                skip_record = _build_skip_record(context, spec, missing_step_keys)
                skipped_variants.append(skip_record)
                dataset_sidecar["skipped_variants"].append(skip_record)
                sidecar_details["skipped_variants"] = skipped_variants
                print(
                    "[ABLATE][VARIANT-SKIP] "
                    f"dataset={dataset} index={variant_index}/{len(specs)} "
                    f"name={spec.variant_name} reason={skip_record['skip_reason']}",
                    flush=True,
                )
                _write_skipped_variants_csv(out_dir, skipped_variants)
                last_seconds = time.perf_counter() - t_variant
                progress_tracker.record_completion(
                    dataset=dataset,
                    variant_name=spec.variant_name,
                    stage="base_skip",
                    last_seconds=last_seconds,
                )
                _write_master_partial_checkpoint(
                    out_dir=out_dir,
                    fold_rows=all_fold_rows,
                    trace_rows=all_trace_rows,
                    feature_catalog=feature_catalog,
                )
                continue
            run_result = run_variant_dataset(
                context,
                spec,
                fold_dir=args.fold_dir,
                max_folds=args.max_folds,
                no_cf=args.no_cf,
            )
            all_fold_rows.extend(run_result["fold_rows"])
            all_trace_rows.extend(run_result["trace_rows"])
            base_specs_by_dataset[dataset][spec.variant_name] = spec
            dataset_sidecar["executed_variants"].append(spec.variant_name)
            dataset_sidecar["fold_details"][spec.variant_name] = run_result["details"]
            last_seconds = time.perf_counter() - t_variant
            print(
                "[ABLATE][VARIANT-DONE] "
                f"dataset={dataset} index={variant_index}/{len(specs)} "
                f"name={spec.variant_name} seconds={last_seconds:.2f} "
                f"fold_rows={len(run_result['fold_rows'])} trace_rows={len(run_result['trace_rows'])}",
                flush=True,
            )
            progress_tracker.record_completion(
                dataset=dataset,
                variant_name=spec.variant_name,
                stage="base",
                last_seconds=last_seconds,
            )
            _write_master_partial_checkpoint(
                out_dir=out_dir,
                fold_rows=all_fold_rows,
                trace_rows=all_trace_rows,
                feature_catalog=feature_catalog,
            )

        dataset_fold_df = pd.DataFrame(all_fold_rows)
        dataset_trace_df = pd.DataFrame(all_trace_rows)
        dataset_summary_df, dataset_feature_profile_df, _dataset_summary_sidecar = build_summary_frames(
            dataset_fold_df,
            dataset_trace_df,
            feature_catalog,
        )
        _write_dataset_checkpoint_artifacts(
            out_dir=out_dir,
            dataset=dataset,
            summary_df=dataset_summary_df,
            feature_profile_df=dataset_feature_profile_df,
            fold_df=dataset_fold_df,
            trace_df=dataset_trace_df,
            partial=True,
        )

    fold_df = pd.DataFrame(all_fold_rows)
    trace_df = pd.DataFrame(all_trace_rows)
    summary_df, feature_profile_df, summary_sidecar = build_summary_frames(fold_df, trace_df, feature_catalog)
    print(
        "[ABLATE][SUMMARY-BUILD] "
        f"summary_rows={len(summary_df)} feature_profile_rows={len(feature_profile_df)} "
        f"trace_rows={len(trace_df)}",
        flush=True,
    )

    strict_selection: Dict[str, List[str]] = {}
    if "strict" in stages:
        strict_selection = select_strict_variant_names(summary_df)
        print(
            "[ABLATE][STRICT-SELECT] "
            f"selection={strict_selection}",
            flush=True,
        )
        for dataset in datasets:
            sidecar_details["datasets"].setdefault(dataset, {})["strict_selection"] = strict_selection.get(dataset, [])
        strict_specs = _append_strict_specs(base_specs_by_dataset, strict_selection)
        progress_tracker.extend_total(len(strict_specs), reason="strict_reruns")
        for strict_index, (dataset, strict_spec) in enumerate(strict_specs, start=1):
            context = contexts[dataset]
            t_strict = time.perf_counter()
            print(
                "[ABLATE][STRICT-START] "
                f"index={strict_index}/{len(strict_specs)} dataset={dataset} "
                f"name={strict_spec.variant_name}",
                flush=True,
            )
            missing_step_keys = validate_variant_step_coverage(context, strict_spec)
            if missing_step_keys:
                skip_record = _build_skip_record(context, strict_spec, missing_step_keys)
                skipped_variants.append(skip_record)
                sidecar_details["datasets"].setdefault(dataset, {}).setdefault("skipped_variants", []).append(skip_record)
                sidecar_details["skipped_variants"] = skipped_variants
                print(
                    "[ABLATE][VARIANT-SKIP] "
                    f"dataset={dataset} index={strict_index}/{len(strict_specs)} "
                    f"name={strict_spec.variant_name} reason={skip_record['skip_reason']}",
                    flush=True,
                )
                _write_skipped_variants_csv(out_dir, skipped_variants)
                last_seconds = time.perf_counter() - t_strict
                progress_tracker.record_completion(
                    dataset=dataset,
                    variant_name=strict_spec.variant_name,
                    stage="strict_skip",
                    last_seconds=last_seconds,
                )
                _write_master_partial_checkpoint(
                    out_dir=out_dir,
                    fold_rows=all_fold_rows,
                    trace_rows=all_trace_rows,
                    feature_catalog=feature_catalog,
                )
                strict_fold_df = pd.DataFrame(all_fold_rows)
                strict_trace_df = pd.DataFrame(all_trace_rows)
                strict_summary_df, strict_feature_profile_df, _strict_summary_sidecar = build_summary_frames(
                    strict_fold_df,
                    strict_trace_df,
                    feature_catalog,
                )
                _write_dataset_checkpoint_artifacts(
                    out_dir=out_dir,
                    dataset=dataset,
                    summary_df=strict_summary_df,
                    feature_profile_df=strict_feature_profile_df,
                    fold_df=strict_fold_df,
                    trace_df=strict_trace_df,
                    partial=True,
                )
                continue
            run_result = run_variant_dataset(
                context,
                strict_spec,
                fold_dir=args.fold_dir,
                max_folds=args.max_folds,
                no_cf=args.no_cf,
            )
            all_fold_rows.extend(run_result["fold_rows"])
            all_trace_rows.extend(run_result["trace_rows"])
            sidecar_details["datasets"][context.dataset]["fold_details"].setdefault(strict_spec.variant_name, [])
            sidecar_details["datasets"][context.dataset]["fold_details"][strict_spec.variant_name].extend(run_result["details"])
            last_seconds = time.perf_counter() - t_strict
            print(
                "[ABLATE][STRICT-DONE] "
                f"index={strict_index}/{len(strict_specs)} dataset={dataset} "
                f"name={strict_spec.variant_name} seconds={last_seconds:.2f}",
                flush=True,
            )
            progress_tracker.record_completion(
                dataset=dataset,
                variant_name=strict_spec.variant_name,
                stage="strict",
                last_seconds=last_seconds,
            )
            _write_master_partial_checkpoint(
                out_dir=out_dir,
                fold_rows=all_fold_rows,
                trace_rows=all_trace_rows,
                feature_catalog=feature_catalog,
            )
            strict_fold_df = pd.DataFrame(all_fold_rows)
            strict_trace_df = pd.DataFrame(all_trace_rows)
            strict_summary_df, strict_feature_profile_df, _strict_summary_sidecar = build_summary_frames(
                strict_fold_df,
                strict_trace_df,
                feature_catalog,
            )
            _write_dataset_checkpoint_artifacts(
                out_dir=out_dir,
                dataset=dataset,
                summary_df=strict_summary_df,
                feature_profile_df=strict_feature_profile_df,
                fold_df=strict_fold_df,
                trace_df=strict_trace_df,
                partial=True,
            )
        fold_df = pd.DataFrame(all_fold_rows)
        trace_df = pd.DataFrame(all_trace_rows)
        summary_df, feature_profile_df, summary_sidecar = build_summary_frames(fold_df, trace_df, feature_catalog)
        print(
            "[ABLATE][SUMMARY-REFRESH] "
            f"summary_rows={len(summary_df)} feature_profile_rows={len(feature_profile_df)} "
            f"trace_rows={len(trace_df)}",
            flush=True,
        )

    ranked_findings_df, ranking_debug = build_ranked_findings(summary_df)
    sidecar_details["summary"] = summary_sidecar
    sidecar_details["ranking"] = ranking_debug
    sidecar_details["strict_selection"] = strict_selection
    sidecar_details["skipped_variants"] = skipped_variants

    summary_df.to_csv(master_summary_path, index=False)
    ranked_findings_df.to_csv(ranked_findings_path, index=False)
    _write_skipped_variants_csv(out_dir, skipped_variants)
    _write_json(master_sidecar_path, sidecar_details)
    print(
        "[ABLATE][WRITE] "
        f"master_summary={master_summary_path} ranked_findings={ranked_findings_path} "
        f"sidecar={master_sidecar_path}",
        flush=True,
    )

    if not trace_df.empty:
        for dataset in datasets:
            _write_dataset_checkpoint_artifacts(
                out_dir=out_dir,
                dataset=dataset,
                summary_df=summary_df,
                feature_profile_df=feature_profile_df,
                fold_df=fold_df,
                trace_df=trace_df,
                partial=False,
            )

    progress_tracker.close()
    print(
        "[ABLATE][DONE] "
        f"seconds={time.perf_counter() - t0:.2f} datasets={datasets}",
        flush=True,
    )

    return {
        "provenance_path": str(provenance_path),
        "master_summary_path": str(master_summary_path),
        "master_sidecar_path": str(master_sidecar_path),
        "ranked_findings_path": str(ranked_findings_path),
        "skipped_variants_path": str(skipped_variants_path),
        "datasets": datasets,
    }


if __name__ == "__main__":
    main()
