from __future__ import annotations

from contextlib import redirect_stdout
import random
import sys
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from llm.src.runtime.counterfactual_service import CounterfactualService
from llm.src.runtime.reproducibility import deterministic_seed, sort_counterfactual_candidates
from llm.src.runtime.reason_codes import NO_FEASIBLE_CF_FOUND, UFCE_EXECUTION_ERROR
from llm.src.runtime.types import CounterfactualCandidate, CounterfactualResult, RuntimeContext, RuntimeDebugTrace, UFCERequest
from ufce.core.cfmethods import ar_cfexp, dice_cfexp


class CounterfactualBackend(Protocol):
    backend_name: str

    def generate(
        self,
        request: UFCERequest,
        context: RuntimeContext,
        *,
        deterministic_seed_value: int | None = None,
        debug_trace: RuntimeDebugTrace | None = None,
    ) -> CounterfactualResult:
        ...


class UFCEBackendAdapter:
    backend_name = "ufce"

    def __init__(self, service: CounterfactualService | None = None) -> None:
        self.service = service or CounterfactualService()

    def generate(
        self,
        request: UFCERequest,
        context: RuntimeContext,
        *,
        deterministic_seed_value: int | None = None,
        debug_trace: RuntimeDebugTrace | None = None,
    ) -> CounterfactualResult:
        del context
        if debug_trace is not None:
            debug_trace.backend_name = self.backend_name
        return self.service.generate(
            request,
            debug_trace=debug_trace,
            deterministic_seed_value=deterministic_seed_value,
        )


class DiCEBackendAdapter:
    backend_name = "dice"

    def generate(
        self,
        request: UFCERequest,
        context: RuntimeContext,
        *,
        deterministic_seed_value: int | None = None,
        debug_trace: RuntimeDebugTrace | None = None,
    ) -> CounterfactualResult:
        return _run_single_backend(
            backend_name=self.backend_name,
            method_label="DiCE",
            runner=lambda: _call_dice(request=request, context=context),
            request=request,
            context=context,
            deterministic_seed_value=deterministic_seed_value,
            debug_trace=debug_trace,
        )


class ARBackendAdapter:
    backend_name = "ar"

    def __init__(self) -> None:
        self._scaler_cache: dict[str, StandardScaler] = {}

    def generate(
        self,
        request: UFCERequest,
        context: RuntimeContext,
        *,
        deterministic_seed_value: int | None = None,
        debug_trace: RuntimeDebugTrace | None = None,
    ) -> CounterfactualResult:
        return _run_single_backend(
            backend_name=self.backend_name,
            method_label="AR",
            runner=lambda: _call_ar(
                request=request,
                context=context,
                scaler=self._get_scaler(context),
            ),
            request=request,
            context=context,
            deterministic_seed_value=deterministic_seed_value,
            debug_trace=debug_trace,
        )

    def _get_scaler(self, context: RuntimeContext) -> StandardScaler:
        dataset_name = str(context.dataset_name)
        scaler = self._scaler_cache.get(dataset_name)
        if scaler is None:
            scaler = StandardScaler().fit(
                context.bundle.Xtrain.loc[:, context.bundle.feature_order].copy()
            )
            self._scaler_cache[dataset_name] = scaler
        return scaler


def resolve_counterfactual_backend(name: str | None) -> CounterfactualBackend:
    normalized = "ufce" if name is None else str(name).strip().lower()
    if normalized == "ufce":
        return UFCEBackendAdapter()
    if normalized == "dice":
        return DiCEBackendAdapter()
    if normalized == "ar":
        return ARBackendAdapter()
    raise ValueError(f"Unsupported counterfactual backend: {name}")


def _run_single_backend(
    *,
    backend_name: str,
    method_label: str,
    runner,
    request: UFCERequest,
    context: RuntimeContext,
    deterministic_seed_value: int | None,
    debug_trace: RuntimeDebugTrace | None,
) -> CounterfactualResult:
    if debug_trace is not None:
        debug_trace.backend_name = backend_name
    try:
        output_df = _run_with_seed(runner=runner, deterministic_seed_value=deterministic_seed_value)
        candidates = _normalize_candidate_frame(
            method_name=method_label,
            output_df=output_df,
            request=request,
            context=context,
        )
        if debug_trace is not None:
            debug_trace.add_ufce_method(method_label, "success" if candidates else "empty", len(candidates))
    except Exception as exc:  # pragma: no cover - exercised through live runs and adapter tests
        if debug_trace is not None:
            debug_trace.add_service_error(method_label, str(exc))
            debug_trace.add_ufce_method(method_label, "error", 0, str(exc))
            debug_trace.reject_path = {"reason_codes": [UFCE_EXECUTION_ERROR]}
        return CounterfactualResult(feasible=False, candidates=[], reason_codes=[UFCE_EXECUTION_ERROR])

    if not candidates:
        if debug_trace is not None:
            debug_trace.reject_path = {"reason_codes": [NO_FEASIBLE_CF_FOUND]}
        return CounterfactualResult(feasible=False, candidates=[], reason_codes=[NO_FEASIBLE_CF_FOUND])

    ordered = sort_counterfactual_candidates(
        candidates=candidates,
        feature_order=list(context.bundle.feature_order),
    )
    if debug_trace is not None:
        winner = ordered[0]
        debug_trace.winning_path = {
            "method": winner.method,
            "rank": winner.rank,
            "changed_features": list(winner.changed_features),
        }
        debug_trace.reject_path = None
    return CounterfactualResult(feasible=True, candidates=ordered, reason_codes=[])


def _run_with_seed(*, runner, deterministic_seed_value: int | None) -> pd.DataFrame:
    if deterministic_seed_value is None:
        return runner()
    with deterministic_seed(deterministic_seed_value):
        random.seed(deterministic_seed_value)
        np.random.seed(deterministic_seed_value)
        return runner()


def _call_dice(*, request: UFCERequest, context: RuntimeContext) -> pd.DataFrame:
    with redirect_stdout(sys.stderr):
        raw_df, _idx, _time_s, _flag = dice_cfexp(
            context.bundle.dataset_df.copy(),
            request.query_row.copy(),
            list(context.policy.numeric_features),
            list(context.policy.f2change),
            50,
            context.bundle.lr,
            dict(context.policy.uf),
            context.policy.label_col,
        )
    return raw_df


def _call_ar(*, request: UFCERequest, context: RuntimeContext, scaler: StandardScaler) -> pd.DataFrame:
    with redirect_stdout(sys.stderr):
        raw_df, _time_s, _idx = ar_cfexp(
            context.bundle.X.loc[:, context.bundle.feature_order].copy(),
            list(context.policy.numeric_features),
            context.bundle.lr,
            request.query_row.copy(),
            dict(context.policy.uf),
            scaler,
            context.bundle.Xtrain.loc[:, context.bundle.feature_order].copy(),
            list(context.policy.f2change),
        )
    return raw_df


def _normalize_candidate_frame(
    *,
    method_name: str,
    output_df: Any,
    request: UFCERequest,
    context: RuntimeContext,
) -> list[CounterfactualCandidate]:
    if not isinstance(output_df, pd.DataFrame) or output_df.empty:
        return []
    feature_order = list(context.bundle.feature_order)
    if not set(feature_order).issubset(output_df.columns):
        raise ValueError(f"{method_name} returned unexpected columns.")
    factual_row = request.query_row.loc[:, feature_order].iloc[0]
    normalized_df = output_df.loc[:, feature_order].reset_index(drop=True)
    candidates: list[CounterfactualCandidate] = []
    for index, row in normalized_df.iterrows():
        profile: dict[str, Any] = {}
        changed: list[str] = []
        for feature_name in feature_order:
            value = row[feature_name]
            feature_type = context.policy.feature_type_map[feature_name]
            if feature_type == "float":
                profile[feature_name] = float(value)
                if abs(float(value) - float(factual_row[feature_name])) > 1e-9:
                    changed.append(feature_name)
            else:
                profile[feature_name] = int(value)
                if int(value) != int(factual_row[feature_name]):
                    changed.append(feature_name)
        candidates.append(
            CounterfactualCandidate(
                method=method_name,
                rank=index + 1,
                profile=profile,
                changed_features=changed,
            )
        )
    return candidates
