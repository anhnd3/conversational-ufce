from __future__ import annotations

import math
import random
from typing import Any, Callable, List, Tuple

import numpy as np
import pandas as pd

from llm.src.runtime.reproducibility import deterministic_seed, sort_counterfactual_candidates
from llm.src.runtime.reason_codes import NO_FEASIBLE_CF_FOUND, UFCE_EXECUTION_ERROR
from llm.src.runtime.types import (
    CounterfactualCandidate,
    CounterfactualResult,
    RuntimeDebugTrace,
    UFCERequest,
)
from ufce.core.cfmethods import dfexp, initUFCE, sfexp, tfexp


Runner = Callable[..., Tuple[pd.DataFrame, float, List[int]]]


class CounterfactualService:
    def __init__(self) -> None:
        initUFCE()

    def generate(
        self,
        request: UFCERequest,
        debug_trace: RuntimeDebugTrace | None = None,
        deterministic_seed_value: int | None = None,
    ) -> CounterfactualResult:
        factual_row = request.query_row.loc[:, request.bundle.feature_order].iloc[0]
        candidates: list[CounterfactualCandidate] = []
        had_exception = False

        for index, (method_name, runner) in enumerate(self._ordered_runners()):
            try:
                output_df = self._run_method(
                    method_name,
                    runner,
                    request,
                    deterministic_seed_value=None if deterministic_seed_value is None else deterministic_seed_value + index,
                )
            except Exception as exc:  # pragma: no cover - verified via monkeypatch in tests
                had_exception = True
                if debug_trace is not None:
                    debug_trace.add_service_error(method_name, str(exc))
                    debug_trace.add_ufce_method(method_name, "error", 0, str(exc))
                continue

            normalized = self._normalize_candidates(method_name, output_df, factual_row, request)
            candidates.extend(normalized)
            if debug_trace is not None:
                status = "success" if normalized else "empty"
                debug_trace.add_ufce_method(method_name, status, len(normalized))

        if candidates:
            ordered_candidates = sort_counterfactual_candidates(
                candidates=candidates,
                feature_order=list(request.bundle.feature_order),
            )
            if debug_trace is not None:
                winner = ordered_candidates[0]
                debug_trace.winning_path = {
                    "method": winner.method,
                    "rank": winner.rank,
                    "changed_features": list(winner.changed_features),
                }
            return CounterfactualResult(
                feasible=True,
                candidates=ordered_candidates,
                reason_codes=[],
            )
        if had_exception:
            if debug_trace is not None:
                debug_trace.reject_path = {"reason_codes": [UFCE_EXECUTION_ERROR]}
            return CounterfactualResult(
                feasible=False,
                candidates=[],
                reason_codes=[UFCE_EXECUTION_ERROR],
            )
        if debug_trace is not None:
            debug_trace.reject_path = {"reason_codes": [NO_FEASIBLE_CF_FOUND]}
        return CounterfactualResult(
            feasible=False,
            candidates=[],
            reason_codes=[NO_FEASIBLE_CF_FOUND],
        )

    def _ordered_runners(self) -> list[tuple[str, Runner]]:
        return [
            ("sfexp", sfexp),
            ("dfexp", dfexp),
            ("tfexp", tfexp),
        ]

    def _run_method(
        self,
        method_name: str,
        runner: Runner,
        request: UFCERequest,
        deterministic_seed_value: int | None,
    ) -> pd.DataFrame:
        if deterministic_seed_value is None:
            return self._execute_runner(method_name, runner, request)
        with deterministic_seed(deterministic_seed_value):
            random.seed(deterministic_seed_value)
            np.random.seed(deterministic_seed_value)
            return self._execute_runner(method_name, runner, request)

    def _execute_runner(self, method_name: str, runner: Runner, request: UFCERequest) -> pd.DataFrame:
        if method_name == "sfexp":
            dataframe, _elapsed, _indexes = runner(
                request.feature_matrix,
                request.positive_class_pool,
                request.query_row,
                request.policy.uf,
                request.policy.step,
                request.policy.f2change,
                request.policy.numeric_features,
                request.policy.categorical_features,
                request.bundle.lr,
                request.policy.desired_outcome,
                1,
                request.bundle.feature_order,
            )
            return dataframe
        dataframe, _elapsed, _indexes = runner(
            request.feature_matrix,
            request.positive_class_pool,
            request.query_row,
            request.policy.uf,
            request.mi_feature_pairs,
            request.policy.numeric_features,
            request.policy.categorical_features,
            request.policy.f2change,
            request.policy.protected_features,
            request.bundle.lr,
            request.policy.desired_outcome,
            1,
            request.bundle.feature_order,
        )
        return dataframe

    def _normalize_candidates(
        self,
        method_name: str,
        output_df: Any,
        factual_row: pd.Series,
        request: UFCERequest,
    ) -> list[CounterfactualCandidate]:
        if not isinstance(output_df, pd.DataFrame):
            raise TypeError("UFCE method '{0}' did not return a DataFrame.".format(method_name))
        if output_df.empty or output_df.shape[1] == 0:
            return []
        if not set(request.bundle.feature_order).issubset(output_df.columns):
            raise ValueError("UFCE method '{0}' returned unexpected columns.".format(method_name))

        normalized_df = output_df.loc[:, request.bundle.feature_order].reset_index(drop=True)
        candidates: list[CounterfactualCandidate] = []
        for index, row in normalized_df.iterrows():
            profile = self._serialize_profile_row(row, request)
            changed_features = self._changed_features(factual_row, row, request)
            candidates.append(
                CounterfactualCandidate(
                    method=method_name,
                    rank=index + 1,
                    profile=profile,
                    changed_features=changed_features,
                )
            )
        return candidates

    def _serialize_profile_row(self, row: pd.Series, request: UFCERequest) -> dict[str, Any]:
        profile: dict[str, Any] = {}
        for feature_name in request.bundle.feature_order:
            value = row[feature_name]
            feature_type = request.policy.feature_type_map[feature_name]
            if feature_type == "float":
                profile[feature_name] = float(value)
            else:
                profile[feature_name] = int(value)
        return profile

    def _changed_features(self, factual_row: pd.Series, candidate_row: pd.Series, request: UFCERequest) -> list[str]:
        changed: list[str] = []
        for feature_name in request.bundle.feature_order:
            factual_value = factual_row[feature_name]
            candidate_value = candidate_row[feature_name]
            feature_type = request.policy.feature_type_map[feature_name]
            if feature_type == "float":
                if not math.isclose(float(factual_value), float(candidate_value), rel_tol=0.0, abs_tol=1e-9):
                    changed.append(feature_name)
            else:
                if int(factual_value) != int(candidate_value):
                    changed.append(feature_name)
        return changed
