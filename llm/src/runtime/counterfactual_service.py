from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Callable, List, Tuple

import numpy as np
import pandas as pd

from llm.src.runtime.reproducibility import deterministic_seed, sort_counterfactual_candidates
from llm.src.runtime.reason_codes import (
    GENERATION_ERROR,
    NO_CANDIDATE_GENERATED,
    NO_FEASIBLE_CF_FOUND,
    NO_VALID_FLIP,
    UFCE_EXECUTION_ERROR,
)
from llm.src.runtime.types import (
    CounterfactualCandidate,
    CounterfactualResult,
    RuntimeDebugTrace,
    UFCERequest,
)
from ufce.core.cfmethods import dfexp, initUFCE, sfexp, tfexp


Runner = Callable[..., Tuple[pd.DataFrame, float, List[int]]]


@dataclass(frozen=True)
class UFCECoreConfig:
    desired_outcome: int | None = None
    force_flip: bool = True
    allow_legacy_non_flipping_output: bool = False
    collect_raw_candidate_trace: bool = True


class CounterfactualService:
    def __init__(self, config: UFCECoreConfig | None = None) -> None:
        self.config = config or UFCECoreConfig()
        initUFCE()

    def generate(
        self,
        request: UFCERequest,
        debug_trace: RuntimeDebugTrace | None = None,
        deterministic_seed_value: int | None = None,
        config: UFCECoreConfig | None = None,
    ) -> CounterfactualResult:
        active_config = config or self.config
        factual_row = request.query_row.loc[:, request.bundle.feature_order].iloc[0]
        candidates: list[CounterfactualCandidate] = []
        had_exception = False
        trace_summary = {
            "force_flip": bool(active_config.force_flip),
            "raw_outputs_seen": 0,
            "non_flipping_rejected": 0,
            "flip_valid_outputs": 0,
        }

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
            trace_summary["raw_outputs_seen"] += len(normalized)
            normalized = self._apply_force_flip(
                normalized,
                request=request,
                config=active_config,
                trace_summary=trace_summary,
            )
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
                debug_trace.core_status = "valid_counterfactual_found"
                debug_trace.trace_summary = dict(trace_summary)
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
                debug_trace.core_status = "generation_error"
                debug_trace.trace_summary = dict(trace_summary)
                debug_trace.reject_path = {"reason_codes": [GENERATION_ERROR, UFCE_EXECUTION_ERROR]}
            return CounterfactualResult(
                feasible=False,
                candidates=[],
                reason_codes=[UFCE_EXECUTION_ERROR],
            )
        if debug_trace is not None:
            debug_trace.core_status = "no_valid_counterfactual"
            debug_trace.trace_summary = dict(trace_summary)
            if trace_summary["raw_outputs_seen"] == 0:
                debug_trace.reject_path = {"reason_codes": [NO_CANDIDATE_GENERATED]}
            else:
                debug_trace.reject_path = {"reason_codes": [NO_VALID_FLIP]}
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

    def _apply_force_flip(
        self,
        candidates: list[CounterfactualCandidate],
        *,
        request: UFCERequest,
        config: UFCECoreConfig,
        trace_summary: dict[str, Any],
    ) -> list[CounterfactualCandidate]:
        if not config.force_flip:
            if config.allow_legacy_non_flipping_output:
                return candidates
            return candidates
        desired_outcome = request.policy.desired_outcome if config.desired_outcome is None else config.desired_outcome
        kept: list[CounterfactualCandidate] = []
        for candidate in candidates:
            frame = pd.DataFrame([candidate.profile], columns=request.bundle.feature_order)
            prediction = int(request.bundle.lr.predict(frame)[0])
            if prediction == int(desired_outcome):
                trace_summary["flip_valid_outputs"] += 1
                kept.append(candidate)
            else:
                trace_summary["non_flipping_rejected"] += 1
        return kept

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
