"""Method adapters with a common raw-query generation contract."""

import itertools
import random
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from .config import DESIRED_CLASS, MAX_CANDIDATES


@dataclass
class GenerationResult:
    candidates: pd.DataFrame = field(default_factory=pd.DataFrame)
    generated_candidate_count: int = 0
    evaluated_candidate_count: int = 0
    internally_rejected_count: int = 0
    error_type: str = ""
    unsupported_reason: str = ""
    candidate_space: str = "encoded"
    error_traceback: str = ""


class EncodedSpace:
    def __init__(self, train_frame, feature_schema):
        self.feature_names = list(feature_schema["feature_names"])
        self.numeric = list(feature_schema["numeric_features"])
        self.categorical = list(feature_schema["categorical_features"])
        self.numeric_medians = {
            name: float(pd.to_numeric(train_frame[name], errors="coerce").median())
            for name in self.numeric
        }
        self.domains = {
            name: sorted(train_frame[name].dropna().astype(str).unique().tolist())
            for name in self.categorical
        }
        self.codes = {name: {value: index for index, value in enumerate(values)} for name, values in self.domains.items()}

    def encode(self, frame):
        out = pd.DataFrame(index=frame.index)
        for name in self.numeric:
            out[name] = pd.to_numeric(frame[name], errors="coerce").fillna(self.numeric_medians[name]).astype(float)
        for name in self.categorical:
            values = frame[name].astype("string").fillna("__MISSING__").astype(str)
            out[name] = values.map(self.codes[name]).fillna(-1).astype(float)
        return out.loc[:, self.feature_names]

    def decode(self, frame):
        out = pd.DataFrame(index=frame.index)
        for name in self.numeric:
            out[name] = pd.to_numeric(frame[name], errors="coerce")
        for name in self.categorical:
            values = self.domains[name]
            codes = pd.to_numeric(frame[name], errors="coerce").round().astype("Int64")
            out[name] = codes.map({index: value for index, value in enumerate(values)}).astype("string")
        return out.loc[:, self.feature_names]


class EncodedModel:
    def __init__(self, bundle, space):
        self.bundle = bundle
        self.space = space
        # UFCE and AR inspect coef_ in a few code paths.  Exposing no fake
        # coefficients is safer than pretending ordinal categorical codes are
        # the frozen one-hot model.

    def predict(self, frame):
        encoded = frame.loc[:, self.space.feature_names].copy()
        raw = self.space.decode(encoded)
        return self.bundle.predict_raw(raw)

    def predict_proba(self, frame):
        encoded = frame.loc[:, self.space.feature_names].copy()
        raw = self.space.decode(encoded)
        # DiCE's classifier interface requires an (n_samples, n_classes)
        # matrix, while the evaluation helper intentionally exposes only the
        # positive-class vector.
        prepared = self.bundle._prepare(raw)
        return np.asarray(self.bundle.model.predict_proba(prepared))


def compute_mi_pairs(reference_encoded, feature_names, seed=0, top_k=100):
    """Compute pairwise MI with auditable estimator-call counts."""
    start = time.perf_counter()
    rows = []
    calls = 0
    for left, right in itertools.combinations(feature_names, 2):
        try:
            score = float(mutual_info_regression(
                reference_encoded[[left]], reference_encoded[right], random_state=seed
            )[0])
        except Exception:
            score = 0.0
        calls += 1
        rows.append((score, left, right))
    rows.sort(key=lambda item: (-item[0], item[1], item[2]))
    retained = [[left, right] for _, left, right in rows[:top_k]]
    return retained, {
        "mi_estimator_calls": int(calls),
        "retained_pairs": int(len(retained)),
        "mi_seconds": float(time.perf_counter() - start),
    }

def rank_mi_pairs(reference_encoded, feature_names, seed=0):
    """Return a deterministic, fully auditable score-ordered MI ranking."""
    started = time.perf_counter()
    rows = []
    for left, right in itertools.combinations(feature_names, 2):
        try:
            score = float(mutual_info_regression(
                reference_encoded[[left]], reference_encoded[right], random_state=seed
            )[0])
        except Exception:
            score = 0.0
        rows.append((score, left, right))
    rows.sort(key=lambda item: (-item[0], item[1], item[2]))
    ranked = [
        {"rank": index + 1, "score": score, "left_feature": left, "right_feature": right}
        for index, (score, left, right) in enumerate(rows)
    ]
    return ranked, {
        "mi_estimator_calls": int(len(rows)),
        "retained_pairs": int(len(rows)),
        "mi_seconds": float(time.perf_counter() - started),
    }

class BaseAdapter:
    name = "base"

    def __init__(self, raw_train, reference_frame, bundle, feature_schema, policies, regime_name, seed=0, mi_top_k=100,
                 method_name=None, shared_mi=None, shared_reference_state=None,
                 categorical_actionable=None, mi_pairs_override=None, mi_summary_override=None):
        self.name = method_name or self.name
        self.raw_train = raw_train
        self.reference_frame = reference_frame
        self.bundle = bundle
        self.feature_schema = feature_schema
        self.policies = policies
        self.regime_name = regime_name
        self.seed = seed
        shared_reference_state = shared_reference_state or {}
        self.space = shared_reference_state.get("space")
        if self.space is None:
            self.space = EncodedSpace(raw_train, feature_schema)
        self.reference_encoded = shared_reference_state.get("reference_encoded")
        if self.reference_encoded is None:
            self.reference_encoded = self.space.encode(reference_frame)
        self.model = shared_reference_state.get("model")
        if self.model is None:
            self.model = EncodedModel(bundle, self.space)
        self.shared_reference_state = shared_reference_state
        self.desired_reference = shared_reference_state.get("desired_reference")
        self.feature_names = list(feature_schema["feature_names"])
        self.actionable = list(policies[regime_name]["actionable_features"])
        self.categorical_actionable = list(categorical_actionable or [])
        self.policy = policies[regime_name]
        self.setup_seconds = 0.0
        self.mi_summary = {"mi_estimator_calls": 0, "retained_pairs": 0, "mi_seconds": 0.0}
        self.shared_mi = shared_mi
        self.mi_pairs_override = mi_pairs_override
        self.mi_summary_override = mi_summary_override
        self._setup(mi_top_k)

    def _setup(self, mi_top_k):
        self.setup_seconds = 0.0

    def generate(self, raw_query, desired_class=DESIRED_CLASS, raw_constraint_policy=None, max_candidates=MAX_CANDIDATES, seed=0, timeout_seconds=60):
        raise NotImplementedError


class UFCEAdapter(BaseAdapter):
    def _setup(self, mi_top_k):
        started = time.perf_counter()
        # UFCE pair search is restricted to actionable digital features.  This
        # avoids asking the legacy regression/categorical helper to perturb an
        # immutable one-hot domain while keeping every retained pair inside the
        # declared raw constraint policy.
        if self.name == "UFCE-FF1":
            self.mi_pairs = []
            self.mi_summary = {"mi_estimator_calls": 0, "retained_pairs": 0, "mi_seconds": 0.0}
        elif self.mi_pairs_override is not None:
            self.mi_pairs = [list(pair) for pair in self.mi_pairs_override]
            self.mi_summary = dict(self.mi_summary_override or {})
            self.mi_summary.setdefault("retained_pairs", len(self.mi_pairs))
        elif self.shared_mi is not None:
            self.mi_pairs, self.mi_summary = self.shared_mi
        else:
            self.mi_pairs, self.mi_summary = compute_mi_pairs(self.reference_encoded, self.actionable, self.seed, mi_top_k)
        # Cache the desired-class reference rows once per method/regime.  The
        # old path decoded and predicted the full reference pool for every query.
        self.desired_reference = self.shared_reference_state.get("desired_reference")
        if self.desired_reference is None:
            self.desired_reference = self.reference_encoded.loc[
                self.model.predict(self.reference_encoded) == int(DESIRED_CLASS)
            ].reset_index(drop=True)
        try:
            from ufce.ufce_ff import cfmethods
            if getattr(cfmethods.ufc, "_external_reference_token", None) != id(self.reference_encoded):
                cfmethods.initUFCE(
                radius=1e9,
                # Preserve the legacy neighbor-count setting; the external
                # radius query remains uncapped and uses all rows in radius.
                n_neighbors=min(1000, max(1, len(self.reference_encoded) - 1)),
                )
                cfmethods.ufc._external_reference_token = id(self.reference_encoded)
            cfmethods.ufc._external_categorical_features = set(self.categorical_actionable)
            if len(self.desired_reference) > 10000:
                cfmethods.ufc.cache_reference_bounds(self.desired_reference)
            self.neighbor_backend_policy = "exact_all_rows_if_conservative_radius_bound_proves_superset_else_kdtree"
            self.cfmethods = cfmethods
        except Exception as exc:
            self.cfmethods = None
            self.setup_error = str(exc)
        self.setup_seconds = float(time.perf_counter() - started)

    def generate(self, raw_query, desired_class=DESIRED_CLASS, raw_constraint_policy=None, max_candidates=MAX_CANDIDATES, seed=0, timeout_seconds=60):
        if self.cfmethods is None:
            return GenerationResult(error_type="ufce_import_error", unsupported_reason=getattr(self, "setup_error", "UFCE import failed"))
        random.seed(seed)
        np.random.seed(seed)
        query_encoded = self.space.encode(raw_query)
        reference = self.reference_encoded
        desired_reference = self.desired_reference
        if desired_reference.empty:
            return GenerationResult()
        uf = {}
        for feature in self.actionable:
            rule = raw_constraint_policy["features"].get(feature, {})
            if not rule.get("immutable", True):
                uf[feature] = float(rule["upper"] - rule["lower"])
        for feature in self.categorical_actionable:
            rule = raw_constraint_policy["features"].get(feature, {})
            if not rule.get("immutable", True):
                codes = [self.space.codes[feature][str(value)] for value in rule.get("allowed_values", []) if str(value) in self.space.codes[feature]]
                if len(codes) >= 2:
                    uf[feature] = [min(codes), max(codes)]
        # A raw query with a missing actionable value makes that feature
        # immutable for this query. UFCE's interval builder indexes every
        # feature in f2change through `uf`, so pass only features that have an
        # explicitly materialized per-query movement interval.
        query_actionable = [feature for feature in self.actionable if feature in uf] + [feature for feature in self.categorical_actionable if feature in uf]
        perturb_steps = {
            feature: (
                float(raw_constraint_policy["features"][feature].get("step") or 1)
                if raw_constraint_policy["features"][feature].get("integer")
                else 0.1
            )
            for feature in query_actionable
        }
        query_mi_pairs = [
            pair for pair in self.mi_pairs
            if len(pair) == 2 and pair[0] in uf and pair[1] in uf
        ]
        protected = [name for name in self.feature_names if name not in self.actionable and name not in self.categorical_actionable]
        # The upstream UFCE routines use `numf` to decide which fields may be
        # generated.  All encoded columns are numeric, while `protected` and
        # `uf` enforce the raw action policy.
        try:
            if self.name == "UFCE-FF1":
                result = self.cfmethods.sfexp(
                    reference, desired_reference, query_encoded, uf, perturb_steps, query_actionable,
                    self.feature_names, self.categorical_actionable, self.model, desired_class, max_candidates,
                    self.feature_names, return_stats=True,
                )
            elif self.name == "UFCE-FF2":
                result = self.cfmethods.dfexp(
                    reference, desired_reference, query_encoded, uf, query_mi_pairs, query_actionable,
                    self.categorical_actionable, self.feature_names, protected, self.model, desired_class, max_candidates,
                    self.feature_names, return_stats=True,
                )
            else:
                result = self.cfmethods.tfexp(
                    reference, desired_reference, query_encoded, uf, query_mi_pairs, query_actionable,
                    self.categorical_actionable, query_actionable, protected, self.model, desired_class, max_candidates,
                    self.feature_names, return_stats=True,
                )
            candidates = result[0] if isinstance(result, tuple) else result
            if not isinstance(candidates, pd.DataFrame):
                candidates = pd.DataFrame()
            candidates = candidates.loc[:, [name for name in self.feature_names if name in candidates.columns]].drop_duplicates().head(max_candidates)
            return GenerationResult(
                candidates=candidates.reset_index(drop=True),
                generated_candidate_count=int(len(candidates)),
                evaluated_candidate_count=int(len(candidates)),
            )
        except Exception as exc:
            return GenerationResult(error_type=type(exc).__name__, unsupported_reason=str(exc))


class DiceAdapter(BaseAdapter):
    name = "DiCE"

    def _setup(self, mi_top_k):
        started = time.perf_counter()
        self.mi_summary = {"mi_estimator_calls": 0, "retained_pairs": 0, "mi_seconds": 0.0}
        try:
            import dice_ml
            self.dice_ml = dice_ml
            train_raw = self.reference_frame.loc[:, self.feature_names].copy()
            self.numeric_imputations = {}
            self.categorical_imputations = {}
            for feature in self.feature_schema["numeric_features"]:
                values = pd.to_numeric(self.raw_train[feature], errors="coerce")
                median = values.median()
                self.numeric_imputations[feature] = float(median) if pd.notna(median) else 0.0
                train_raw[feature] = pd.to_numeric(train_raw[feature], errors="coerce").fillna(self.numeric_imputations[feature]).astype(float)
            for feature in self.feature_schema["categorical_features"]:
                values = self.raw_train[feature].dropna().astype(str)
                mode = values.mode()
                self.categorical_imputations[feature] = str(mode.iloc[0]) if len(mode) else "__MISSING__"
                train_raw[feature] = train_raw[feature].astype("string").fillna(self.categorical_imputations[feature]).astype(str)
            train_raw["__target__"] = self.bundle.predict_raw(self.reference_frame.loc[:, self.feature_names])
            self.dice_data = dice_ml.Data(
                dataframe=train_raw,
                continuous_features=list(self.feature_schema["numeric_features"]),
                outcome_name="__target__",
            )
            # A sampled reference can omit a legal training category. Keep its
            # rows unchanged; query-time proxying is limited to immutable
            # categories and restored before independent raw verification.
            self.reference_categorical_domains = {
                feature: set(train_raw[feature].astype(str).unique().tolist())
                for feature in self.feature_schema["categorical_features"]
            }
            self.dice_model = dice_ml.Model(model=self.bundle.model, backend="sklearn")
            self.explainer = dice_ml.Dice(self.dice_data, self.dice_model, method="random")
        except Exception as exc:
            self.dice_ml = None
            self.setup_error = str(exc)
            self.setup_traceback = traceback.format_exc()
        self.setup_seconds = float(time.perf_counter() - started)

    def generate(self, raw_query, desired_class=DESIRED_CLASS, raw_constraint_policy=None, max_candidates=MAX_CANDIDATES, seed=0, timeout_seconds=60):
        if self.dice_ml is None:
            return GenerationResult(error_type="dice_import_error", unsupported_reason=getattr(self, "setup_error", "DiCE setup failed"), candidate_space="raw", error_traceback=getattr(self, "setup_traceback", ""))
        random.seed(seed)
        np.random.seed(seed)
        query_raw = raw_query.loc[:, self.feature_names].copy()
        for feature, value in self.numeric_imputations.items():
            query_raw[feature] = pd.to_numeric(query_raw[feature], errors="coerce").fillna(value)
        for feature, value in self.categorical_imputations.items():
            query_raw[feature] = query_raw[feature].astype("string").fillna(value).astype(str)
        query_for_dice = query_raw.copy()
        policy_features = raw_constraint_policy["features"]
        for feature, domain in self.reference_categorical_domains.items():
            value = str(query_for_dice.iloc[0][feature])
            if value in domain:
                continue
            rule = policy_features.get(feature, {})
            if not rule.get("immutable", True):
                return GenerationResult(candidate_space="raw", unsupported_reason="actionable category %s=%r is absent from reference schema" % (feature, value))
            if not domain:
                return GenerationResult(candidate_space="raw", unsupported_reason="empty reference category domain: %s" % feature)
            query_for_dice.loc[query_for_dice.index[0], feature] = sorted(domain)[0]
        permitted = {}
        features_to_vary = []
        for feature in self.actionable:
            rule = raw_constraint_policy["features"].get(feature, {})
            if not rule.get("immutable", True) and float(rule["upper"]) > float(rule["lower"]):
                permitted[feature] = [float(rule["lower"]), float(rule["upper"])]
                features_to_vary.append(feature)
        for feature in self.categorical_actionable:
            rule = raw_constraint_policy["features"].get(feature, {})
            if not rule.get("immutable", True):
                values = list(rule.get("allowed_values", []))
                if len(values) >= 2:
                    permitted[feature] = values
                    features_to_vary.append(feature)
        if not features_to_vary:
            return GenerationResult(candidate_space="raw")
        try:
            explanation = self.explainer.generate_counterfactuals(
                query_for_dice,
                total_CFs=max_candidates,
                desired_class=int(desired_class),
                features_to_vary=features_to_vary,
                permitted_range=permitted,
                sample_size=1000,
                random_seed=int(seed),
            )
            if explanation is None or not explanation.cf_examples_list:
                return GenerationResult(candidate_space="raw")
            candidate = explanation.cf_examples_list[0].final_cfs_df
            if candidate is None:
                return GenerationResult(candidate_space="raw")
            candidate = candidate.drop(columns=["__target__"], errors="ignore")
            candidate = candidate.loc[:, self.feature_names].head(max_candidates)
            # Restore exact immutable raw values (including missing values and
            # schema proxies) before the independent verifier checks results.
            for feature, rule in policy_features.items():
                if rule.get("immutable", True):
                    candidate.loc[:, feature] = raw_query.iloc[0][feature]
            return GenerationResult(candidates=candidate.reset_index(drop=True), generated_candidate_count=int(len(candidate)), evaluated_candidate_count=int(len(candidate)), candidate_space="raw")
        except Exception as exc:
            return GenerationResult(error_type=type(exc).__name__, unsupported_reason=str(exc), candidate_space="raw", error_traceback=traceback.format_exc())


class ARAdapter(BaseAdapter):
    name = "AR"

    def _setup(self, mi_top_k):
        self.mi_summary = {"mi_estimator_calls": 0, "retained_pairs": 0, "mi_seconds": 0.0}
        started = time.perf_counter()
        try:
            from recourse import ActionSet, RecourseBuilder
            self.ActionSet = ActionSet
            self.RecourseBuilder = RecourseBuilder
            self.numeric_features = list(self.feature_schema["numeric_features"])
            self.category_feature = "dedicacion"
            self.numeric_medians = {
                name: float(pd.to_numeric(self.raw_train[name], errors="coerce").median())
                for name in self.numeric_features
            }
            self.action_reference = self.reference_frame.loc[:, self.numeric_features].copy()
            for name in self.numeric_features:
                self.action_reference[name] = pd.to_numeric(self.action_reference[name], errors="coerce").fillna(self.numeric_medians[name])
            self.action_reference[self.category_feature] = (
                self.reference_frame[self.category_feature].astype("string").map({"TC": 0, "TP": 1}).fillna(0).astype(int)
            )
        except Exception as exc:
            self.RecourseBuilder = None
            self.setup_error = str(exc)
            self.setup_traceback = traceback.format_exc()
        self.setup_seconds = float(time.perf_counter() - started)

    def _raw_linear_model(self, raw_query, mutable_names):
        model = self.bundle.model.named_steps["model"]
        coef = np.asarray(model.coef_, dtype=float).reshape(-1)
        scaler = self.bundle.preprocessor.named_transformers_["numeric"].named_steps["scaler"]
        raw_coefs = {name: float(coef[index] / scaler.scale_[index]) for index, name in enumerate(self.numeric_features)}
        factual = raw_query.loc[:, self.feature_names].copy()
        baseline_score = float(np.asarray(self.bundle.model.decision_function(self.bundle._prepare(factual))).reshape(-1)[0])
        vector = {}
        coefs = {}
        for feature in mutable_names:
            if feature == self.category_feature:
                current = str(factual.iloc[0][feature])
                other = "TP" if current == "TC" else "TC"
                changed = factual.copy()
                changed.loc[changed.index[0], feature] = other
                other_score = float(np.asarray(self.bundle.model.decision_function(self.bundle._prepare(changed))).reshape(-1)[0])
                vector[feature] = 1.0 if current == "TP" else 0.0
                coefs[feature] = other_score - baseline_score if current == "TC" else baseline_score - other_score
            else:
                vector[feature] = float(pd.to_numeric(factual.iloc[0][feature], errors="coerce"))
                coefs[feature] = raw_coefs[feature]
        intercept = baseline_score - sum(coefs[name] * vector[name] for name in mutable_names)
        return np.asarray([coefs[name] for name in mutable_names], dtype=float), float(intercept), np.asarray([vector[name] for name in mutable_names], dtype=float)

    def generate(self, raw_query, desired_class=DESIRED_CLASS, raw_constraint_policy=None, max_candidates=MAX_CANDIDATES, seed=0, timeout_seconds=60):
        if self.RecourseBuilder is None:
            return GenerationResult(unsupported_reason=getattr(self, "setup_error", "AR import failed"), candidate_space="raw", error_traceback=getattr(self, "setup_traceback", ""))
        mutable_names = []
        custom_bounds = {}
        feature_types = {}
        for feature in self.actionable:
            rule = raw_constraint_policy["features"].get(feature, {})
            if rule.get("immutable", True) or float(rule.get("upper", 0.0)) <= float(rule.get("lower", 0.0)):
                continue
            mutable_names.append(feature)
            custom_bounds[feature] = (float(rule["lower"]), float(rule["upper"]), "absolute")
            feature_types[feature] = "integer" if rule.get("integer") else "continuous"
        cat_rule = raw_constraint_policy["features"].get(self.category_feature, {})
        factual_cat = raw_query.iloc[0].get(self.category_feature)
        if not cat_rule.get("immutable", True) and not pd.isna(factual_cat) and str(factual_cat) in ("TC", "TP"):
            mutable_names.append(self.category_feature)
            custom_bounds[self.category_feature] = (0, 1, "absolute")
            feature_types[self.category_feature] = "binary"
        if not mutable_names:
            return GenerationResult(candidate_space="raw")
        try:
            coefficients, intercept, x = self._raw_linear_model(raw_query, mutable_names)
            keep = [i for i, name in enumerate(mutable_names) if name == self.category_feature or coefficients[i] > 1e-12]
            mutable_names = [mutable_names[i] for i in keep]
            coefficients = coefficients[keep]
            x = x[keep]
            custom_bounds = {name: custom_bounds[name] for name in mutable_names}
            feature_types = {name: feature_types[name] for name in mutable_names}
            if not mutable_names:
                return GenerationResult(candidate_space="raw")
            baseline_score = float(np.asarray(self.bundle.model.decision_function(self.bundle._prepare(raw_query.loc[:, self.feature_names]))).reshape(-1)[0])
            intercept = baseline_score - float(np.dot(coefficients, x))
            action_set = self.ActionSet(
                self.action_reference.loc[:, mutable_names],
                custom_bounds=custom_bounds,
                default_bounds=(0.0, 1.0, "absolute"),
                print_flag=False,
                check_flag=False,
            )
            action_set.set_alignment(coefficients)
            for index, feature in enumerate(mutable_names):
                action_set[feature].actionable = True
                if feature == self.category_feature:
                    action_set[feature].step_direction = 0
                    continue
                action_set[feature].step_direction = 1
                rule = raw_constraint_policy["features"][feature]
                if rule.get("integer"):
                    action_set[feature].variable_type = int
                    action_set[feature].step_type = "absolute"
                    action_set[feature].step_size = 1
                else:
                    action_set[feature].variable_type = float
                    action_set[feature].step_type = "absolute"
                    action_set[feature].step_size = (float(rule["upper"]) - float(rule["lower"])) / 20.0
                    action_set[feature]._grid = np.linspace(float(rule["lower"]), float(rule["upper"]), 21)
            builder = self.RecourseBuilder(
                action_set=action_set,
                x=x,
                coefficients=coefficients,
                intercept=intercept,
                mip_cost_type="total",
                solver="cplex",
                print_flag=False,
            )
            solutions = builder.populate(
                total_items=int(max_candidates),
                enumeration_type="distinct_subsets",
                time_limit=max(0.1, float(timeout_seconds)),
                display_flag=False,
            )
            candidates = []
            for solution in solutions:
                if not solution.get("feasible"):
                    continue
                action = np.asarray(solution["actions"], dtype=float)
                candidate = raw_query.loc[:, self.feature_names].iloc[0].copy()
                for index, feature in enumerate(mutable_names):
                    new_value = x[index] + action[index]
                    if feature == self.category_feature:
                        candidate[feature] = "TP" if int(round(new_value)) == 1 else "TC"
                    else:
                        candidate[feature] = int(round(new_value)) if feature_types[feature] == "integer" else float(new_value)
                candidates.append(candidate)
            frame = pd.DataFrame(candidates, columns=self.feature_names).drop_duplicates().head(max_candidates)
            return GenerationResult(
                candidates=frame.reset_index(drop=True),
                generated_candidate_count=int(len(frame)),
                evaluated_candidate_count=int(len(frame)),
                candidate_space="raw",
            )
        except Exception as exc:
            if isinstance(exc, KeyError) and exc.args == ("cost_var_names",):
                return GenerationResult(
                    internally_rejected_count=1,
                    candidate_space="raw",
                    error_traceback="AR_EMPTY_ACTION_GRID_HANDLED: pinned actionable-recourse raises KeyError instead of returning an empty solution set.\n" + traceback.format_exc(),
                )
            return GenerationResult(error_type=type(exc).__name__, unsupported_reason=str(exc), candidate_space="raw", error_traceback=traceback.format_exc())


def make_adapter(name, *args, **kwargs):
    kwargs.setdefault("method_name", name)
    if name.startswith("UFCE-FF"):
        adapter = UFCEAdapter(*args, **kwargs)
        return adapter
    if name == "DiCE":
        return DiceAdapter(*args, **kwargs)
    if name == "AR":
        return ARAdapter(*args, **kwargs)
    raise ValueError("Unknown method %s" % name)
