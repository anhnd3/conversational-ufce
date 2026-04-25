# pip install dice_ml
# pip install actionable-recourse

import time
import random
import pandas as pd
import dice_ml
import recourse as rs
from dice_ml.utils import helpers # helper functions
from sklearn.model_selection import train_test_split

from ufce import UFCE
ufc = None


import pandas as pd
import numpy as np

# 1. Initialize the global variable as None
ufc = None

# 2. Add the custom initialization function
def initUFCE(
    radius=500,
    n_neighbors=1000,
    contprox_metric='euclidean',
    min_act=3,
    min_feas=2,
    atol=1e-5,
):
    """
    Call this function in your reproduction script before running experiments.
    It overrides the global 'ufc' instance with custom parameters.
    """
    global ufc
    ufc = UFCE(
        radius=radius,
        n_neighbors=n_neighbors,
        contprox_metric=contprox_metric,
        min_actionable_other=min_act,
        min_actionable_feasible_other=min_feas,
        atol=atol,
    )

# calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def _apply_distance_scaler(df, distance_scaler):
    """
    Apply an affine scaler (value - medians) / mads on configured columns.
    This keeps non-scaled columns unchanged and supports constant-column guards.
    """
    if distance_scaler is None or df is None:
        return df
    if not isinstance(df, pd.DataFrame):
        return df
    if df.empty:
        return df.copy()

    scale_cols = distance_scaler.get("scale_cols", None)
    if scale_cols is None:
        scale_cols = list(getattr(distance_scaler.get("medians", pd.Series(dtype=float)), "index", []))
    cols = list(scale_cols)
    if len(cols) == 0:
        return df.copy()
    cols = [c for c in cols if c in df.columns]
    if len(cols) == 0:
        return df.copy()

    out = df.copy()
    med = distance_scaler["medians"].reindex(cols)
    mad = distance_scaler["mads"].reindex(cols)
    out.loc[:, cols] = (out.loc[:, cols] - med) / mad

    const_cols = [c for c in distance_scaler.get("constant_cols", []) if c in cols]
    if len(const_cols) != 0:
        out.loc[:, const_cols] = 0.0
    return out

# find the row with the least Euclidean distance for specified continuous features
def find_best_row(df, test_instance, continuous_features, distance_scaler=None):
    dist_df = _apply_distance_scaler(df, distance_scaler)
    dist_test = _apply_distance_scaler(test_instance, distance_scaler)

    dist_df['proximity'] = dist_df.apply(
        lambda row: euclidean_distance(
            row[continuous_features].values,
            dist_test[continuous_features].values,
        ),
        axis=1,
    )

    best_row = df.loc[dist_df['proximity'].idxmin()]
    best_row = best_row.copy()
    best_row["proximity"] = float(dist_df.loc[dist_df["proximity"].idxmin(), "proximity"])
    return best_row


def _filter_flipping_candidates(candidates, model, desired_outcome, order, apply_filter=True):
    if not isinstance(candidates, pd.DataFrame) or candidates.empty:
        return pd.DataFrame()
    if not apply_filter:
        return candidates.reset_index(drop=True)
    if all(col in candidates.columns for col in order):
        pred_input = candidates[order]
    else:
        pred_input = candidates
    preds = np.asarray(model.predict(pred_input)).reshape(-1)
    return candidates.loc[preds == int(desired_outcome)].reset_index(drop=True)


def _init_method_stats(n_instances):
    return {
        "n_instances": int(n_instances),
        "n_candidates_raw_total": 0,
        "n_candidates_flip_total": 0,
        "n_instances_with_flip_cf": 0,
        "n_empty_after_filter": 0,
        "coverage": 0.0,
    }


def _finalize_method_stats(stats):
    denom = int(stats.get("n_instances", 0))
    num = int(stats.get("n_instances_with_flip_cf", 0))
    stats["coverage"] = float(num / denom) if denom > 0 else 0.0
    return stats


def _debug_enabled(debug_ctx):
    return isinstance(debug_ctx, dict) and bool(debug_ctx.get("enabled", False))


def _debug_trace_positions(debug_ctx):
    if not _debug_enabled(debug_ctx):
        return set()
    trace_positions = debug_ctx.get("trace_positions", [])
    return set(int(v) for v in trace_positions)


def _debug_scaled_l2_numf(factual_df, cf_df, numf, distance_scaler):
    if not isinstance(factual_df, pd.DataFrame) or not isinstance(cf_df, pd.DataFrame):
        return float("nan")
    if factual_df.empty or cf_df.empty:
        return float("nan")
    cols = [c for c in numf if c in factual_df.columns and c in cf_df.columns]
    if len(cols) == 0:
        return float("nan")
    factual_dist = _apply_distance_scaler(factual_df.loc[:, cols], distance_scaler)
    cf_dist = _apply_distance_scaler(cf_df.loc[:, cols], distance_scaler)
    return float(np.linalg.norm(cf_dist.iloc[0].to_numpy(dtype=float) - factual_dist.iloc[0].to_numpy(dtype=float)))


def _debug_emit(debug_ctx, event_type, payload, fallback_message=None):
    if isinstance(debug_ctx, dict) and bool(debug_ctx.get("enabled", False)):
        events = debug_ctx.setdefault("events", [])
        events.append(
            {
                "event_type": str(event_type),
                **dict(payload),
            }
        )
        if bool(debug_ctx.get("structured_only", False)):
            return
    if fallback_message is not None:
        print(fallback_message)


def _trace_frame(frame, order):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=list(order))
    if not all(col in frame.columns for col in order):
        return pd.DataFrame(columns=list(order))
    return frame.loc[:, list(order)].copy().reset_index(drop=True)

def changes_per_cf(x, cf):
    features = (list(x.columns))
    nbr_changes = 0
    for j in features:
        if cf[j].values != x[j].values:
            nbr_changes += 1
    return nbr_changes

def feasible_2method(test, cflist, changeable_features, uf):
    temp = pd.DataFrame()
    features = (list(test.columns))
    for x in range(len(cflist)):
        nbr_act, f_list = ufc.nbr_actionable_cfn(test, cflist[x:x + 1], features, changeable_features)
        spar_count = changes_per_cf(test, cflist[x:x + 1])
        if nbr_act/spar_count >= 0.2: 
            count_in = 0
            for j in f_list:
                limit = test[j][0] + uf[j]
                if cflist.at[x, j] <= limit:
                    count_in += 1
            if nbr_act == count_in:
                temp = pd.concat([temp, cflist[x:x + 1]], axis=0, ignore_index=True)
    return temp

def dice_cfexp(df, X_test, numf, f2change, no_cf, bb, uf, outcome_label):
    """
    :param df: dataset
    :param X_test: test set
    :param numf: numerical features
    :param f2change: features to change
    :param outcome_label: class label
    :param no_cf: required number of counterfactuals
    :param bb: blackbox model
    :return dice_cfs: dice counterfactuals
    """
    start = time.perf_counter()
    d = dice_ml.Data(dataframe=df, continuous_features=numf, outcome_name= outcome_label)
    m = dice_ml.Model(model=bb, backend="sklearn")
    exp = dice_ml.Dice(d, m, method="random")
    dice_cfs = pd.DataFrame()
    cf = pd.DataFrame()
    flag = 0
    idx = []
    for x in range(len(X_test)):
        cf = pd.DataFrame()
        try:
            e1 = exp.generate_counterfactuals(X_test[x:x+1], total_CFs=no_cf, desired_class="opposite", features_to_vary= f2change, permitted_range=None)#, permitted_range=uf_ranges
            if e1 is None or e1.cf_examples_list is None or len(e1.cf_examples_list) == 0:
                continue
            cf_example = e1.cf_examples_list[0]
            if cf_example is None or cf_example.final_cfs_df is None or len(cf_example.final_cfs_df) == 0:
                continue
            foundcfs = cf_example.final_cfs_df[0:no_cf]
            if foundcfs is not None and len(foundcfs) > 0:
                flag = 1
                if len(foundcfs) > 1:
                    best_row = find_best_row(foundcfs, X_test[x:x+1], numf)
                    cf = best_row.to_frame().T
                else:
                    cf = foundcfs.iloc[0:1].copy()
        except Exception as e:
            print(f"Error: {e}")
        if flag != 0 and cf.empty != True:
            dice_cfs = pd.concat([dice_cfs, cf], ignore_index = True, axis = 0)
            flag = 0
            idx.append(x)
    if len(dice_cfs) != 0:
        flag = 1
    end = time.perf_counter()
    dicetime = end-start
    if len(X_test) != 0:
        dicetime = dicetime/len(X_test)
    return dice_cfs, idx, dicetime, flag

def dice_cfexp_in(df, X_test, numf, f2change, no_cf, bb, uf, outcome_label):
    """
    :param df: dataset
    :param X_test: test set
    :param numf: numerical features
    :param f2change: features to change
    :param outcome_label: class label
    :param no_cf: required number of counterfactuals
    :param bb: blackbox model
    :return dice_cfs: dice counterfactuals
    """
    start = time.perf_counter()
    d = dice_ml.Data(dataframe=df, continuous_features=numf, outcome_name= outcome_label)
    m = dice_ml.Model(model=bb, backend="sklearn")
    exp = dice_ml.Dice(d, m, method="random")
    dice_cfs = pd.DataFrame()
    cf = pd.DataFrame()
    flag = 0
    idx = []
    for x in range(len(X_test)):
        cf = pd.DataFrame()
        uf_ranges = {}
        current = X_test.iloc[x]
        for feat in f2change:
            if feat not in numf or feat not in current.index or feat not in uf:
                continue
            try:
                current_val = float(current[feat])
                uf_delta = float(uf[feat])
            except (TypeError, ValueError):
                continue
            low, high = sorted([current_val, current_val + uf_delta])
            uf_ranges[feat] = [low, high]
        permitted_range = uf_ranges if len(uf_ranges) > 0 else None
        try:
            e1 = exp.generate_counterfactuals(X_test[x:x+1], total_CFs=no_cf, desired_class="opposite", features_to_vary= f2change, permitted_range=permitted_range)#, permitted_range=uf_ranges
            if e1 is None or e1.cf_examples_list is None or len(e1.cf_examples_list) == 0:
                continue
            cf_example = e1.cf_examples_list[0]
            if cf_example is None or cf_example.final_cfs_df is None or len(cf_example.final_cfs_df) == 0:
                continue
            foundcfs = cf_example.final_cfs_df[0:no_cf]
            if foundcfs is not None and len(foundcfs) > 0:
                flag = 1
                if len(foundcfs) > 1:
                    best_row = find_best_row(foundcfs, X_test[x:x+1], numf)
                    cf = best_row.to_frame().T
                else:
                    cf = foundcfs.iloc[0:1].copy()
        except Exception as e:
            print(f"Error: {e}")
        if flag != 0 and cf.empty != True:
            dice_cfs = pd.concat([dice_cfs, cf], ignore_index = True, axis = 0)
            flag = 0
            idx.append(x)
    if len(dice_cfs) != 0:
        flag = 1
    end = time.perf_counter()
    dicetime = end-start
    if len(X_test) != 0:
        dicetime = dicetime/len(X_test)
    return dice_cfs, idx, dicetime, flag

def ar_cfexp(X, numf, bb, X_test, uf, scaler, X_train, f2change):
    """
    :param X: X data
    :param numf: numerical features
    :param bb: blackbox model
    :param X_test: test set
    :return ar_cfs: ar counterfactuals
    """
    start = time.perf_counter()
    flag = 0
    idx = []
    A = rs.ActionSet(X)
    from IPython.core.display import display, HTML
    clf = bb
    A.set_alignment(clf)
    finalarcfs = pd.DataFrame()
    cf = pd.DataFrame()
    for x in range(len(X_test)):
        try:
            fs = rs.Flipset(X_test[x:x + 1].values, action_set=A, clf=clf)
            fs.populate(enumeration_type='distinct_subsets', total_items = 5) 
            f_list = numf
            candi_cfs = pd.DataFrame() 
            if len(fs) > 1:
                for i in range(len(fs)):
                    feat2change = fs.df['features'][i]
                    values_2change = fs.df['x_new'][i]
                    changed_instance = X_test[x:x+1].copy()
                    for f, i in enumerate(feat2change):
                        changed_instance[i] = values_2change[f]
                    candi_cfs = pd.concat([candi_cfs, changed_instance], ignore_index=True, axis=0)
                if len(candi_cfs) > 1:
                    best_row = find_best_row(candi_cfs, X_test[x:x+1], numf)
                    cf = best_row.to_frame().T
                    idx.append(x)
                    finalarcfs = pd.concat([finalarcfs, cf], ignore_index=True, axis=0)
                    finalarcfs = finalarcfs.drop(['proximity'], axis=1)
            else:
                feat2change = fs.df['features']
                values_2change = fs.df['x_new']
                changed_instance = X_test[x:x+1].copy()
                for f, i in enumerate(feat2change):
                    changed_instance[i] = values_2change[f]
                idx.append(x)
                finalarcfs = pd.concat([finalarcfs, changed_instance], ignore_index=True, axis=0)
        except Exception as e:
            print(f"Error: {e}")
    end = time.perf_counter()
    artime = end-start
    if len(X_test) != 0:
        artime = artime / len(X_test)
    return finalarcfs, artime, idx

def sfexp(
    X,
    data_lab1,
    X_test,
    uf,
    step,
    f2change,
    numf,
    catf,
    bb,
    desired_outcome,
    k,
    order,
    return_stats=False,
    flip_filter_enabled=True,
    distance_data_lab1=None,
    distance_X_test=None,
    distance_scaler=None,
    debug_ctx=None,
    return_trace=False,
):
    """
    :param X:
    :param data_lab1:
    :param X_test:
    :param uf:
    :param step:
    :param f2change:
    :param numf:
    :param catf:
    :param bb:
    :param desired_outcome:
    :param k:
    :return:
    """
    oneF_cfdf = pd.DataFrame()
    testout = pd.DataFrame()
    cf = pd.DataFrame()
    found_indexes = []
    intervald = dict()
    method_stats = _init_method_stats(len(X_test))
    trace_enabled = _debug_enabled(debug_ctx)
    trace_positions = _debug_trace_positions(debug_ctx)
    trace_rows = []
    start = time.perf_counter()
    for t in range(len(X_test)):
        n = 0
        search_data = data_lab1 if distance_data_lab1 is None else distance_data_lab1
        search_query = X_test[t:t+1] if distance_X_test is None else distance_X_test[t:t+1]
        trace_this_instance = trace_enabled and (int(t) in trace_positions)
        if trace_this_instance:
            nn, idx, nn_meta = ufc.NNkdtree(search_data, search_query, return_meta=True)
            source_id_map = debug_ctx.get("source_row_ids", {})
            source_row_id = source_id_map.get(int(t), int(t))
            _debug_emit(
                debug_ctx,
                "ufce1_neighborhood",
                {
                    "instance_pos": int(t),
                    "source_row_id": int(source_row_id),
                    "k_retrieved": nn_meta.get("k_retrieved", "NA(radius-only)"),
                    "within_radius_count": int(nn_meta.get("within_radius_count", len(idx))),
                    "neighbor_idx_head": [int(i) for i in nn_meta.get("neighbor_idx_head", [int(i) for i in idx[:5]])],
                },
                fallback_message=(
                    "[DBG][UFCE1][NN] "
                    f"instance_pos={t}, source_row_id={source_row_id}, "
                    f"k_retrieved={nn_meta.get('k_retrieved', 'NA(radius-only)')}, "
                    f"within_radius_count={nn_meta.get('within_radius_count', len(idx))}, "
                    f"neighbor_idx_head={nn_meta.get('neighbor_idx_head', [int(i) for i in idx[:5]])}"
                ),
            )
        else:
            nn, idx = ufc.NNkdtree(search_data, search_query)

        if trace_this_instance:
            old_radius = float(getattr(ufc, "radius", 0.0))
            for probe_radius in debug_ctx.get("radius_probe", []):
                probe_within_radius_count = 0
                probe_l2 = float("nan")
                try:
                    ufc.radius = float(probe_radius)
                    nn_probe, idx_probe, probe_meta = ufc.NNkdtree(search_data, search_query, return_meta=True)
                    probe_within_radius_count = int(probe_meta.get("within_radius_count", len(idx_probe)))
                    if distance_data_lab1 is not None:
                        nn_probe_raw = data_lab1.iloc[idx_probe].reset_index(drop=True)
                    else:
                        nn_probe_raw = nn_probe
                    if nn_probe_raw.empty != True:
                        interval_probe = ufc.make_intervals(nn_probe_raw, uf, f2change, X_test[t:t+1])
                        random_state = random.getstate()
                        try:
                            cc_probe = ufc.Single_F(
                                X_test[t:t+1],
                                catf,
                                interval_probe,
                                bb,
                                desired_outcome,
                                step,
                                debug_ctx=None,
                                instance_pos=None,
                            )
                        finally:
                            random.setstate(random_state)
                        cc_probe = _filter_flipping_candidates(
                            cc_probe,
                            bb,
                            desired_outcome,
                            order,
                            apply_filter=flip_filter_enabled,
                        )
                        if isinstance(cc_probe, pd.DataFrame) and cc_probe.empty != True:
                            if len(cc_probe) > 1:
                                best_probe = find_best_row(
                                    cc_probe,
                                    X_test[t:t+1],
                                    numf,
                                    distance_scaler=distance_scaler,
                                ).to_frame().T
                            else:
                                best_probe = cc_probe[:1].copy()
                            probe_l2 = _debug_scaled_l2_numf(
                                factual_df=X_test[t:t+1],
                                cf_df=best_probe,
                                numf=numf,
                                distance_scaler=distance_scaler,
                            )
                finally:
                    ufc.radius = old_radius
                _debug_emit(
                    debug_ctx,
                    "ufce1_radius_probe",
                    {
                        "instance_pos": int(t),
                        "radius": float(probe_radius),
                        "within_radius_count": int(probe_within_radius_count),
                        "chosen_cf_scaled_l2": float(probe_l2),
                    },
                    fallback_message=(
                        "[DBG][UFCE1][RadiusProbe] "
                        f"instance_pos={t}, radius={probe_radius}, "
                        f"within_radius_count={probe_within_radius_count}, "
                        f"chosen_cf_scaled_l2={probe_l2}"
                    ),
                )
        if distance_data_lab1 is not None:
            nn = data_lab1.iloc[idx].reset_index(drop=True)
        instance_raw_candidates = pd.DataFrame(columns=list(order))
        instance_flip_candidates = pd.DataFrame(columns=list(order))
        instance_selected = pd.DataFrame(columns=list(order))
        nn_meta_summary = {
            "within_radius_count": int(len(idx)),
            "radius": float(getattr(ufc, "radius", 0.0)),
            "n_neighbors": int(getattr(ufc, "n_neighbors", 0)),
        }
        if trace_this_instance and "nn_meta" in locals():
            nn_meta_summary = {
                "within_radius_count": int(nn_meta.get("within_radius_count", len(idx))),
                "radius": float(nn_meta.get("radius", getattr(ufc, "radius", 0.0))),
                "n_neighbors": int(getattr(ufc, "n_neighbors", 0)),
            }
        if nn.empty != True:
            interval = ufc.make_intervals(nn, uf, f2change, X_test[t:t+1])
            cc = ufc.Single_F(
                X_test[t:t+1],
                catf,
                interval,
                bb,
                desired_outcome,
                step,
                debug_ctx=debug_ctx if trace_enabled else None,
                instance_pos=t,
            )
            instance_raw_candidates = _trace_frame(cc, order)
            raw_count = len(cc) if isinstance(cc, pd.DataFrame) else 0
            method_stats["n_candidates_raw_total"] += int(raw_count)

            # Gate UFCE1 with the same flipping-candidate filter used by UFCE2/UFCE3.
            cc = _filter_flipping_candidates(
                cc,
                bb,
                desired_outcome,
                order,
                apply_filter=flip_filter_enabled,
            )
            instance_flip_candidates = _trace_frame(cc, order)
            flip_count = len(cc) if isinstance(cc, pd.DataFrame) else 0
            method_stats["n_candidates_flip_total"] += int(flip_count)
            if raw_count > 0 and flip_count == 0:
                method_stats["n_empty_after_filter"] += 1
            if flip_count > 0:
                method_stats["n_instances_with_flip_cf"] += 1

            if cc.empty != True:
                if len(cc) > 1:
                    best_row = find_best_row(cc, X_test[t:t+1], numf, distance_scaler=distance_scaler)
                    cf = best_row.to_frame().T
                    instance_selected = _trace_frame(cf, order)
                    found_indexes.append(t)
                    oneF_cfdf = pd.concat([oneF_cfdf, cf], ignore_index=True, axis=0)
                    oneF_cfdf = oneF_cfdf.drop(['proximity'], axis=1)
                else:
                    instance_selected = _trace_frame(cc[:1], order)
                    found_indexes.append(t)
                    oneF_cfdf = pd.concat([oneF_cfdf, cc[:1]], ignore_index=True, axis=0)
        if return_trace:
            trace_rows.append(
                {
                    "instance_pos": int(t),
                    "method": "UFCE1",
                    "generated_candidates_df": instance_raw_candidates,
                    "label_flip_candidates_df": instance_flip_candidates,
                    "selected_candidates_df": instance_selected,
                    "search_meta": nn_meta_summary,
                    "search_parameters": {
                        "radius": float(getattr(ufc, "radius", 0.0)),
                        "n_neighbors": int(getattr(ufc, "n_neighbors", 0)),
                    },
                    "source_path": "single_feature",
                }
            )
    end = time.perf_counter()
    onetime = end - start
    if len(X_test) != 0:
        onetime = onetime/len(X_test)
    method_stats = _finalize_method_stats(method_stats)
    if return_stats:
        if return_trace:
            return oneF_cfdf, onetime, found_indexes, method_stats, trace_rows
        return oneF_cfdf, onetime, found_indexes, method_stats
    if return_trace:
        return oneF_cfdf, onetime, found_indexes, trace_rows
    return oneF_cfdf, onetime, found_indexes 

def dfexp(
    X,
    data_lab1,
    X_test,
    uf,
    F,
    numf,
    catf,
    features,
    protectf,
    bb,
    desired_outcome,
    k,
    order,
    return_stats=False,
    flip_filter_enabled=True,
    distance_data_lab1=None,
    distance_X_test=None,
    distance_scaler=None,
    return_trace=False,
):
    start = time.perf_counter()
    desired_outcome = desired_outcome
    k = k
    foundidx = []
    intervald = dict()
    perturb_step = {}
    twoF_cfdf = pd.DataFrame()
    testout = pd.DataFrame()
    protectedf = protectf
    method_stats = _init_method_stats(len(X_test))
    trace_rows = []

    for t in range(len(X_test)):
        search_data = data_lab1 if distance_data_lab1 is None else distance_data_lab1
        search_query = X_test[t:t+1] if distance_X_test is None else distance_X_test[t:t+1]
        nn, idx = ufc.NNkdtree(search_data, search_query)
        if distance_data_lab1 is not None:
            nn = data_lab1.iloc[idx].reset_index(drop=True)
        instance_generated = pd.DataFrame(columns=list(order))
        instance_flip_candidates = pd.DataFrame(columns=list(order))
        instance_selected = pd.DataFrame(columns=list(order))
        if nn.empty != True:
            intervals = ufc.make_uf_nn_interval(nn, uf, F[:], X_test[t:t+1])
            cc2, cfsexp2 = ufc.Double_F(X, X_test[t:t+1], protectedf, F[:], catf, numf, intervals, features, bb, desired_outcome, order, k)
            raw_generated_df = cfsexp2 if isinstance(cfsexp2, pd.DataFrame) and cfsexp2.empty != True else cc2
            instance_generated = _trace_frame(raw_generated_df, order)
            raw_primary = len(cc2) if isinstance(cc2, pd.DataFrame) else 0
            raw_explore = len(cfsexp2) if isinstance(cfsexp2, pd.DataFrame) else 0
            method_stats["n_candidates_raw_total"] += int(raw_primary + raw_explore)

            cc2 = _filter_flipping_candidates(
                cc2,
                bb,
                desired_outcome,
                order,
                apply_filter=flip_filter_enabled,
            )
            selected_rows = _filter_flipping_candidates(
                cfsexp2,
                bb,
                desired_outcome,
                order,
                apply_filter=flip_filter_enabled,
            )
            instance_flip_candidates = _trace_frame(selected_rows, order)
            flip_primary = len(cc2) if isinstance(cc2, pd.DataFrame) else 0
            flip_explore = len(selected_rows) if isinstance(selected_rows, pd.DataFrame) else 0
            method_stats["n_candidates_flip_total"] += int(flip_primary + flip_explore)
            if (raw_primary + raw_explore) > 0 and (flip_primary + flip_explore) == 0:
                method_stats["n_empty_after_filter"] += 1
            if (flip_primary + flip_explore) > 0:
                method_stats["n_instances_with_flip_cf"] += 1

            # print("[DBG][UFCE2] returned best rows:", len(cc2), "explore rows:", len(cfsexp2))
            if cc2.empty != True:
                if len(cc2) > 1:
                    best_row = find_best_row(
                        cc2.copy(),
                        X_test[t:t+1],
                        numf,
                        distance_scaler=distance_scaler,
                    )
                    cf = best_row.to_frame().T
                    instance_selected = _trace_frame(cf, order)
                    foundidx.append(t)
                    twoF_cfdf = pd.concat([twoF_cfdf, cf], ignore_index=True, axis=0)
                    if 'proximity' in twoF_cfdf.columns:
                        twoF_cfdf = twoF_cfdf.drop(['proximity'], axis=1)
                else:
                    instance_selected = _trace_frame(cc2[:1], order)
                    foundidx.append(t)
                    twoF_cfdf = pd.concat([twoF_cfdf, cc2[:1]], ignore_index=True, axis=0)
            else:
                if selected_rows.empty != True:
                    if len(selected_rows) > 1:
                        best_row = find_best_row(
                            selected_rows.copy(),
                            X_test[t:t+1],
                            numf,
                            distance_scaler=distance_scaler,
                        )
                        selected = best_row.to_frame().T
                        if 'proximity' in selected.columns:
                            selected = selected.drop(['proximity'], axis=1)
                    else:
                        selected = selected_rows[:1]
                    instance_selected = _trace_frame(selected, order)
                    foundidx.append(t)
                    twoF_cfdf = pd.concat([twoF_cfdf, selected], ignore_index=True, axis=0)
        if return_trace:
            trace_rows.append(
                {
                    "instance_pos": int(t),
                    "method": "UFCE2",
                    "generated_candidates_df": instance_generated,
                    "label_flip_candidates_df": instance_flip_candidates,
                    "selected_candidates_df": instance_selected,
                    "search_meta": {
                        "within_radius_count": int(len(idx)),
                        "radius": float(getattr(ufc, "radius", 0.0)),
                        "n_neighbors": int(getattr(ufc, "n_neighbors", 0)),
                    },
                    "search_parameters": {
                        "radius": float(getattr(ufc, "radius", 0.0)),
                        "n_neighbors": int(getattr(ufc, "n_neighbors", 0)),
                        "mi_feature_pairs": [list(item) for item in F[:]],
                    },
                    "source_path": "double_feature",
                    "raw_primary_count": int(raw_primary if "raw_primary" in locals() else 0),
                    "raw_explore_count": int(raw_explore if "raw_explore" in locals() else 0),
                }
            )
    end = time.perf_counter()
    twotime = end-start
    if len(X_test) != 0:
        twotime = twotime/len(X_test)
    method_stats = _finalize_method_stats(method_stats)
    if return_stats:
        if return_trace:
            return twoF_cfdf, twotime, foundidx, method_stats, trace_rows
        return twoF_cfdf, twotime, foundidx, method_stats
    if return_trace:
        return twoF_cfdf, twotime, foundidx, trace_rows
    return twoF_cfdf, twotime, foundidx 

def tfexp(
    X,
    data_lab1,
    X_test,
    uf,
    F,
    numf,
    catf,
    feature2change,
    protectdf,
    bb,
    desired_outcome,
    k,
    order,
    return_stats=False,
    flip_filter_enabled=True,
    distance_data_lab1=None,
    distance_X_test=None,
    distance_scaler=None,
    return_trace=False,
):
    """
    :param X:
    :param data_lab1:
    :param X_test:
    :param uf:
    :param F:
    :param numf:
    :param catf:
    :param feature2change:
    :param protectdf:
    :param bb:
    :param desired_outcome:
    :param k:
    :return:
    """
    start = time.perf_counter()
    perturb_step = {}
    foundidx = []
    intervald = dict()
    threeF_cfdf = pd.DataFrame()
    testout = pd.DataFrame()
    method_stats = _init_method_stats(len(X_test))
    trace_rows = []
    for t in range(len(X_test)):
        n=0
        search_data = data_lab1 if distance_data_lab1 is None else distance_data_lab1
        search_query = X_test[t:t+1] if distance_X_test is None else distance_X_test[t:t+1]
        nn, idx = ufc.NNkdtree(search_data, search_query)
        if distance_data_lab1 is not None:
            nn = data_lab1.iloc[idx].reset_index(drop=True)
        instance_generated = pd.DataFrame(columns=list(order))
        instance_flip_candidates = pd.DataFrame(columns=list(order))
        instance_selected = pd.DataFrame(columns=list(order))
        if nn.empty != True:
            intervals = ufc.make_uf_nn_interval(nn, uf, F[:], X_test[t:t+1]) 
            cc3, cfsexp2 = ufc.Triple_F(X, X_test[t:t+1], protectdf, F[:], catf, numf, intervals, feature2change, bb, desired_outcome, order, k) 
            raw_generated_df = cfsexp2 if isinstance(cfsexp2, pd.DataFrame) and cfsexp2.empty != True else cc3
            instance_generated = _trace_frame(raw_generated_df, order)
            raw_primary = len(cc3) if isinstance(cc3, pd.DataFrame) else 0
            raw_explore = len(cfsexp2) if isinstance(cfsexp2, pd.DataFrame) else 0
            method_stats["n_candidates_raw_total"] += int(raw_primary + raw_explore)

            cc3 = _filter_flipping_candidates(cc3, bb, desired_outcome, order, apply_filter=flip_filter_enabled)
            selected_rows = _filter_flipping_candidates(
                cfsexp2,
                bb,
                desired_outcome,
                order,
                apply_filter=flip_filter_enabled,
            )
            instance_flip_candidates = _trace_frame(selected_rows, order)
            flip_primary = len(cc3) if isinstance(cc3, pd.DataFrame) else 0
            flip_explore = len(selected_rows) if isinstance(selected_rows, pd.DataFrame) else 0
            method_stats["n_candidates_flip_total"] += int(flip_primary + flip_explore)
            if (raw_primary + raw_explore) > 0 and (flip_primary + flip_explore) == 0:
                method_stats["n_empty_after_filter"] += 1
            if (flip_primary + flip_explore) > 0:
                method_stats["n_instances_with_flip_cf"] += 1

            # print("[DBG][UFCE3] returned best rows:", len(cc3), "explore rows:", len(cfsexp2))
            if cc3.empty != True:
                if len(cc3) > 1:
                    best_row = find_best_row(
                        cc3.copy(),
                        X_test[t:t+1],
                        numf,
                        distance_scaler=distance_scaler,
                    )
                    cf = best_row.to_frame().T
                    instance_selected = _trace_frame(cf, order)
                    foundidx.append(t)
                    threeF_cfdf = pd.concat([threeF_cfdf, cf], ignore_index=True, axis=0)
                    if 'proximity' in threeF_cfdf.columns:
                        threeF_cfdf = threeF_cfdf.drop(['proximity'], axis=1)
                else:
                    instance_selected = _trace_frame(cc3[:1], order)
                    foundidx.append(t)
                    threeF_cfdf = pd.concat([threeF_cfdf, cc3[:1]], ignore_index=True, axis=0)
            else:
                if selected_rows.empty != True:
                    if len(selected_rows) > 1:
                        best_row = find_best_row(
                            selected_rows.copy(),
                            X_test[t:t+1],
                            numf,
                            distance_scaler=distance_scaler,
                        )
                        selected = best_row.to_frame().T
                        if 'proximity' in selected.columns:
                            selected = selected.drop(['proximity'], axis=1)
                    else:
                        selected = selected_rows[:1]
                    instance_selected = _trace_frame(selected, order)
                    foundidx.append(t)
                    threeF_cfdf = pd.concat([threeF_cfdf, selected], ignore_index=True, axis=0)
        if return_trace:
            trace_rows.append(
                {
                    "instance_pos": int(t),
                    "method": "UFCE3",
                    "generated_candidates_df": instance_generated,
                    "label_flip_candidates_df": instance_flip_candidates,
                    "selected_candidates_df": instance_selected,
                    "search_meta": {
                        "within_radius_count": int(len(idx)),
                        "radius": float(getattr(ufc, "radius", 0.0)),
                        "n_neighbors": int(getattr(ufc, "n_neighbors", 0)),
                    },
                    "search_parameters": {
                        "radius": float(getattr(ufc, "radius", 0.0)),
                        "n_neighbors": int(getattr(ufc, "n_neighbors", 0)),
                        "mi_feature_pairs": [list(item) for item in F[:]],
                    },
                    "source_path": "triple_feature",
                    "raw_primary_count": int(raw_primary if "raw_primary" in locals() else 0),
                    "raw_explore_count": int(raw_explore if "raw_explore" in locals() else 0),
                }
            )
    end = time.perf_counter()
    threetime = end-start
    if len(X_test) != 0:
        threetime = threetime / len(X_test)
    method_stats = _finalize_method_stats(method_stats)
    if return_stats:
        if return_trace:
            return threeF_cfdf, threetime, foundidx, method_stats, trace_rows
        return threeF_cfdf, threetime, foundidx, method_stats
    if return_trace:
        return threeF_cfdf, threetime, foundidx, trace_rows
    return threeF_cfdf, threetime, foundidx 
