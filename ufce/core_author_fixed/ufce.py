# -*- coding: utf-8 -*-
"""
Created on Tuesday 22 FEB 2022

@author: Muhammad Suffian
"""

import pandas as pd
import numpy as np
import json
import math
import random
#%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split, cross_val_score

from simplenlg.framework import *
from simplenlg.lexicon import *
from simplenlg.realiser.english import *
from simplenlg.phrasespec import *
from simplenlg.features import *
from .goodness import *


"""
 This is the module that could be utilized to take the user feedback in the form of user preferences
 Initially the user preferences will be the intervals, later these preferences would be taken from the user with an interface.
"""


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


class UFCE():
    
    def __init__(
        self,
        radius=500,
        n_neighbors=1000,
        contprox_metric='euclidean',
        actionability_threshold=0.30,
        atol=1e-5,
    ):
        # self.dataset = 'bank'
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.contprox_metric = contprox_metric
        self.actionability_threshold = float(actionability_threshold)
        self.atol = float(atol)
        # Optional runtime debug context injected by caller (e.g., hypertune script).
        self.debug_ctx = None
        #self.selected_features = user_selected_features
        #self.intervals = user_preferences
        #self.features = ['age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education', 'Mortgage', 'SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard']

#categorical data handler function
    def categorical_handler(self, test_instance, user_cat_feature_list):
        """
        :param test_instance:
        :param user_cat_feature_list:
        :return:
        """
        for feature in user_cat_feature_list:
            if float(test_instance.loc[:, feature].values) != 1:
                test_instance.loc[:, feature] = 1.0
        return test_instance


    def barplot(self, methods, means, x_pos, serror, title, ylabel, path, save=False):
        """
        :param methods: names of cf-methods
        :param means: list of mean values of any evaluation metric
        :param x_pos: len(methods) to plot on x-axes
        :param serror: list of standard error of evaluation metric
        :param title: title for the figure
        :param ylabel: y-label for figure
        :param path: path to save the figure
        :param save: specify boolean flag to save or not
        :return: plot the figure and save it
        """
        fig, ax = plt.subplots(figsize=(3.5,3))
        colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']
        ax.bar(x_pos, means, yerr=serror, align='center', alpha=0.5, ecolor='black', color=colors, log = False, width=0.5, capsize=5)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods)
        ax.set_title(title)
        ax.yaxis.grid(False)
        # Save the figure and show
        plt.tight_layout()
        if save==True:
            plt.savefig(path, dpi=400, bbox_inches="tight")
        plt.show()

    def make_mi_scores(self, X, y, discrete_features):
        """
        :param X: features in the data set
        :param y: label in the data set
        :param discrete_features:
        :return:
        """
        from sklearn.feature_selection import mutual_info_classif
        mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        return mi_scores

    def plot_mi_scores(self, scores):
        """
        :param scores: list of feature scores
        :return:
        """
        scores = scores.sort_values(ascending=True)
        width = np.arange(len(scores))
        ticks = list(scores.index)
        plt.barh(width, scores)
        plt.yticks(width, ticks)
        plt.title("Individual Scores")
        
    def CF_Gower_dist(self, test, cf):
        """
        :param test: test instance
        :param cf: counterfactual
        :return: distance between test instance and counterfactual
        """
        distance = 0
        X = pd.concat([test, cf], ignore_index=True, axis=0)
        X = np.asarray(X)
        X = gower.gower_matrix(X)
        print(X)
        d = gower.gower_topn(X, X, n=2)
        print(d)
        distance = d['values'][0]
        return distance
    
    def store_CFs_with_Gower(self, path_to_cfdf, Xtest, test_inst_no, filename, k):
        """
        :param path_to_cfdf: a specific dataframe holding CFs from one, two, three methods
        :param Xtest: test set
        :param test_inst_no: specific test instance
        :param filename: filename
        :return: found_cfs: set of found counterfactuals
        """
        found_cfs = pd.DataFrame()
        found = 0
        try:
            cfs = pd.read_csv(path_to_cfdf + filename)
        except pd.io.common.EmptyDataError:
            print('File is empty')
        else:
            cfs.drop_duplicates(inplace=True)
            out = pd.DataFrame()
            out = pd.concat([Xtest[test_inst_no:test_inst_no+1], cfs], ignore_index=True, axis=0)
            X = np.asarray(out)
            gower.gower_matrix(X)
            d = gower.gower_topn(out, out, n=k)
            for i in range(len(d['index'])):
                v = d['index'][i]
                c = cfs.iloc[v]
                c = c.to_frame().T
                c = c.reset_index(drop=True)
                found_cfs = pd.concat([found_cfs, c], ignore_index=True, axis=0)
            found_cfs.drop_duplicates(inplace=True)
        found_cfs.to_csv(path_to_cfdf + 'gower_cfs' + '.csv', index=False)
        return found_cfs
    
    def get_top_MI_features (self, X, num_f):
        """
        :param X: feature space
        :param num_f: numerical features
        :return: list of list for feature pairs in descending order
        """
        from sklearn.feature_selection import mutual_info_regression
        from sklearn.feature_selection import mutual_info_classif
        matrix = dict()
        for f in num_f:
            for fl in num_f:
                mi_scores = mutual_info_regression(X[f].to_frame(), X[fl], discrete_features='auto')#
                if fl !=f:
                    score = mi_scores[0]
                    matrix[score] = [fl,f]
        D = dict(sorted(matrix.items(), reverse=True))
        filtered = []
        for i in D.keys():
            p1 = set(D[i])
            if p1 not in filtered:
                filtered.append(p1)
        feature_list = []
        for f in filtered:
            flist = list(f)
            feature_list.append(flist)
        return feature_list
    
    def NNkdtree(self, data_lab1, test_inst, return_meta=False):
        """
        :param data_lab1: desired space
        :param test_inst:
        :param radius: radius for search in the space, should be faithful to data distribution
        :return:
        """
        import numpy as np
        from scipy.spatial import KDTree
        tree = KDTree(data_lab1)
        idx = tree.query_ball_point(test_inst.values[0], r=self.radius)
        nn = pd.DataFrame.from_records(tree.data[idx], columns=test_inst.columns)
        if return_meta:
            meta = {
                "k_retrieved": "NA(radius-only)",
                "within_radius_count": int(len(idx)),
                "radius": float(self.radius),
                "neighbor_idx_head": [int(i) for i in idx[:5]],
            }
            return nn, idx, meta
        return nn, idx

    def get_cfs_validated(self, df, model, desired_outcome):
        """
        :param df: dataframe of found nearest neighbours
        :param model: ML model
        :param desired_outcome:
        :return:
        """
        cfs = pd.DataFrame()
        for c in range(len(df)):
            p = model.predict(df[c:c+1])
            if p==desired_outcome:
                cfs = pd.concat([cfs, df[c:c+1]], ignore_index=True)
        return cfs
    
    def feat2change(self, test, nn_cf):
        """
        :param test: test instance
        :param nn_cf: nearest counterfactual
        :return:
        """
        feat2change = []
        for f in test.columns:
            if test[f].values != nn_cf[f].values:
                feat2change.append(f)
        return feat2change
    
    def make_intervals(self, nn, uf, feat2change, test):
        """
        :param nn: nearest neighbourhood dataframe
        :param uf: user feedback dictionary
        :param feat2change: feature to change
        :param test: test instance
        :return: feature intervals (dictionary)
        """
        intervals = dict()
        for f in feat2change:
            if f in uf.keys() and f in test.columns and f in nn.columns:
                lower, upper = self._feedback_bounds(test[f].iloc[0], uf[f])
                nn_lower = float(nn[f].min())
                nn_upper = float(nn[f].max())
                lower = max(float(lower), nn_lower)
                upper = min(float(upper), nn_upper)
                if lower <= upper:
                    intervals[f] = [lower, upper]
        return intervals
   
    def make_uf_nn_interval(self,nn, uf, feature_pairs, test):
        """
        :param nn: nearest neighbourhood data points
        :param uf: user feedback dictionary
        :param feature_pairs: feature pairs
        :param test: test instance
        :return: feature intervals dictionary
        """
        faithful_interval = dict()
        for featurepair in feature_pairs:
            f1 = featurepair[0]
            f2 = featurepair[1]
            for feature in (f1, f2):
                if feature not in uf or feature not in test.columns or feature not in nn.columns:
                    continue
                lower, upper = self._feedback_bounds(test[feature].iloc[0], uf[feature])
                nn_lower = float(nn[feature].min())
                nn_upper = float(nn[feature].max())
                lower = max(float(lower), nn_lower)
                upper = min(float(upper), nn_upper)
                if lower <= upper:
                    faithful_interval[feature] = [lower, upper]
        return faithful_interval

       
    def pred_for_binsearch(
        self,
        tempdf,
        feature,
        start,
        mid,
        end,
        model,
        candidate_value=None,
        debug_ctx=None,
        instance_pos=None,
    ):
        """
        :param tempdf: a temporary dataframe
        :param feature: feature
        :param start: start value
        :param mid: mid value
        :param: end: end value
        :param model: ML blackbox
        :return pred, tempdf: prediction and related dataframe
        """
        assigned = candidate_value if candidate_value is not None else mid
        tempdf.loc[:, feature] = assigned
        pred = model.predict(tempdf)
        if debug_ctx is not None and bool(debug_ctx.get("enabled", False)):
            trace_positions = debug_ctx.get("trace_positions_set", set())
            if instance_pos in trace_positions:
                pred_arr = np.asarray(pred).reshape(-1)
                pred_label = pred_arr[0] if pred_arr.size > 0 else "NA"
                assigned_value = tempdf[feature].iloc[0] if feature in tempdf.columns and len(tempdf) > 0 else "NA"
                print(
                    "[DBG][UFCE1][Assign] "
                    f"instance_pos={instance_pos}, feature={feature}, "
                    f"mid={mid}, candidate_value={candidate_value}, "
                    f"assigned_value={assigned_value}, pred={pred_label}"
                )
        return pred, tempdf
       
   
    def Single_F(
        self,
        test_instance,
        u_cat_f_list,
        user_term_intervals,
        model,
        outcome,
        step,
        debug_ctx=None,
        instance_pos=None,
    ):
        """
        :param test_instance:
        :param u_cat_f_list:
        :param user_term_intervals: user defined values for each feature
        :param model:
        :param outcome:
        :param step: values of feature distribution need to use in binary search for moving to next by adding this value 
        :return cfdfout: single feature counterfactuals
        """
        cfdfout = pd.DataFrame()
        tempdf = pd.DataFrame()
        tempdfcat = pd.DataFrame()
        
        trace_enabled = debug_ctx is not None and bool(debug_ctx.get("enabled", False))
        trace_positions = debug_ctx.get("trace_positions_set", set()) if trace_enabled else set()
        trace_this_instance = instance_pos in trace_positions if trace_enabled else False

        for feature in user_term_intervals.keys():
            if feature not in u_cat_f_list:
                tempdf = test_instance.copy()
                one_feature_data = pd.DataFrame()
                interval_term_range = user_term_intervals[feature]
                if len(interval_term_range) != 0 and interval_term_range[0] != interval_term_range[1]:
                    start = float(interval_term_range[0])
                    end = float(interval_term_range[1])
                    step_size = float(step.get(feature, 1.0))
                    cfdf = pd.DataFrame()
                    lower, upper = (start, end) if start <= end else (end, start)
                    if step_size <= 0:
                        step_size = 1.0
                    if float(step_size).is_integer() and float(lower).is_integer() and float(upper).is_integer():
                        f1_space = [item for item in range(int(lower), int(upper) + 1, int(step_size))]
                    else:
                        f1_space = list(np.round(np.arange(lower, upper + step_size, step_size), 6))
                    if start > end:
                        f1_space = list(reversed(f1_space))
                    while len(f1_space) != 0:
                        tempdf = test_instance.copy()
                        if len(f1_space) != 0:
                            low = 0
                            high = len(f1_space) - 1
                            mid = (high - low) // 2
                        candidate_value = f1_space[mid] if len(f1_space) > 0 else None
                        if trace_this_instance:
                            is_numf = feature in debug_ctx.get("numf_set", set())
                            if feature in debug_ctx.get("scale_cols_set", set()):
                                scaling_status = "scaled_numeric"
                                scaled_min, scaled_max = debug_ctx.get("scaled_bounds", {}).get(feature, ("NA", "NA"))
                            else:
                                scaling_status = "unchanged_non_numeric"
                                scaled_min, scaled_max = ("NA", "NA")
                            x_factual = test_instance[feature].iloc[0] if feature in test_instance.columns else "NA"
                            print(
                                "[DBG][UFCE1][Assign] "
                                f"instance_pos={instance_pos}, feature={feature}, x_factual={x_factual}, "
                                f"f1_space_len={len(f1_space)}, mid={mid}, candidate_value={candidate_value}, "
                                f"is_numf={is_numf}, scaling_status={scaling_status}, "
                                f"scaled_min={scaled_min}, scaled_max={scaled_max}"
                            )
                        pred, tempdf1 = self.pred_for_binsearch(
                            tempdf,
                            feature,
                            start,
                            mid,
                            end,
                            model,
                            candidate_value=candidate_value,
                            debug_ctx=debug_ctx,
                            instance_pos=instance_pos,
                        )
                        if trace_this_instance:
                            assigned_value = tempdf1[feature].iloc[0] if feature in tempdf1.columns else "NA"
                            pred_arr = np.asarray(pred).reshape(-1)
                            pred_label = pred_arr[0] if pred_arr.size > 0 else "NA"
                            flip_hit = bool(pred_label == outcome)
                            print(
                                "[DBG][UFCE1][Assign] "
                                f"instance_pos={instance_pos}, feature={feature}, candidate_vs_assigned="
                                f"({candidate_value}, {assigned_value}), flip_hit={flip_hit}"
                            )
                        pred_label = np.asarray(pred).reshape(-1)
                        if pred_label.size > 0 and int(pred_label[0]) == int(outcome):
                            cfdf = tempdf1.copy()
                            cfdfout = pd.concat([cfdfout, cfdf], ignore_index=True, axis=0)
                        try:
                            del f1_space[:mid+1]
                        except:
                            pass
            else:
                tempdfcat = test_instance.copy()
                current_value = float(tempdfcat.loc[:, feature].iloc[0])
                tempdfcat.loc[:, feature] = 0.0 if current_value == 1.0 else 1.0
                pred = model.predict(tempdfcat)
                pred_label = np.asarray(pred).reshape(-1)
                if pred_label.size > 0 and int(pred_label[0]) == int(outcome):
                    cfdfout = pd.concat([cfdfout, tempdfcat], ignore_index=True, axis=0)
        return cfdfout

    # Double-Feature
    def regressionModel(self, df, f_independent, f_dependent):
        """
        :param df: dataframe of data
        :param f_independent: training space
        :param f_dependent: feature whose value to predict
        :return:
        """
        X = np.array(df.loc[:, df.columns != f_dependent])
        y = np.array(df.loc[:, df.columns == f_dependent])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
        linear_reg = LinearRegression()
        linear_reg.fit(X_train, y_train.ravel())
        y_pred = linear_reg.predict(X_test)
        from sklearn.metrics import mean_squared_error
        import math
        mse = mean_squared_error(y_test, y_pred)
        msse = math.sqrt(mean_squared_error(y_test, y_pred))
        return linear_reg, mse, msse

    def catclassifyModel(self, df, f_independent, f_dependent):
        """
        :param df:
        :param f_independent:
        :param f_dependent:
        :return:
        """
        X = np.array(df.loc[:, df.columns != f_dependent])
        y = np.array(df.loc[:, df.columns == f_dependent])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
        log_reg = LogisticRegression(solver='lbfgs', max_iter=1000)
        log_reg.fit(X_train, y_train.ravel())
        y_pred = log_reg.predict(X_test)
        ba = balanced_accuracy_score(y_test, y_pred)
        return log_reg, ba

    def _candidate_grid(self, lower, upper, feature, step=None, max_random=20):
        lower = float(lower)
        upper = float(upper)
        if lower > upper:
            lower, upper = upper, lower
        step = {} if step is None else step
        step_size = float(step.get(feature, 1.0))
        if step_size <= 0:
            step_size = 1.0
        if float(step_size).is_integer() and float(lower).is_integer() and float(upper).is_integer():
            return [item for item in range(int(lower), int(upper) + 1, int(step_size))]
        values = list(np.round(np.arange(lower, upper + step_size, step_size), 6))
        return [float(v) for v in values if lower - self.atol <= float(v) <= upper + self.atol]

    def _feature_interval_contains(self, row, feature, intervals):
        if intervals is None or feature not in intervals or feature not in row.columns:
            return True
        lower, upper = intervals[feature]
        try:
            value = float(row[feature].iloc[0])
            return value >= float(lower) - self.atol and value <= float(upper) + self.atol
        except Exception:
            return row[feature].iloc[0] in (lower, upper)

    def _predict_label(self, model, row, order=None):
        pred_input = row
        if order is not None and all(col in row.columns for col in order):
            pred_input = row.loc[:, list(order)]
        pred = np.asarray(model.predict(pred_input)).reshape(-1)
        return int(pred[0]) if pred.size > 0 else None

    def _strict_candidates(
        self,
        candidates,
        test_instance,
        features,
        changeable_features,
        protected_features,
        model,
        desired_outcome,
        uf,
        order=None,
        intervals=None,
        plausibility_data=None,
        require_plausible=True,
    ):
        if not isinstance(candidates, pd.DataFrame) or candidates.empty:
            return pd.DataFrame(columns=list(order) if order is not None else [])

        features = list(features)
        changeable_features = set(features if changeable_features is None else changeable_features)
        protected_features = set([] if protected_features is None else protected_features)
        rows = []
        factual = test_instance.loc[:, features].reset_index(drop=True)
        for pos in range(len(candidates)):
            row = candidates[pos:pos + 1].copy()
            if not all(col in row.columns for col in features):
                continue
            row = row.loc[:, features].reset_index(drop=True)
            changed = self._changed_features(factual, row, features)
            if len(changed) == 0:
                continue
            if any(feature not in changeable_features for feature in changed):
                continue
            if any(feature in protected_features for feature in changed):
                continue
            if not self._within_user_feedback(factual, row, 0, changed, uf):
                continue
            if any(not self._feature_interval_contains(row, feature, intervals) for feature in changed):
                continue
            try:
                if self._predict_label(model, row, order) != int(desired_outcome):
                    continue
            except Exception:
                continue
            rows.append(row)
        if not rows:
            return pd.DataFrame(columns=list(order) if order is not None else features)

        out = pd.concat(rows, ignore_index=True, axis=0)
        if require_plausible and plausibility_data is not None and len(plausibility_data) != 0:
            try:
                X = plausibility_data.loc[:, features]
                scaler = StandardScaler().fit(X)
                X_train = np.vstack([factual.values.reshape(1, -1), X.values])
                nX_train = scaler.transform(X_train)
                ncf_list = scaler.transform(out.loc[:, features].values)
                clf = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True)
                clf.fit(nX_train)
                labels = np.asarray(clf.predict(ncf_list)).reshape(-1)
                out = out.iloc[np.where(labels == 1)[0]].reset_index(drop=True)
            except Exception:
                return pd.DataFrame(columns=list(order) if order is not None else features)
        return out.reset_index(drop=True)

    def _coerce_prediction_value(self, value, categorical=False):
        arr = np.asarray(value).reshape(-1)
        if arr.size == 0:
            return None
        if categorical:
            return float(int(arr[0]))
        return float(arr[0])

    def _append_if_valid_candidate(self, store, candidate, order):
        if all(col in candidate.columns for col in order):
            candidate = candidate.loc[:, list(order)].copy()
        return pd.concat([store, candidate], ignore_index=True, axis=0, sort=False)

    def Double_F(
        self,
        df,
        test_instance,
        protected_features,
        feature_pairs,
        u_cat_f_list,
        numf,
        user_term_intervals,
        features,
        model,
        desired_outcome,
        order,
        k,
        step=None,
        changeable_features=None,
        uf=None,
        plausibility_data=None,
    ):
        """
        :param df: dataframe
        :param test_instance:
        :param protected_features:
        :param feature_pairs:
        :param u_cat_f_list:
        :param numf:
        :param user_term_intervals:
        :param features:
        :param model:
        :param desired_outcome:
        :return:
        """
        pd.set_option('display.max_columns', None)
        cfdf = pd.DataFrame()
        two_feature_explore = pd.DataFrame()
        protected = set([] if protected_features is None else protected_features)
        mutable = set(user_term_intervals.keys() if changeable_features is None else changeable_features)
        uf = {} if uf is None else uf
        for f1, f2 in feature_pairs:
            if f1 not in mutable or f2 not in mutable:
                continue
            if f1 in protected or f2 in protected:
                continue
            if f1 not in user_term_intervals or f2 not in user_term_intervals:
                continue

            candidates = pd.DataFrame()
            if f1 in numf:
                f1_space = self._candidate_grid(user_term_intervals[f1][0], user_term_intervals[f1][1], f1, step)
            else:
                f1_space = [user_term_intervals[f1][1]]

            predictor = None
            if f2 in numf:
                predictor, _mse, _rmse = self.regressionModel(df, f1, f2)
            elif f2 in u_cat_f_list:
                predictor, _ba = self.catclassifyModel(df, f1, f2)

            for f1_value in f1_space:
                candidate = test_instance.copy()
                candidate.loc[:, f1] = f1_value
                if predictor is None:
                    candidate.loc[:, f2] = user_term_intervals[f2][1]
                else:
                    pred_input = candidate.loc[:, candidate.columns != f2]
                    f2_value = self._coerce_prediction_value(predictor.predict(pred_input.values), categorical=f2 in u_cat_f_list)
                    if f2_value is None:
                        continue
                    candidate.loc[:, f2] = f2_value
                candidate = candidate.loc[:, list(order)]
                two_feature_explore = self._append_if_valid_candidate(two_feature_explore, candidate, order)
                if not self._feature_interval_contains(candidate, f2, user_term_intervals):
                    continue
                candidates = self._append_if_valid_candidate(candidates, candidate, order)

            strict = self._strict_candidates(
                candidates,
                test_instance,
                order,
                mutable,
                protected,
                model,
                desired_outcome,
                uf,
                order=order,
                intervals=user_term_intervals,
                plausibility_data=plausibility_data,
                require_plausible=True,
            )
            cfdf = pd.concat([cfdf, strict], ignore_index=True, axis=0, sort=False)
            if len(cfdf) >= k:
                break
        return cfdf.iloc[:k].reset_index(drop=True), two_feature_explore

    def Triple_F(
        self,
        df,
        test_instance,
        protected_features,
        feature_pairs,
        u_cat_f_list,
        numf,
        user_term_intervals,
        features_2change,
        model,
        desired_outcome,
        order,
        k,
        step=None,
        uf=None,
        plausibility_data=None,
    ):
        """
        :param df:
        :param test_instance:
        :param protected_features:
        :param feature_pairs:
        :param u_cat_f_list:
        :param numf:
        :param user_term_intervals:
        :param features_2change:
        :param model:
        :param desired_outcome:
        :return:
        """
        cfdf = pd.DataFrame()
        three_feature_explore = pd.DataFrame()
        protected = set([] if protected_features is None else protected_features)
        mutable = set(features_2change)
        uf = {} if uf is None else uf
        for f1, f2 in feature_pairs:
            if f1 not in mutable or f2 not in mutable:
                continue
            if f1 in protected or f2 in protected:
                continue
            if f1 not in user_term_intervals or f2 not in user_term_intervals:
                continue

            if f1 in numf:
                f1_space = self._candidate_grid(user_term_intervals[f1][0], user_term_intervals[f1][1], f1, step, max_random=8)
            else:
                f1_space = [user_term_intervals[f1][1]]

            f2_predictor = None
            if f2 in numf:
                f2_predictor, _mse, _rmse = self.regressionModel(df, f1, f2)
            elif f2 in u_cat_f_list:
                f2_predictor, _ba = self.catclassifyModel(df, f1, f2)

            f3_predictors = {}
            candidates = pd.DataFrame()
            for f1_value in f1_space:
                base = test_instance.copy()
                base.loc[:, f1] = f1_value
                if f2_predictor is None:
                    base.loc[:, f2] = user_term_intervals[f2][1]
                else:
                    f2_input = base.loc[:, base.columns != f2]
                    f2_value = self._coerce_prediction_value(f2_predictor.predict(f2_input.values), categorical=f2 in u_cat_f_list)
                    if f2_value is None:
                        continue
                    base.loc[:, f2] = f2_value
                if not self._feature_interval_contains(base, f2, user_term_intervals):
                    continue

                for f3 in features_2change:
                    if f3 == f1 or f3 == f2:
                        continue
                    if f3 in protected or f3 not in mutable or f3 not in user_term_intervals:
                        continue
                    candidate = base.copy()
                    if f3 in numf:
                        if f3 not in f3_predictors:
                            f3_predictors[f3] = self.regressionModel(df, f1, f3)[0]
                        f3_predictor = f3_predictors[f3]
                        f3_input = candidate.loc[:, candidate.columns != f3]
                        f3_value = self._coerce_prediction_value(f3_predictor.predict(f3_input.values), categorical=False)
                    elif f3 in u_cat_f_list:
                        if f3 not in f3_predictors:
                            f3_predictors[f3] = self.catclassifyModel(df, f1, f3)[0]
                        f3_predictor = f3_predictors[f3]
                        f3_input = candidate.loc[:, candidate.columns != f3]
                        f3_value = self._coerce_prediction_value(f3_predictor.predict(f3_input.values), categorical=True)
                    else:
                        continue
                    if f3_value is None:
                        continue
                    candidate.loc[:, f3] = f3_value
                    candidate = candidate.loc[:, list(order)]
                    three_feature_explore = self._append_if_valid_candidate(three_feature_explore, candidate, order)
                    if not self._feature_interval_contains(candidate, f3, user_term_intervals):
                        continue
                    candidates = self._append_if_valid_candidate(candidates, candidate, order)
            strict = self._strict_candidates(
                candidates,
                test_instance,
                order,
                mutable,
                protected,
                model,
                desired_outcome,
                uf,
                order=order,
                intervals=user_term_intervals,
                plausibility_data=plausibility_data,
                require_plausible=True,
            )
            cfdf = pd.concat([cfdf, strict], ignore_index=True, axis=0, sort=False)
            if len(cfdf) >= k:
                return cfdf.iloc[:k].reset_index(drop=True), three_feature_explore
        return cfdf.iloc[:k].reset_index(drop=True), three_feature_explore

    def mad_cityblock(self, u, v, mad):
        u = _validate_vector(u)
        v = _validate_vector(v)
        l1_diff = abs(u - v)
        l1_diff_mad = l1_diff / mad
        return l1_diff_mad.sum()

    # In following some functions are customised according to our needs, the orginal source of these functions belongs to:
    #"Guidotti, R. Counterfactual explanations and how to find them: literature review and benchmarking. Data Min Knowl Disc (2022). https://doi.org/10.1007/s10618-022-00831-6
    
    # Begin> 3rd party adapted ///////
    
    def continuous_distance(self, x, cf_list, continuous_features, metric='euclidean', X=None, agg=None):
        """
        :param x:
        :param cf_list:
        :param continuous_features:
        :param metric:
        :param X:
        :param agg:
        :return:
        """
        if self.contprox_metric != '': 
            metric = self.contprox_metric
            
        if metric == 'mad':
            if X is None:
                dist = cdist(x.loc[:, continuous_features], cf_list.loc[:, continuous_features], metric='euclidean')
            else:
                mad = median_absolute_deviation(X[:, continuous_features], axis=0)
                mad = np.array([v if v != 0 else 1.0 for v in mad])

                def _mad_cityblock(u, v):
                    return self.mad_cityblock(u, v, mad)

                dist = cdist(x.reshape(1, -1)[:, continuous_features], cf_list[:, continuous_features], metric=_mad_cityblock)
        else:
            dist = cdist(x.loc[:, continuous_features], cf_list.loc[:, continuous_features], metric=metric)
        if agg is None or agg == 'mean':
            if len(dist) != 0:
                return np.mean(dist)
            else:
                return 0

        if agg == 'max':
            return np.max(dist)

        if agg == 'min':
            return np.min(dist)

    def categorical_distance(self, x, cf_list, categorical_features, metric='jaccard', agg=None):
        """
        :param x:
        :param cf_list:
        :param categorical_features:
        :param metric:
        :param agg:
        :return:
        """
        dist = cdist(x.loc[:, categorical_features], cf_list.loc[:, categorical_features], metric=metric)

        if agg is None or agg == 'mean':
            return np.mean(dist)

        if agg == 'max':
            return np.max(dist)

        if agg == 'min':
            return np.min(dist)
    
    def distance_e2j(self, x, cf_list, continuous_features, categorical_features, ratio_cont=None, agg=None):
        """
        :param x:
        :param cf_list:
        :param continuous_features:
        :param categorical_features:
        :param ratio_cont:
        :param agg:
        :return:
        """
        nbr_features = cf_list.shape[1]
        dist_cont = continuous_distance(x, cf_list, continuous_features, metric='euclidean', X=None, agg=agg)
        dist_cate = categorical_distance(x, cf_list, categorical_features, metric='jaccard', agg=agg)
        if ratio_cont is None:
            ratio_continuous = len(continuous_features) / nbr_features
            ratio_categorical = len(categorical_features) / nbr_features
        else:
            ratio_continuous = ratio_cont
            ratio_categorical = 1.0 - ratio_cont
        dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
        return dist
        
    def lofn(self, x, cf_list, X, scaler, return_details=False, apply_scaler=True):
        """
        :param x: test instance
        :param cf_list: list of counterfactuals or single counterfactual
        :param X: feature data space
        :param scaler: scaler model
        :return: 1, 0
        """
        details = {
            "label": 0,
            "decision_function": np.nan,
            "score_samples": np.nan,
        }
        if x.empty != True and cf_list.empty != True:
            X_base = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
            X_train = np.vstack([x.values.reshape(1, -1), X_base])
            cf_base = cf_list.values if isinstance(cf_list, pd.DataFrame) else np.asarray(cf_list)
            if apply_scaler and scaler is not None:
                nX_train = scaler.transform(X_train)
                ncf_list = scaler.transform(cf_base)
            else:
                nX_train = np.asarray(X_train, dtype=float)
                ncf_list = np.asarray(cf_base, dtype=float)

            clf = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True) 
            clf.fit(nX_train)
            lof_values = clf.predict(ncf_list)
            try:
                decision_scores = clf.decision_function(ncf_list)
                decision_value = float(np.asarray(decision_scores).reshape(-1)[0])
            except Exception:
                decision_value = np.nan
            try:
                score_samples = clf.score_samples(ncf_list)
                score_value = float(np.asarray(score_samples).reshape(-1)[0])
            except Exception:
                score_value = np.nan
            label = int(np.asarray(lof_values).reshape(-1)[0]) if np.asarray(lof_values).size > 0 else 0
            details = {
                "label": label,
                "decision_function": decision_value,
                "score_samples": score_value,
            }
        else:
            lof_values = 0
        if return_details:
            if x.empty == True or cf_list.empty == True:
                return details
            return details
        return lof_values

    def implausibility(
        self,
        cfdf,
        Xtest,
        Xtrain,
        K,
        idx,
        method_name="unknown",
        return_details=False,
        use_standard_scaler=True,
        lof_space_label="raw_lof",
    ):
        from sklearn.preprocessing import StandardScaler
        """
        :param path_to_cfdf:
        :param Xtest:
        :param Xtrain:
        :param K: The total number of test instances using for the evaluation
        :return:
        """
        # Implausibility - local outlier factor - lof
        tempone = dict()
        scaler = None
        if use_standard_scaler:
            scaler = StandardScaler() # check verify the scaler
            scaler = scaler.fit(Xtrain)
        result = 0
        passed_idx = []
        pair_details = []
        trace_enabled = False
        trace_this_method = False
        trace_max = 0
        if isinstance(getattr(self, "debug_ctx", None), dict):
            trace_enabled = bool(self.debug_ctx.get("enabled", False))
            trace_this_method = str(method_name).upper() == "UFCE1"
            trace_max = int(self.debug_ctx.get("trace_max_instances", 0))
            if trace_enabled and trace_this_method:
                print(
                    "[DBG][UFCE1][LOF] "
                    f"space={lof_space_label}"
                )
        if len(Xtest) != 0 and len(cfdf) != 0:
            traced_inliers = 0
            traced_outliers = 0
            valid_pairs = 0
            for t in range(len(cfdf)):
                x_row = Xtest[t:t + 1]
                cf_row = cfdf[t:t + 1]
                is_valid = True
                if x_row.empty or cf_row.empty:
                    is_valid = False
                else:
                    x_vals = pd.to_numeric(x_row.iloc[0], errors="coerce")
                    cf_vals = pd.to_numeric(cf_row.iloc[0], errors="coerce")
                    if bool(x_vals.isna().any()) or bool(cf_vals.isna().any()):
                        is_valid = False
                if not is_valid:
                    pair_details.append(
                        {
                            "pair_pos": int(t),
                            "valid": False,
                            "label": 0,
                            "decision_function": np.nan,
                            "score_samples": np.nan,
                        }
                    )
                    continue
                valid_pairs += 1
                details = self.lofn(
                    x_row,
                    cf_row,
                    Xtrain[:],
                    scaler,
                    return_details=True,
                    apply_scaler=bool(use_standard_scaler),
                )
                res = int(details.get("label", 0))
                pair_details.append(
                    {
                        "pair_pos": int(t),
                        "valid": True,
                        "label": int(res),
                        "decision_function": float(details.get("decision_function", np.nan)),
                        "score_samples": float(details.get("score_samples", np.nan)),
                    }
                )
                if trace_enabled and trace_this_method and t < trace_max:
                    label_text = "inlier" if res == 1 else "outlier"
                    print(
                        "[DBG][UFCE1][LOF] "
                        f"pair_pos={t}, label={label_text}, "
                        f"decision_function={details.get('decision_function', np.nan)}, "
                        f"score_samples={details.get('score_samples', np.nan)}"
                    )
                    if res == 1:
                        traced_inliers += 1
                    else:
                        traced_outliers += 1
                if res == 1:
                    result += 1
                    passed_idx.append(int(t))
            if trace_enabled and trace_this_method:
                print(
                    "[DBG][UFCE1][LOF] "
                    f"traced_summary=inliers:{traced_inliers}, outliers:{traced_outliers}"
                )
            if return_details:
                return {
                    "count": int(result),
                    "passed_idx": passed_idx,
                    "pair_details": pair_details,
                    "n_pairs_input": int(len(cfdf)),
                    "n_pairs_valid": int(valid_pairs),
                    "lof_space_label": str(lof_space_label),
                    "use_standard_scaler": bool(use_standard_scaler),
                }
        else:
            if return_details:
                return {
                    "count": 0,
                    "passed_idx": [],
                    "pair_details": [],
                    "n_pairs_input": int(len(cfdf)),
                    "n_pairs_valid": 0,
                    "lof_space_label": str(lof_space_label),
                    "use_standard_scaler": bool(use_standard_scaler),
                }
            return result
        return result
        
    # Sparsity
    #cf feature changes and avg change
    def nbr_changes_per_cfn(self, x, cf_list):
        """
        Calculates number of feature changes using epsilon tolerance to 
        ignore float precision noise from inverse scaling. [cite: 1, 2]
        """
        features = list(x.columns)
        nbr_changes = 0
        for j in features:
            # Extract scalar values safely
            val_factual = float(x[j].iloc[0])
            val_cf = float(cf_list[j].iloc[0])
            
            # Use np.isclose with a standard absolute tolerance (atol)
            if not np.isclose(val_factual, val_cf, atol=1e-5):
                nbr_changes += 1
        return nbr_changes

    def avg_nbr_changes_per_cfn(self, x, cf_list, continuous_features):
        return np.mean(self.nbr_changes_per_cfn(x, cf_list, continuous_features))
     
    def sparsity_count(self, cfdf, Xtest, cont_features, idx):
        """
        :param path_to_cfdf:
        :param K:
        :param Xtest:
        :param cont_features:
        :return:
        """
        ## SPARSITY : nbr of changes per CF
        result = 0
        tempone = dict()
        for t in range(len(cfdf)):
            res = self.nbr_changes_per_cfn(Xtest[t:t + 1], cfdf[t:t+1])
            result += res
            tempone[t] = res
        if result != 0:
            return tempone, result / len(cfdf)
        else:
            return tempone, result

    def nbr_actionable_cfn(self, x, cf_list, features, f2change):
        """
        Identifies actionable changes using epsilon tolerance.
        """
        f_list = []
        nbr_actionable = 0
        for j in features:
            val_factual = float(x[j].iloc[0])
            val_cf = float(cf_list[j].iloc[0])
            
            # Apply floating-point fix here as well [cite: 1, 2]
            if not np.isclose(val_factual, val_cf, atol=self.atol) and j in f2change:
                nbr_actionable += 1
                f_list.append(j)
        return nbr_actionable, f_list

    def changes_per_cf(self, x, cf):
        """
        Helper for feasibility checks using epsilon tolerance.
        """
        features = list(x.columns)
        nbr_changes = 0
        for j in features:
            val_factual = float(x[j].iloc[0])
            val_cf = float(cf[j].iloc[0])
            
            if not np.isclose(val_factual, val_cf, atol=self.atol):
                nbr_changes += 1
        return nbr_changes

    def _changed_features(self, x, cf, features):
        changed = []
        for feature in features:
            if feature not in x.columns or feature not in cf.columns:
                continue
            left = x[feature].iloc[0]
            right = cf[feature].iloc[0]
            try:
                is_changed = not np.isclose(float(left), float(right), atol=self.atol, rtol=0.0)
            except Exception:
                is_changed = str(left) != str(right)
            if is_changed:
                changed.append(feature)
        return changed

    def _feedback_bounds(self, current_value, feedback_spec):
        try:
            current = float(current_value)
        except Exception:
            current = current_value

        if isinstance(feedback_spec, (list, tuple, np.ndarray)) and len(feedback_spec) >= 2:
            lower = float(feedback_spec[0])
            upper = float(feedback_spec[1])
            return (min(lower, upper), max(lower, upper))

        try:
            delta = float(feedback_spec)
            cur = float(current)
        except Exception:
            return (current, current)

        if cur in (0.0, 1.0) and abs(delta) == 1.0:
            return (0.0, 1.0)
        candidate = cur + delta
        return (min(cur, candidate), max(cur, candidate))

    def _within_user_feedback(self, X_test, cfdf, row_pos, changed_features, uf):
        for feature in changed_features:
            if feature not in uf:
                return False
            lower, upper = self._feedback_bounds(X_test.at[row_pos, feature], uf[feature])
            try:
                value = float(cfdf.at[row_pos, feature])
            except Exception:
                value = cfdf.at[row_pos, feature]
            try:
                if value < lower - self.atol or value > upper + self.atol:
                    return False
            except Exception:
                if value != lower and value != upper:
                    return False
        return True

    def _actionability_stats(self, x, cf, features, changeable_features):
        changed = self._changed_features(x, cf, features)
        actionable = [feature for feature in changed if feature in changeable_features]
        ratio = len(actionable) / len(changed) if len(changed) != 0 else 0.0
        return changed, actionable, ratio

    def actionability(self, cfdf, X_test, features, changeable_features, idx, uf, method):
        """
        :param cfdf: counterfactuals (s)
        :param K: no. or length of test set
        :param Xtest: test set
        :param features:
        :param changeable_features:
        :return:
        """
        count = 0
        flag = 0
        idx1 = []
        cfs = pd.DataFrame()
        temp = dict()
        X_test.reset_index(drop = True, inplace = True)
        cfdf.reset_index(drop=True, inplace=True)
        for x in range(len(cfdf)):
            if method =="other":
                changed, f_list, ratio = self._actionability_stats(
                    X_test[x:x + 1],
                    cfdf[x:x + 1],
                    features,
                    changeable_features,
                )
                if (
                    len(changed) != 0
                    and ratio >= self.actionability_threshold
                    and self._within_user_feedback(X_test, cfdf, x, f_list, uf)
                ):
                    cfs = pd.concat([cfs, cfdf[x:x + 1]], ignore_index=True, axis=0)
                    flag = 1
                    idx1.append(x)
                    temp[x] = ratio
                        
            else:
                count, f_list = self.nbr_actionable_cfn(X_test[x:x + 1], cfdf[x:x + 1], features, changeable_features)
                cfs = pd.concat([cfs, cfdf[x:x + 1]], ignore_index=True, axis=0)
                flag = 1
                idx1.append(x)
                temp[x] = count
        return cfs, flag, idx1, temp
    
    # End> 3rd party adapted ///////
    
    def diverse_CFs(self, test, nn_valid, uf, c_f):
        """
        test: test instance
        nn_valid: valid nearest neighbors (df)
        uf: user feedback (dict)
        c_f: changeable features (dict)
        :return cfs : diverse counterfactual(s)
        """
        cfs = pd.DataFrame()
        cfs = nn_valid
        for i in range(len(c_f)):
            cfs = cfs[cfs[c_f[i]].between(test[c_f[i]].values[0], (test[c_f[i]].values + uf[c_f[i]])[0])]
        return cfs

    # Begin> 3rd party adapted /////// 
    def count_diversity(self, cf_list, features, nbr_features, continuous_features):
        """
        :param cf_list:
        :param features:
        :param nbr_features:
        :param continuous_features:
        :return:
        """
        nbr_cf = cf_list.shape[0]
        nbr_changes = 0
        for i in range(nbr_cf):
            for j in range(i + 1, nbr_cf):
                for k in features:
                    if cf_list[i:i + 1][k].values != cf_list[j:j + 1][k].values:
                        nbr_changes += 1 if j in continuous_features else 0.5
        return nbr_changes / (nbr_cf * nbr_cf * nbr_features) if nbr_changes != 0 else 0.0
    # End> 3rd party adapted ///////
    

    def feasibility(
        self,
        X_test,
        cffile,
        X_train,
        features,
        changeable_features,
        model,
        desired_outcome,
        uf,
        idx,
        method,
        return_details=False,
        use_standard_scaler=True,
        lof_space_label="raw_lof",
    ):
        X_test.reset_index(drop=True, inplace=True)
        X_train.reset_index(drop = True, inplace = True)
        cffile.reset_index(drop=True, inplace=True)
        cflist = cffile
        scaler = None
        if use_standard_scaler:
            scaler = StandardScaler()  # check verify the scaler
            scaler = scaler.fit(X_train[:])
        feasible = 0
        feas = 0
        temp = pd.DataFrame()
        passed_idx = []
        pair_details = []
        valid_pairs = 0
        # temptest = pd.DataFrame()
        if cflist.empty != True:
            for x in range(len(cflist)):
                x_row = X_test[x:x + 1]
                cf_row = cflist[x:x + 1]
                is_valid = True
                if x_row.empty or cf_row.empty:
                    is_valid = False
                else:
                    x_vals = pd.to_numeric(x_row.iloc[0], errors="coerce")
                    cf_vals = pd.to_numeric(cf_row.iloc[0], errors="coerce")
                    if bool(x_vals.isna().any()) or bool(cf_vals.isna().any()):
                        is_valid = False
                if not is_valid:
                    pair_details.append(
                        {
                            "pair_pos": int(x),
                            "valid": False,
                            "plausible": False,
                            "actionable_for_feas": False,
                            "passed": False,
                            "reason": "invalid_numeric",
                        }
                    )
                    continue
                try:
                    pred_input = cf_row.loc[:, features] if all(col in cf_row.columns for col in features) else cf_row
                    pred = np.asarray(model.predict(pred_input)).reshape(-1)
                    is_valid = pred.size > 0 and int(pred[0]) == int(desired_outcome)
                except Exception:
                    is_valid = False
                if not is_valid:
                    pair_details.append(
                        {
                            "pair_pos": int(x),
                            "valid": False,
                            "plausible": False,
                            "actionable_for_feas": False,
                            "passed": False,
                            "reason": "desired_outcome_fail",
                        }
                    )
                    continue
                valid_pairs += 1
                if method == "other":
                    plaus_details = self.lofn(
                        x_row,
                        cf_row,
                        X_train[:],
                        scaler,
                        return_details=True,
                        apply_scaler=bool(use_standard_scaler),
                    )
                    plaus = int(plaus_details.get("label", 0))
                    if plaus == 1:
                        changed, f_list, ratio = self._actionability_stats(
                            x_row,
                            cf_row,
                            features,
                            changeable_features,
                        )
                        if (
                            len(changed) != 0
                            and ratio >= self.actionability_threshold
                            and self._within_user_feedback(X_test, cflist, x, f_list, uf)
                        ):
                            feas += 1
                            temp = pd.concat([temp, cf_row], axis=0, ignore_index=True)
                            passed_idx.append(int(x))
                            pair_details.append(
                                {
                                    "pair_pos": int(x),
                                    "valid": True,
                                    "plausible": True,
                                    "actionable_for_feas": True,
                                    "passed": True,
                                    "reason": "passes_all",
                                    "actionability_ratio": float(ratio),
                                    "actionability_threshold": float(self.actionability_threshold),
                                    "lof_space_label": str(lof_space_label),
                                    "use_standard_scaler": bool(use_standard_scaler),
                                }
                            )
                            continue
                        pair_details.append(
                            {
                                "pair_pos": int(x),
                                "valid": True,
                                "plausible": True,
                                "actionable_for_feas": False,
                                "passed": False,
                                "reason": "actionability_threshold_fail",
                                "actionability_ratio": float(ratio),
                                "actionability_threshold": float(self.actionability_threshold),
                            }
                        )
                        continue
                    pair_details.append(
                        {
                            "pair_pos": int(x),
                            "valid": True,
                            "plausible": False,
                            "actionable_for_feas": False,
                            "passed": False,
                            "reason": "lof_outlier",
                        }
                    )
                    continue
                            
                else:    
                    plaus_details = self.lofn(
                        x_row,
                        cf_row,
                        X_train[:],
                        scaler,
                        return_details=True,
                        apply_scaler=bool(use_standard_scaler),
                    )
                    plaus = int(plaus_details.get("label", 0)) #make it 1000 in other cases, except movie
                    if plaus == 1:
                        feas += 1
                        temp = pd.concat([temp, cf_row], axis=0, ignore_index=True)
                        passed_idx.append(int(x))
                        pair_details.append(
                            {
                                "pair_pos": int(x),
                                "valid": True,
                                "plausible": True,
                                "actionable_for_feas": True,
                                "passed": True,
                                "reason": "passes_all",
                            }
                        )
                    else:
                        pair_details.append(
                            {
                                "pair_pos": int(x),
                                "valid": True,
                                "plausible": False,
                                "actionable_for_feas": False,
                                "passed": False,
                                "reason": "lof_outlier",
                            }
                        )

            if feas != 0:
                if return_details:
                    return feas, temp, {
                        "count": int(feas),
                        "passed_idx": passed_idx,
                        "pair_details": pair_details,
                        "n_pairs_input": int(len(cflist)),
                        "n_pairs_valid": int(valid_pairs),
                        "lof_space_label": str(lof_space_label),
                        "use_standard_scaler": bool(use_standard_scaler),
                    }
                return feas, temp
            else:
                feas = feasible  
        else:
            feas = feasible
        if return_details:
            return feas, temp, {
                "count": int(feas),
                "passed_idx": passed_idx,
                "pair_details": pair_details,
                "n_pairs_input": int(len(cflist)),
                "n_pairs_valid": int(valid_pairs),
                "lof_space_label": str(lof_space_label),
                "use_standard_scaler": bool(use_standard_scaler),
            }
        return feas, temp

    
    def get_highly_correlated(self, df, features, threshold=0.5):
        """
        :param df:
        :param features:
        :param threshold:
        :return:
        """
        corr_df = df[features].corr()  # get correlations
        correlated_features = np.where(np.abs(corr_df) > threshold)
        correlated_features = [(corr_df.iloc[x, y], x, y) for x, y in zip(*correlated_features) if x != y and x < y]  # avoid duplication
        s_corr_list = sorted(correlated_features, key=lambda x: -abs(x[0]))  # sort by correlation value
        corr_dict = dict()
        if s_corr_list == []:
            print("There are no highly correlated features with correlation:", threshold)
        else:
            for v, i, j in s_corr_list:
                cols = df[features].columns
                corr_dict[corr_df.index[i]] = corr_df.columns[j]

        keys_list = corr_dict.keys()
        feature_list = []
        features_to_use = []
        for key in keys_list:
            feature_list.append(key)
            feature_list.append(corr_dict[key])
        features_to_use.append(feature_list[0])
        features_to_use.append(feature_list[1])
        return corr_dict, features_to_use


    def candidate_counterfactuals_df(self, df1, df2, df3, path):
        """
        :param df1:
        :param df2:
        :param df3:
        :param path:
        :return:
        """
        #path = 'C:\\Users\\~\\~\\data\\'
        f = 'Final_merged_df_with_all_combinations'
        df_2_return = pd.DataFrame()
        df_2_return = pd.concat([df1, df2], ignore_index=True, axis=0)
        df_2_return = pd.concat([df_2_return, df3], ignore_index=True, axis=0)
        df_2_return = df_2_return.transform(np.sort)
        df_2_return.to_csv(path + '' + f + '' + '.csv')
        return df_2_return


    def train_Outliers_isolation_model(self, df):
        """
        :param df:
        :return:
        """
        from sklearn.ensemble import IsolationForest
        df1 = df.copy()
        outlier_model = IsolationForest(n_estimators=100, max_samples=1000, contamination=.05, max_features=df1.shape[1])
        outlier_model.fit(df1)
        outliers_predicted = outlier_model.predict(df1)

        return outlier_model

    def get_Outlier_isolation_prediction(self, model, cf_instance):
        """
        :param model:
        :param cf_instance:
        :return:
        """
        predicted = model.predict(cf_instance)
        print(predicted)
