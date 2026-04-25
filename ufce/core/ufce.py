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
        min_actionable_other=3,
        min_actionable_feasible_other=2,
        atol=1e-5,
    ):
        # self.dataset = 'bank'
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.contprox_metric = contprox_metric
        self.min_actionable_other = int(min_actionable_other)
        self.min_actionable_feasible_other = int(min_actionable_feasible_other)
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
        f_start = 0
        f_end = 0
        for f in feat2change:
            if f in uf.keys():
                f_start = test[f].values
                max_limit = test[f].values + uf[f]
                if isinstance(uf[f], float):
                    space = np.arange(test[f].values, max_limit, 0.1)
                    if len(space) != 0:
                        f_end = space[-1] #random.choice(space)  # test[f].values + uf[f]
                    else:
                        f_end = test[f].values
                else:
                    space = np.arange(test[f].values, max_limit, 1)
                    if len(space) != 0:
                        f_end = space[-1] #random.choice(space)  # test[f].values + uf[f]
                    else:
                        f_end = test[f].values
                if f_end >= nn[f].max():
                    intervals[f] = [f_start[0], nn[f].max()]
                else:
                    intervals[f] = [f_start[0], f_end]
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
            f1_start, f1_end, f2_start, f2_end = 0, 0, 0, 0
            f1_start = test[f1].values
            ###
            max_limit1 = f1_start + uf[f1] 
            if isinstance(uf[f1], float):
                space1 = np.arange(f1_start, max_limit1, 0.1)
                if len(space1) != 0:
                    f1_end = space1[-1] #random.choice(space1)
                else:
                    f1_end = f1_start
            else:
                space1 = np.arange(f1_start, max_limit1, 1)
                if len(space1) != 0:
                    f1_end = space1[-1] #random.choice(space1)
                else:
                    f1_end = f1_start
            ###
            f2_start = test[f2].values
            ##
            max_limit2 = f2_start + uf[f2] 
            if isinstance(uf[f2], float):
                space2 = np.arange(f2_start, max_limit2, 0.1)
                if len(space2) != 0:
                    f2_end = space2[-1] #random.choice(space2)  # test[f].values + uf[f]
                else:
                    f2_end = f2_start
            else:
                space2 = np.arange(f2_start, max_limit2, 1)
                if len(space2) != 0:
                    f2_end = space2[-1] #random.choice(space2)  # test[f].values + uf[f]
                else:
                    f2_end = f2_start

            if f1_end >= nn[f1].max():
                faithful_interval[f1] = [test[f1].values[0], nn[f1].max()]
            else:
                faithful_interval[f1] = [test[f1].values[0], f1_end]
            if f2_end >= nn[f2].max():
                faithful_interval[f2] = [test[f2].values[0], nn[f2].max()]
            else:
                faithful_interval[f2] = [test[f2].values[0], f2_end]
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
                    start = interval_term_range[0]
                    end = interval_term_range[1]
                    step_size = step[feature]
                    cfdf = pd.DataFrame()
                    if isinstance(start, int) and isinstance(end, int):
                        f1_space = [item for item in range(start, end + 1)]
                    else:
                        f1_space = sorted(np.round(random.uniform(start, end), 2) for _ in range(20))
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
                        if pred == outcome:
                            cfdf = tempdf1.copy()
                            cfdfout = pd.concat([cfdfout, cfdf], ignore_index=True, axis=0)
                        try:
                            del f1_space[:mid+1]
                        except:
                            pass
            else:
                tempdfcat = test_instance.copy()
                tempdfcat.loc[:, feature] = 1.0 if tempdfcat.loc[:, feature].values else 1.0
                pred = model.predict(tempdfcat)
                if pred == outcome:
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

    def Double_F(self, df, test_instance, protected_features, feature_pairs, u_cat_f_list, numf, user_term_intervals, features, model, desired_outcome, order, k):
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
        count = 0
        cfdf = pd.DataFrame()
        two_feature_explore = pd.DataFrame()
        temptempdf = pd.DataFrame()
        temptempdf = test_instance.copy()
        feature_to_use_list = feature_pairs
        for f in feature_to_use_list:
            f1 = f[0]
            f2 = f[1]
            two_feature_data = pd.DataFrame()
            temptempdf = pd.DataFrame()
            tempdf1 = pd.DataFrame()
            tempdf1 = test_instance.copy()
            if (f1 in numf and f2 in numf) and (f1 not in protected_features and f2 not in protected_features):  # both numerical
                if f1 and f2 in user_term_intervals.keys():
                    interval_term_range1 = user_term_intervals[f1]
                    interval_term_range2 = user_term_intervals[f2]
                    start1 = int(interval_term_range1[0])
                    end1 = int(interval_term_range1[1])
                    start2 = int(interval_term_range2[0])
                    end2 = int(interval_term_range2[1])
                    reg_model, mse, rmse = self.regressionModel(df, f1, f2)
                    if isinstance(start1, int) and isinstance(end1, int):
                        f1_space = [item for item in range(start1, end1 + 1)]
                    else:
                        f1_space = sorted(np.round(random.uniform(start1, end1), 2) for _ in range(20))
                    while len(f1_space) != 0:
                        tempdf1 = test_instance.copy()
                        if len(f1_space) != 0:
                            low = 0
                            high = len(f1_space) - 1
                            mid = (high - low) // 2
                        tempdf1.loc[:, f1] = f1_space[mid]
                        temptempdf = tempdf1.copy()
                        if isinstance(start2, int) and isinstance(end2, int):
                            f2_space = [item for item in range(start2, end2 + 1)]
                        else:
                            f2_space = sorted(np.round(random.uniform(start2, end2), 2) for _ in range(20))

                        while len(f2_space) != 0:
                            if len(f2_space) != 0:
                                low_in = 0
                                high_in = len(f2_space) - 1
                                mid_in = (high_in - low_in) // 2
                            tempdf1.loc[:, f2] = f2_space[mid_in]
                            temptempdf = tempdf1.copy()
                            temptempdf = temptempdf[order]
                            two_feature_explore = pd.concat(
                                [two_feature_explore, temptempdf],
                                ignore_index=True,
                                axis=0,
                                sort=False,
                            )
                            pred = model.predict(temptempdf)
                            pred_label = int(np.asarray(pred).reshape(-1)[0])
                            if pred_label == int(desired_outcome):
                                cfdf = pd.concat([cfdf, temptempdf], ignore_index=True, axis=0, sort=False)
                            if len(cfdf) >= k:
                                break
                            try:
                                del f2_space[:mid_in + 1]
                            except:
                                pass
                        try:
                            del f1_space[:mid+1]
                        except:
                            pass
                
            elif (f1 in u_cat_f_list and f2 in u_cat_f_list) and f1 and f2 not in protected_features:  # both categorical
                if f1 and f2 in user_term_intervals.keys():
                    tempdfcat = test_instance.copy()
                    tempdfcat.loc[:, f1] = user_term_intervals[f1][1] # 0.0 if tempdfcat.loc[:, f1].values else 1.0
                    tempdfcat.loc[:, f2] = user_term_intervals[f2][1] #0.0 if tempdfcat.loc[:, f2].values else 1.0
                    two_feature_explore = pd.concat([two_feature_explore, tempdfcat], ignore_index=True, axis=0)
                    tempdfcat = tempdfcat[order]
                    pred = model.predict(tempdfcat)
                    if pred == desired_outcome:
                        cfdf = pd.concat([cfdf, tempdfcat], ignore_index=True, axis=0)
                    if len(cfdf) == k:
                        break

            elif (f1 in numf and f2 in u_cat_f_list) and f1 and f2 not in protected_features:  # num -> cat (binary classification)
                if f1 and f2 in user_term_intervals.keys():
                    interval_term_range1 = user_term_intervals[f1]
                    interval_term_range2 = user_term_intervals[f2]
                    start1 = int(interval_term_range1[0])
                    end1 = int(interval_term_range1[1])
                    start2 = int(interval_term_range2[0])
                    end2 = int(interval_term_range2[1])
                    log_model, ba = self.catclassifyModel(df, f1, f2)
                    if isinstance(start1, int) and isinstance(end1, int):
                        f1_space = [item for item in range(start1, end1 + 1)]
                    else:
                        f1_space = sorted(np.round(random.uniform(start1, end1), 2) for _ in range(20))
                    while len(f1_space) != 0:
                        tempdf1 = test_instance.copy()
                        low = 0
                        high = len(f1_space) - 1
                        mid = (high - low) // 2
                        tempdf1.loc[:, f1] = f1_space[mid]
                        temptempdf = tempdf1.copy()
                        tempdf1 = tempdf1.loc[:, tempdf1.columns != f2]
                        f2_val = log_model.predict(tempdf1.values)
                        if isinstance(f2, float):
                            temptempdf.loc[:, f2] = f2_val[0]
                        else:
                            temptempdf.loc[:, f2] = float(int(f2_val[0]))
                        if f2_val >= start2 and f2_val <= df[f2].max(): 
                            two_feature_explore = pd.concat([two_feature_explore, temptempdf], ignore_index=True,
                                                                axis=0)
                            temptempdf = temptempdf[order]
                            pred = model.predict(temptempdf)
                            if pred == desired_outcome:  #
                                cfdf = pd.concat([cfdf, temptempdf], ignore_index=True, axis=0)
                            if len(cfdf) == k:
                                break
                        try:
                            del f1_space[:mid+1]
                        except:
                            pass
            elif (f1 in u_cat_f_list and f2 in numf) and (f1 and f2 not in protected_features): # cat and num
                if f1 and f2 in user_term_intervals.keys():
                    temptempdf = tempdf1.copy()
                    tempdf1.loc[:, f1] = user_term_intervals[f1][1] 
                    reg_model, mse, rmse = self.regressionModel(df, f1, f2)
                    tempdf1 = tempdf1.loc[:, tempdf1.columns != f2]
                    f2_val = reg_model.predict(tempdf1.values)
                    if isinstance(f2, float):
                        temptempdf.loc[:, f2] = f2_val[0]
                    else:
                        temptempdf.loc[:, f2] = float(int(f2_val[0]))
                    two_feature_explore = pd.concat([two_feature_explore, temptempdf], ignore_index=True, axis=0,
                                                       sort=False)
                    if int(f2_val[0]) <= df[f2].max():  
                        temptempdf = temptempdf[order]
                        pred = model.predict(temptempdf)
                        if pred == desired_outcome:
                            cfdf = pd.concat([cfdf, temptempdf], ignore_index=True, axis=0)
                        if len(cfdf) == k:
                            break
            else:
                print("could'nt found counterfactuals for the features: ", f1, f2)
        if cfdf.empty != True:
            if all(col in cfdf.columns for col in order):
                pred_input = cfdf[order]
            else:
                pred_input = cfdf
            preds = np.asarray(model.predict(pred_input)).reshape(-1)
            cfdf = cfdf.loc[preds == int(desired_outcome)].reset_index(drop=True)
            if len(cfdf) > k:
                cfdf = cfdf.iloc[:k].reset_index(drop=True)
        return cfdf, two_feature_explore

    def Triple_F(self, df, test_instance, protected_features, feature_pairs, u_cat_f_list, numf, user_term_intervals, features_2change, model, desired_outcome, order, k):
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
        count = 0
        cfdf = pd.DataFrame()
        three_feature_explore = pd.DataFrame()
        temptempdf = pd.DataFrame()
        temptempdf = test_instance.copy()
        feature_to_use_list = feature_pairs
        for f in feature_to_use_list:
            f1 = f[0]
            f2 = f[1]
            two_feature_data = pd.DataFrame()
            temptempdf = pd.DataFrame()
            tempdf1 = pd.DataFrame()
            tempdf1 = test_instance.copy()
            if (f1 and f2 in numf) and (f1 and f2 not in protected_features):  # both numerical
                if f1 and f2 in user_term_intervals.keys():
                    interval_term_range1 = user_term_intervals[f1]
                    interval_term_range2 = user_term_intervals[f2]
                    start1 = int(interval_term_range1[0])
                    end1 = int(interval_term_range1[1])
                    start2 = int(interval_term_range2[0])
                    end2 = int(interval_term_range2[1])
                    reg_model, mse, rmse = self.regressionModel(df, f1, f2)
                    if isinstance(start1, int)and isinstance(end1, int):
                        f1_space = [item for item in range(start1, end1 + 1)]
                    else:
                        f1_space = sorted(np.round(random.uniform(start1, end1), 2) for _ in range(8))
                    while len(f1_space) != 0:
                        if len(f1_space) != 0:
                            low = 0
                            high = len(f1_space) - 1
                            mid = (high - low) // 2
                        else:
                            break
                        tempdf1 = test_instance.copy() 
                        tempdf1.loc[:, f1] = f1_space[mid]
                        temptempdf = tempdf1.copy()
                        tempdf1 = tempdf1.loc[:, tempdf1.columns != f2]
                        f2_val = reg_model.predict(tempdf1.values)
                        if isinstance(f2, float):
                            temptempdf.loc[:, f2] = f2_val[0]
                        else:
                            temptempdf.loc[:, f2] = float(int(f2_val[0]))
                        if int(f2_val[0]) >= start2 and int(f2_val[0]) <= end2:
                            for f3 in features_2change:
                                if f3 != f1 and f3 != f2 and f3 in user_term_intervals.keys(): # new condition is added due to wine about f3 in userterms.
                                    if f3 in numf: # if f3 is numerical
                                        reg_model_inner, mse, rmse = self.regressionModel(df, f1, f3)
                                        tempdf1 = temptempdf.copy()
                                        tempdf1 = tempdf1.loc[:, tempdf1.columns != f3]
                                        f3_val = reg_model_inner.predict(tempdf1.values)
                                        interval_term_range3 = user_term_intervals[f3]
                                        start3 = int(interval_term_range3[0])
                                        end3 = int(interval_term_range3[1])
                                        if int(f3_val[0]) >= start3 and int(f3_val[0]) <= end3:
                                            if isinstance(f3, float):
                                                temptempdf.loc[:, f3] = f3_val[0]
                                            else:
                                                temptempdf.loc[:, f3] = float(int(f3_val[0]))
                                            three_feature_explore = pd.concat([three_feature_explore, temptempdf],
                                                                                    ignore_index=True, axis=0, sort=False)
                                            tempdf1 = temptempdf.copy()
                                            temptempdf = temptempdf[order]
                                            pred = model.predict(temptempdf)
                                            if pred == desired_outcome:  #
                                                cfdf = pd.concat([cfdf, temptempdf], ignore_index=True, axis=0,
                                                                        sort=False)
                                            if len(cfdf) >= k:
                                                 break
                                    else: #f3 in u_cat_f_list: #f3 is categorical
                                        log_model_inner, ba = self.catclassifyModel(df, f1, f3)
                                        tempdf1 = temptempdf.copy()
                                        tempdf1 = tempdf1.loc[:, tempdf1.columns != f3]
                                        f3_val = log_model_inner.predict(tempdf1.values)
                                        three_feature_explore = pd.concat([three_feature_explore, temptempdf], ignore_index=True,
                                                                                  axis=0, sort=False)
                                        tempdf1 = temptempdf.copy()
                                        temptempdf = temptempdf[order]
                                        pred = model.predict(temptempdf)
                                        if pred == desired_outcome:  #
                                            cfdf = pd.concat([cfdf, temptempdf], ignore_index=True, axis=0, sort=False)
                                        if len(cfdf) >= k:
                                            break
                        try:
                            del f1_space[:mid+1]
                        except:
                            pass
            elif f1 and f2 in u_cat_f_list:  # both categorical
                # for feature in [f1, f2]:
                if f1 and f2 in user_term_intervals.keys():
                    tempdfcat = test_instance.copy()
                    tempdfcat.loc[:, f1] = user_term_intervals[f1][1] #0.0 if tempdfcat.loc[:, f1].values else 1.0
                    tempdfcat.loc[:, f2] = user_term_intervals[f2][1] #0.0 if tempdfcat.loc[:, f2].values else 1.0
                    tempdfcat = tempdfcat[order]
                    three_feature_explore = pd.concat([three_feature_explore, tempdfcat], ignore_index=True, axis=0, sort=False)
                    pred = model.predict(tempdfcat)
                    if pred == desired_outcome:  #
                        cfdf = pd.concat([cfdf, tempdfcat], ignore_index=True, axis=0, sort=False)
                    if len(cfdf) == k:
                        break
                    for f3 in features_2change:
                        if f3 != f1 and f3 != f2 and f3 not in protected_features:
                            if f3 in numf:  # if f3 is numerical
                                reg_model_inner, mse, rmse = self.regressionModel(df, f1, f3)
                                tempdf1 = tempdfcat.copy()
                                temptempdf = tempdfcat.copy()
                                tempdf1 = tempdf1.loc[:, tempdf1.columns != f3]
                                if tempdf1.empty != True:
                                    f3_val = reg_model_inner.predict(tempdf1.values)
                                else:
                                    pass
                                if isinstance(f3, float):
                                    temptempdf.loc[:, f3] = f3_val[0]
                                else:
                                    temptempdf.loc[:, f3] = float(int(f3_val[0]))
                                three_feature_explore = pd.concat([three_feature_explore, temptempdf],
                                                                  ignore_index=True, axis=0, sort=False)
                                tempdf1 = temptempdf.copy()
                                pred = model.predict(temptempdf)
                                if pred == desired_outcome:  #
                                    cfdf = pd.concat([cfdf, temptempdf], ignore_index=True, axis=0, sort=False)
                                if len(cfdf) == k:
                                    break
                            else:  # f3 is categorical
                                log_model_inner, ba = self.catclassifyModel(df, f1, f3)
                                # if ba > .5:
                                temptempdf = tempdfcat.copy()
                                tempdf1 = tempdfcat.copy()
                                tempdf1 = tempdf1.loc[:, tempdf1.columns != f3]
                                f3_val = log_model_inner.predict(tempdf1.values)
                                three_feature_explore = pd.concat([three_feature_explore, temptempdf],
                                                                      ignore_index=True, axis=0, sort=False)
                                if isinstance(f3, float):
                                    temptempdf.loc[:, f3] = f3_val[0]
                                else:
                                    temptempdf.loc[:, f3] = float(int(f3_val[0]))
                                temptempdf = temptempdf[order]
                                pred = model.predict(temptempdf)
                                if pred == desired_outcome:  #
                                    cfdf = pd.concat([cfdf, temptempdf], ignore_index=True, axis=0, sort=False)
                                if len(cfdf) == k:
                                    break

            elif f1 in numf and f2 in u_cat_f_list:  # num -> cat (binary classification)
                if f1 and f2 in user_term_intervals.keys():
                    interval_term_range1 = user_term_intervals[f1]
                    start1 = interval_term_range1[0]
                    end1 = interval_term_range1[1]
                    log_model, ba = self.catclassifyModel(df, f1, f2)
                    if isinstance(start1, int) and isinstance(end1, int):
                        f1_space = [item for item in range(start1, end1 + 1)]
                    else:
                        f1_space = sorted(np.round(random.uniform(start1, end1), 2) for _ in range(8))
                    # if ba >= 0.5:
                    while len(f1_space) != 0:
                        if len(f1_space) != 0:
                            low = 0
                            high = len(f1_space) - 1
                            mid = (high - low) // 2
                        else:
                            pass
                        tempdf1.loc[:, f1] = f1_space[mid]
                        temptempdf = tempdf1.copy()
                        tempdf1 = tempdf1.loc[:, tempdf1.columns != f2]
                        f2_val = log_model.predict(tempdf1.values)
                        if isinstance(f2, float):
                            temptempdf.loc[:, f2] = f2_val[0]
                        else:
                            temptempdf.loc[:, f2] = float(int(f2_val[0]))
                        if f2_val <= df[f2].max(): #f2_val >= df[f2].min() and 
                            for f3 in features_2change:
                                if f3 != f1 and f3 != f2 and f3 not in protected_features:
                                    if f3 in numf: # if f3 is numerical
                                        reg_model_inner, mse, rmse = self.regressionModel(df, f1, f3)
                                        tempdf1 = temptempdf.copy()
                                        tempdf1 = tempdf1.loc[:, tempdf1.columns != f3]
                                        f3_val = reg_model_inner.predict(tempdf1.values)
                                        if isinstance(f3, float):
                                            temptempdf.loc[:, f3] = f3_val[0]
                                        else:
                                            temptempdf.loc[:, f3] = float(int(f3_val[0]))
                                        three_feature_explore = pd.concat([three_feature_explore, temptempdf],
                                                                                      ignore_index=True, axis=0, sort=False)
                                        tempdf1 = temptempdf.copy()
                                        temptempdf = temptempdf[order]
                                        pred = model.predict(temptempdf)
                                        if pred == desired_outcome:  #
                                            cfdf = pd.concat([cfdf, temptempdf], ignore_index=True, axis=0,
                                                                         sort=False)
                                        if len(cfdf) == k:
                                            break
                                    else: #f3 is categorical
                                        log_model_inner, ba = self.catclassifyModel(df, f1, f3)
                                        tempdf1 = temptempdf.copy()
                                        tempdf1 = tempdf1.loc[:, tempdf1.columns != f3]
                                        f3_val = log_model_inner.predict(tempdf1.values)
                                        if isinstance(f3, float):
                                            temptempdf.loc[:, f3] = f3_val[0]
                                        else:
                                            temptempdf.loc[:, f3] = float(int(f3_val[0]))
                                        three_feature_explore = pd.concat([three_feature_explore, temptempdf], ignore_index=True, axis=0, sort=False)
                                        tempdf1 = temptempdf.copy()
                                        temptempdf = temptempdf[order]
                                        pred = model.predict(temptempdf)
                                        if pred == desired_outcome:  #
                                            cfdf = pd.concat([cfdf, temptempdf], ignore_index=True, axis=0, sort=False)
                                        if len(cfdf) == k:
                                            break
                        try:
                            del f1_space[:mid+1]
                        except:
                            pass
            elif f1 in u_cat_f_list and f2 in numf and (f1 and f2 not in protected_features): # cat and num
                if f1 and f2 in user_term_intervals.keys():
                    temptempdf = tempdf1.copy()
                    tempdf1.loc[:, f1] = user_term_intervals[f1][1] 
                    reg_model, mse, rmse = self.regressionModel(df, f1, f2)
                    tempdf1 = tempdf1.loc[:, tempdf1.columns != f2]
                    f2_val = reg_model.predict(tempdf1.values)
                    if isinstance(f2, float):
                        temptempdf.loc[:, f2] = f2_val[0]
                    else:
                        temptempdf.loc[:, f2] = int(f2_val[0])
                    if int(f2_val[0]) <= df[f2].max():  
                        for f3 in features_2change:
                            if f3 != f1 and f3 != f2 and f3 not in protected_features:
                                if f3 in numf:  # if f3 is numerical
                                    reg_model_inner, mse, rmse = self.regressionModel(df, f1, f3)
                                    tempdf1 = temptempdf.copy()
                                    tempdf1 = tempdf1.loc[:, tempdf1.columns != f3]
                                    f3_val = reg_model_inner.predict(tempdf1.values)
                                    if isinstance(f3, float):
                                        temptempdf.loc[:, f3] = f3_val[0]
                                    else:
                                        temptempdf.loc[:, f3] = int(f3_val[0])
                                    three_feature_explore = pd.concat([three_feature_explore, temptempdf],
                                                                          ignore_index=True, axis=0, sort=False)
                                    tempdf1 = temptempdf.copy()
                                    temptempdf = temptempdf[order]
                                    pred = model.predict(temptempdf)
                                    if pred == desired_outcome:  #
                                        cfdf = pd.concat([cfdf, temptempdf], ignore_index=True, axis=0,
                                                             sort=False)
                                    if len(cfdf) == k:
                                        break
                                else:  # f3 is categorical
                                    log_model_inner, ba = self.catclassifyModel(df, f1, f3)
                                    tempdf1 = temptempdf.copy()
                                    tempdf1 = tempdf1.loc[:, tempdf1.columns != f3]
                                    f3_val = log_model_inner.predict(tempdf1.values)
                                    if isinstance(f3, float):
                                        temptempdf.loc[:, f3] = f3_val[0]
                                    else:
                                        temptempdf.loc[:, f3] = int(f3_val[0])
                                    three_feature_explore = pd.concat([three_feature_explore, temptempdf],
                                                                              ignore_index=True, axis=0, sort=False)
                                    tempdf1 = temptempdf.copy()
                                    temptempdf = temptempdf[order]
                                    pred = model.predict(temptempdf)
                                    if pred == desired_outcome:  #
                                        cfdf = pd.concat([cfdf, temptempdf], ignore_index=True, axis=0,
                                                                 sort=False)
                                    if len(cfdf) == k:
                                        break
            else:
                print("Could'nt found counterfactuals for the features: ", f1, f2)
        if cfdf.empty != True:
            if all(col in cfdf.columns for col in order):
                pred_input = cfdf[order]
            else:
                pred_input = cfdf
            preds = np.asarray(model.predict(pred_input)).reshape(-1)
            cfdf = cfdf.loc[preds == int(desired_outcome)].reset_index(drop=True)
            if len(cfdf) > k:
                cfdf = cfdf.iloc[:k].reset_index(drop=True)
        return cfdf, three_feature_explore

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
            mad = median_absolute_deviation(X[:, continuous_features], axis=0)
            mad = np.array([v if v != 0 else 1.0 for v in mad])

            def _mad_cityblock(u, v):
                return mad_cityblock(u, v, mad)

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
                nbr_act, f_list = self.nbr_actionable_cfn(X_test[x:x + 1], cfdf[x:x + 1], features, changeable_features)
                if nbr_act >= self.min_actionable_other:  
                    count_in = 0
                    for j in f_list:
                        limit = X_test.at[x, j] + uf[j]
                        if cfdf.at[x, j] <= limit:
                            count_in += 1
                    if nbr_act == count_in:    
                        cfs = pd.concat([cfs, cfdf[x:x + 1]], ignore_index=True, axis=0)
                        flag = 1
                        idx1.append(x)
                        temp[x] = nbr_act
                        
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
                        count, f_list = self.nbr_actionable_cfn(x_row, cf_row, features, changeable_features)
                        if count >= self.min_actionable_feasible_other: 
                            count_in = 0
                            for j in f_list:
                                limit = X_test.at[x, j] + uf[j]
                                if cflist.at[x, j] <= limit:
                                    count_in += 1
                            if count == count_in:
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
                                    "reason": "actionability_limit_fail",
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
