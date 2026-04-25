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
from goodness import *


"""
 This is the module that could be utilized to take the user feedback in the form of user preferences
 Initially the user preferences will be the intervals, later these preferences would be taken from the user with an interface.
"""


def _feature_matrix(values, features):
    if values is None:
        return None
    if hasattr(values, "loc"):
        return values.loc[:, features].to_numpy()
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr[:, features]


class UFCE():
    
    def __init__(self):
        self.dataset = 'bank'
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
    
    def NNkdtree(self, data_lab1, test_inst, radius):
        """
        :param data_lab1: desired space
        :param test_inst:
        :param radius: radius for search in the space, should be faithful to data distribution
        :return:
        """
        import numpy as np
        from scipy.spatial import KDTree
        tree = KDTree(data_lab1)
        idx = tree.query_ball_point(test_inst.values[0], r=radius)
        nn = pd.DataFrame.from_records(tree.data[idx], columns=test_inst.columns)
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

       
    def pred_for_binsearch(self, tempdf, feature, start, mid, end, model):
        """
        :param tempdf: a temporary dataframe
        :param feature: feature
        :param start: start value
        :param mid: mid value
        :param: end: end value
        :param model: ML blackbox
        :return pred, tempdf: prediction and related dataframe
        """
        tempdf.loc[:, feature] = mid
        pred = model.predict(tempdf)
        return pred, tempdf
       
   
    def Single_F(self, test_instance, u_cat_f_list, user_term_intervals, model, outcome, step):
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
                        pred, tempdf1 = self.pred_for_binsearch(tempdf, feature, start, mid, end, model)
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
                            pred = model.predict(temptempdf)
                            if pred == desired_outcome:  #
                                cfdf = pd.concat([cfdf, temptempdf], ignore_index=True, axis=0, sort=False)
                            cfdf = pd.concat([cfdf, temptempdf], ignore_index=True, axis=0)
                            if len(cfdf) == k:
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
                                            if len(cfdf) == k:
                                                 break
                                            cfdf = pd.concat([cfdf, temptempdf], ignore_index=True, axis=0,
                                                                        sort=False)
                                            if len(cfdf) == k:
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
                                            if len(cfdf) == k:
                                                break
                                        cfdf = pd.concat([cfdf, temptempdf], ignore_index=True, axis=0,
                                                                     sort=False)
                                        if len(cfdf) == k:
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
        if metric == 'mad':
            if X is None:
                raise ValueError("X is required when metric='mad'")
            x_values = _feature_matrix(x, continuous_features)
            cf_values = _feature_matrix(cf_list, continuous_features)
            X_values = _feature_matrix(X, continuous_features)
            mad = median_absolute_deviation(X_values, axis=0)
            mad = np.array([v if v != 0 else 1.0 for v in mad])

            def _mad_cityblock(u, v):
                return mad_cityblock(u, v, mad)

            dist = cdist(x_values, cf_values, metric=_mad_cityblock)
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
        
    def lofn(self, x, cf_list, X, scaler):
        """
        :param x: test instance
        :param cf_list: list of counterfactuals or single counterfactual
        :param X: feature data space
        :param scaler: scaler model
        :return: 1, 0
        """
        if x.empty != True and cf_list.empty != True:
            X_train = np.vstack([x.values.reshape(1, -1), X])
            nX_train = scaler.transform(X_train) 
            ncf_list = scaler.transform(cf_list)

            clf = LocalOutlierFactor(n_neighbors=100, novelty=True) 
            clf.fit(nX_train)
            lof_values = clf.predict(ncf_list)
        else:
            lof_values = 0
        return lof_values

    def implausibility(self, cfdf, Xtest, Xtrain, K, idx):
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
        scaler = StandardScaler() # check verify the scaler
        scaler = scaler.fit(Xtrain)
        result = 0
        if len(Xtest) != 0 and len(cfdf) != 0:
            for t in range(len(cfdf)):
                res = int(self.lofn(Xtest[t:t + 1], cfdf[t:t + 1], Xtrain[:], scaler)) #lof - local outlier factor
                if res == 1:
                    result += 1
        else:
            return result
        return result
        
    # Sparsity
    #cf feature changes and avg change
    def nbr_changes_per_cfn(self, x, cf_list):
        """
        :param x: test instance
        :param cf_list: couneterfactuals(s)
        :param continuous_features:
        :return: nbr_changes
        """
        # nbr_features = cf_list.shape[1]
        features = (list(x.columns))
        nbr_changes = 0
        for j in features:
            if cf_list[j].values != x[j].values:
                # if j in continuous_features:
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
        :param x:
        :param cf_list:
        :param features:
        :param f2change:
        :return:
        """
        f_list = []
        nbr_actionable = 0
        for j in features:
            if cf_list[j].values != x[j].values and j in f2change:
                nbr_actionable += 1
                f_list.append(j)
        return nbr_actionable, f_list

    def changes_per_cf(self, x, cf):
        features = (list(x.columns))
        nbr_changes = 0
        for j in features:
            if cf[j].values != x[j].values:
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
                if nbr_act >= 3:  
                    count_in = 0
                    for j in f_list:
                        limit = X_test.at[x, j] + uf[j]
                        if cfdf.at[x, j] <= limit:
                            count_in += 1
                    if nbr_act == count_in:    
                        cfs = pd.concat([cfs, cfdf[x:x + 1]], ignore_index=True, axis=0)
                        flag = 1
                        idx1.append(x)
                        temp[x] = count
                        
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
    

    def feasibility(self, X_test, cffile, X_train, features, changeable_features, model, desired_outcome, uf, idx, method):
        X_test.reset_index(drop=True, inplace=True)
        X_train.reset_index(drop = True, inplace = True)
        cffile.reset_index(drop=True, inplace=True)
        cflist = cffile
        scaler = StandardScaler()  # check verify the scaler
        scaler = scaler.fit(X_train[:])
        feasible = 0
        feas = 0
        temp = pd.DataFrame()
        # temptest = pd.DataFrame()
        if cflist.empty != True:
            for x in range(len(cflist)):
                if method == "other":
                    plaus = int(self.lofn(X_test[x:x + 1], cflist[x:x + 1], X_train[:], scaler))
                    if plaus == 1:
                        count, f_list = self.nbr_actionable_cfn(X_test[x:x + 1], cflist[x:x + 1], features, changeable_features)
                        if count >= 2: 
                            count_in = 0
                            for j in f_list:
                                limit = X_test.at[x, j] + uf[j]
                                if cflist.at[x, j] <= limit:
                                    count_in += 1
                            if count == count_in:
                                feas += 1
                                temp = pd.concat([temp, cflist[x:x + 1]], axis=0, ignore_index=True)
                            
                else:    
                    plaus = int(self.lofn(X_test[x:x + 1], cflist[x:x + 1], X_train[:], scaler)) #make it 1000 in other cases, except movie
                    if plaus == 1:
                        feas += 1
                        temp = pd.concat([temp, cflist[x:x + 1]], axis=0, ignore_index=True)

            if feas != 0:
                return feas, temp
            else:
                feas = feasible  
        else:
            feas = feasible
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
