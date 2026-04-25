#!/usr/bin/env python
# coding: utf-8
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
# ---

# %%
# %% [markdown]
# #### Import libraries

# %%
import os
import glob
import time
import gower
import random
import pandas as pd
import numpy as np
import json
import seaborn as sns
from numpy import arange
from numpy import hstack
from scipy import stats
from numpy import meshgrid
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from IPython.display import display
from pandas.errors import EmptyDataError
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist, pdist
from scipy.spatial.distance import _validate_vector
from scipy.stats import median_absolute_deviation
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
plt.style.use("seaborn-whitegrid")
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000

# %%
import ufce
from ufce import UFCE
from goodness import *
from cfmethods import *
from evaluations import *
from data_processing import *
from generate_text_explanations import *

# %%
ufc = UFCE()

# %% [markdown]
# this file contain the experiment code for user-feedback analysis with different levels of user-constraints, specificaly customised for Bank Loan dataset.

# %%
def compute_percentage_for_uf_analysis(df, f2c,fcat, seed):
    # Compute the median absolute deviation of each feature
    mad = df.std()
    # Add 5% of the MAD to each feature and store the updated values in a dictionary
    updated_values = {}
    list_update_values = []
    for feature in f2c:
        if feature == 'Education':
            updated_values[feature] = 1
        else:
            if feature == 'CCAvg':
                updated_values[feature] = round(mad[feature] * seed, 2)
            else:
                updated_values[feature] = round(mad[feature] * seed)
    for feature in fcat:
        updated_values[feature] = 1
    list_update_values.append(updated_values)
    return list_update_values

# %%
def modify_testinstance(test, cf, uf):
    temp = test.copy()
    for feature in uf.keys():
        temp[feature] = test[feature].values[0] + uf[feature]
    for f in cf:
        if temp[f].values < cf[f].values or cf[f].values < test[f].values:
            return False
        return True


# %%
# #### Read data
#path = r'/home/~/Downloads/UFCE-4GPU/data/'  # use this path on ubuntu. make sure you have correct path to UFCE folder.
path = r'C:\Users\laboratorio\Documents\GitHub\UFCE\data' # use this path format on windows system, verify your drive path to UFCE
pathbank = r'C:\Users\laboratorio\Documents\GitHub\UFCE\data\bank.csv'
datasetdf = pd.read_csv(pathbank)
datasetdf = datasetdf.sample(frac=1)
# print(datasetdf.mad(), datasetdf.std())

# %%
lr, lr_mean, lr_std, Xtest, Xtrain, X, Y, df = classify_dataset_getModel(datasetdf[:], data_name='bank') # this method returns trained ML model's, cleaned dataframe, and etc. #mlp, mlp_mean, mlp_std, 
models = {lr: [lr_mean, lr_std]}  #, mlp: [mlp_mean, mlp_std]}
print("cross-val mean score of lr", lr_mean)

# %%
print(f'Bank Dataset')

# %%
readpath = r'C:\Users\laboratorio\Documents\GitHub\UFCE\folds\bank\totest\testfold_1_pred_0.csv'
writepath = r'C:\Users\laboratorio\Documents\GitHub\UFCE\folds\bank\totest\\'
features, catf, numf, uf, f2changee, outcome_label, desired_outcome, nbr_features, protectf, data_lab0, data_lab1 = get_bank_user_constraints(df) # this will return user-constraints specific to data set.
# del data_lab1['Unnamed: 0']
# del data_lab1['age']
# del data_lab1['Experience']

# %%
# Take top mutual information sharing pairs of features
# MI_FP = ufc.get_top_MI_features(X, features)
with open('feature_pairs.json', 'r') as file:
    MI_FP = json.load(file)
print(f'\t Top-5 Mutually-informed feature paris:{MI_FP[:8]}')

# %%
f2change = ['Income', 'CCAvg', 'Mortgage', 'Education']
f2cat = ['CDAccount', 'Online', 'SecuritiesAccount', 'CreditCard']

# %%
# different levels of values (specific percentage of MAD represents to a diferent user value)
uf1 = compute_percentage_for_uf_analysis(df, f2change, f2cat, 0.20) # 20% of mad
uf2 = compute_percentage_for_uf_analysis(df, f2change, f2cat, 0.40) # 40% of mad
uf3 = compute_percentage_for_uf_analysis(df, f2change, f2cat, 0.60) # 60% of mad
uf4 = compute_percentage_for_uf_analysis(df, f2change, f2cat, 0.80) # 80% of mad
uf5 = compute_percentage_for_uf_analysis(df, f2change, f2cat, 1.0) # # 100% of mad
ufs = [uf1] #uf1, uf2, uf3, uf4, 
# print(ufs)
models = ['lr'] # the experiment is run for 'lr' (Logistic Regression) as AR method doesn't work on 'mlp'

# %%
mnames = ['ufce1','ufce2', 'ufce3', 'dice', 'ar']
percent_cfs_all = pd.DataFrame()
time_cfs_all=pd.DataFrame()

# for u, uf in enumerate(ufs):
#     print(f' user feedback{u}:', uf[0])

scaler = StandardScaler()  # check verify the scaler
scaler=scaler.fit(Xtrain[:])
        
for u, uf in enumerate(ufs):
    print(f' user feedback{u}:', uf[0])
    cfmethods = ['UFCE2', 'DiCE_in', 'DiCE', 'AR', 'UFCE1', 'UFCE3'] # 
    methodtimes = dict()
    cfcounts = dict()
    k = 1
    bb = lr
    desired_outcome = desired_outcome
    protectf = protectf
    meandf_catproximity = pd.DataFrame(columns=mnames)
    stddf_catproximity = pd.DataFrame(columns=mnames)
    meandf_contproximity = pd.DataFrame(columns=mnames)
    stddf_contproximity = pd.DataFrame(columns=mnames)
    meandf_sparsity = pd.DataFrame(columns=mnames)
    stddf_sparsity = pd.DataFrame(columns=mnames)
    meandf_actionability = pd.DataFrame(columns=mnames)
    stddf_actionability = pd.DataFrame(columns=mnames)
    meandf_plausibility = pd.DataFrame(columns=mnames)
    stddf_plausibility = pd.DataFrame(columns=mnames)
    meandf_feasibility = pd.DataFrame(columns=mnames)
    stddf_feasibility = pd.DataFrame(columns=mnames)
    meandf_diversity = pd.DataFrame(columns=mnames)
    stddf_diversity = pd.DataFrame(columns=mnames)
    meandf_feasibility = pd.DataFrame(columns=mnames)
    stddf_feasibility = pd.DataFrame(columns=mnames)
    cols = ['ufce1mean', 'ufce2mean', 'ufce3mean','dicemean', 'diceInmean', 'armean',  'ufce1std', 'ufce2std',
            'ufce3std', 'dicestd', 'diceInstd', 'arstd']
    jproxidf = pd.DataFrame(columns=cols)
    catproxidf = pd.DataFrame(columns=cols)
    contproxidf = pd.DataFrame(columns=cols)
    spardf = pd.DataFrame(columns=cols)
    actdf = pd.DataFrame(columns=cols)
    plausdf = pd.DataFrame(columns=cols)
    feasidf = pd.DataFrame(columns=cols)
    # for i, file in enumerate(testfolds[:]):
    #     print(f'\t\t\t Test Fold:{i} ')
    #     try:
    #         testset = pd.read_csv(file)
    #     except pd.io.common.EmptyDataError:
    #         print('File is empty or not found')
    #     else:
    testset = pd.read_csv(readpath)
    totest1 = pd.DataFrame()
    totest2 = pd.DataFrame()
    totest3 = pd.DataFrame()
    Xtrain = Xtrain.reset_index(drop=True)
    for i, method in enumerate(cfmethods):
        print(f'\t\t\t\t Method: {method}  --------------')
        k = 25
        no_cf = 5
        if method == 'DiCE':
            dicecfs, idx, methodtimes[i], flag1 = dice_cfexp(df, testset[:k], numf, f2changee, no_cf, bb, uf[0], outcome_label)
            if dicecfs.empty != True:
                dicecfs = dicecfs.drop(['Personal Loan', 'proximity'], axis=1)
            dicetestdata = testset.loc[idx]
            dicetestdata = dicetestdata.reset_index(drop=True)
            dicecfs = dicecfs.reset_index(drop=True)
            dice_actcfs, flag, idx1, dic = ufc.actionability(dicecfs, dicetestdata, features, f2change, idx, uf[0], method="other")
            # print("act", len(dice_actcfs))
            d_plaus = ufc.implausibility(dicecfs, dicetestdata, Xtrain, len(dicecfs), idx)
            # d_plaus_act = ufc.implausibility(dice_actcfs, dicetestdata, Xtrain, len(dice_actcfs), idx1)
            # print("plaus", plaus, plaus_act)
            dice_spar_dic, d_spar = ufc.sparsity_count(dicecfs, dicetestdata, features, idx)
            # d_spar_act = ufc.sparsity_count(dice_actcfs, dicetestdata, features, idx1)
            # print("spar", len(spar), len(spar_act))
            diceprox = []
            diceprox_act = []
            for x in range(len(dicecfs)):
                p = ufc.continuous_distance(dicetestdata[x:x+1], dicecfs[x:x+1], numf, metric='euclidean', agg=None)
                diceprox.append(p)
            # for x in range(len(dice_actcfs)):
            #     pa = ufc.continuous_distance(dicetestdata[x:x+1], dice_actcfs[x:x+1], numf, metric='euclidean', agg=None)
            #     diceprox_act.append(pa)
            d_feas, diceout = ufc.feasibility(dicetestdata, dicecfs, Xtrain, features, f2change, bb, desired_outcome, uf[0], idx, method="other")
            # print("Individual Without User Feedback Results UF-{u}")
            # data = {'Method': ['DiCE'],
            #  'Generated': [len(dicecfs)],
            #  'Valid': [len(dicecfs)],
            #  'Plausible': [d_plaus],
            #  'Actionable': [len(dice_actcfs)],
            #  'Feasible': [d_feas],
            #  'Feasible-Percentage': [(d_feas*100)/k]}
            # df = pd.DataFrame(data)
            # print(f'DiCE Evaluation Metrics for Feasibility:', df.to_latex(float_format="{:0.2f}".format))
            
            # print("feas", d_feas)
            # print("prox_act", np.nanmean(diceprox_act))
            
                # count = 0
                # for x in range(len(testset[:k])):
                #     flag = modify_testinstance(testset[x:x+1], dicecfs[x:x+1], uf[0])
                #     if flag:
                #         count = count + 1
            #     cfcounts[i] = len(dicecfs)/len(testset[:k]) * 100
            # else:
            #     cfcounts[i] = 0
            #print(f'\t\t\t\t Counterfactuals \t:{dicecfs.values}')
            # print(f"---DiCE generated {len(dicecfs)} with UF and {len(dicecfs_v)} without UF-----------")
        elif method == 'DiCE_in':
            dicecfs_in, idx_in, methodtimes[i], flag1 = dice_cfexp_in(df, testset[:k], numf, f2changee, no_cf, bb, uf[0], outcome_label)
            if dicecfs_in.empty != True:
                dicecfs_in = dicecfs_in.drop(['Personal Loan', 'proximity'], axis=1)
            dicetestdata = testset.loc[idx_in]
            dicetestdata = dicetestdata.reset_index(drop=True)
            dicecfs_in = dicecfs_in.reset_index(drop=True)
            dice_actcfs_in, flag, idx1, dic = ufc.actionability(dicecfs_in, dicetestdata, features, f2change, idx_in, uf[0], method="ufc")
            # print("act", len(dice_actcfs))
            d_plaus_in = ufc.implausibility(dicecfs_in, dicetestdata, Xtrain, len(dicecfs_in), idx_in)
            # d_plaus_act = ufc.implausibility(dice_actcfs, dicetestdata, Xtrain, len(dice_actcfs), idx1)
            # print("plaus", plaus, plaus_act)
            dice_spar_dic_in, d_spar_in = ufc.sparsity_count(dicecfs_in, dicetestdata, features, idx_in)
            # d_spar_act = ufc.sparsity_count(dice_actcfs, dicetestdata, features, idx1)
            # print("spar", len(spar), len(spar_act))
            diceprox_in = []
            diceprox_act_in = []
            for x in range(len(dicecfs_in)):
                p = ufc.continuous_distance(dicetestdata[x:x+1], dicecfs_in[x:x+1], numf, metric='euclidean', agg=None)
                diceprox_in.append(p)
            # for x in range(len(dice_actcfs)):
            #     pa = ufc.continuous_distance(dicetestdata[x:x+1], dice_actcfs[x:x+1], numf, metric='euclidean', agg=None)
            #     diceprox_act.append(pa)
            d_feas_in, diceout_in = ufc.feasibility(dicetestdata, dicecfs_in, Xtrain, features, f2change, bb, desired_outcome, uf[0], idx_in, method="ufc")
            # print("Individual Without User Feedback Results UF-{u}")
            # data = {'Method': ['DiCE'],
            #  'Generated': [len(dicecfs)],
            #  'Valid': [len(dicecfs)],
            #  'Plausible': [d_plaus],
            #  'Actionable': [len(dice_actcfs)],
            #  'Feasible': [d_feas],
            #  'Feasible-Percentage': [(d_feas*100)/k]}
            # df = pd.DataFrame(data)
            # print(f'DiCE Evaluation Metrics for Feasibility:', df.to_latex(float_format="{:0.2f}".format))
            
            # print("feas", d_feas)
            # print("prox_act", np.nanmean(diceprox_act))
            
                # count = 0
                # for x in range(len(testset[:k])):
                #     flag = modify_testinstance(testset[x:x+1], dicecfs[x:x+1], uf[0])
                #     if flag:
                #         count = count + 1
            #     cfcounts[i] = len(dicecfs)/len(testset[:k]) * 100
            # else:
            #     cfcounts[i] = 0
            #print(f'\t\t\t\t Counterfactuals \t:{dicecfs.values}')
            # print(f"---DiCE generated {len(dicecfs)} with UF and {len(dicecfs_v)} without UF-----------")
        elif method == 'AR':
            arcfs, artime, idx = ar_cfexp(X, numf, bb, testset[:k], uf[0], scaler, Xtrain, f2changee)
            # arcfs = arcfs.drop(['proximity'], axis=1)
            artestdata = testset.loc[idx]
            artestdata = artestdata.reset_index(drop=True)
            arcfs = arcfs.reset_index(drop=True)
            a_feas, arout = ufc.feasibility(artestdata, arcfs, Xtrain, features, f2change, bb, desired_outcome, uf[0], idx, method="other")
            ar_actcfs, flag, idx1, dic = ufc.actionability(arcfs, artestdata, features, f2change, idx, uf[0], method="other")
            # print("act", len(dicecfs), len(dice_actcfs))
            a_plaus = ufc.implausibility(arcfs, artestdata, Xtrain, len(arcfs), idx)
            # a_plaus_act = ufc.implausibility(ar_actcfs, artestdata, Xtrain, len(ar_actcfs), idx1)
            # print("plaus", plaus, plaus_act)
            ar_spar_dic,a_spar = ufc.sparsity_count(arcfs, artestdata, features, idx)
            # a_spar_act = ufc.sparsity_count(ar_actcfs, artestdata, features, idx1)
            # print("spar", len(spar), len(spar_act))
            arprox = []
            arprox_act = []
            for x in range(len(arcfs)):
                p = ufc.continuous_distance(artestdata[x:x+1], arcfs[x:x+1], numf, metric='euclidean', agg=None)
                arprox.append(p)
            # for x in range(len(ar_actcfs)):
            #     pa = ufc.continuous_distance(artestdata[x:x+1], ar_actcfs[x:x+1], numf, metric='euclidean', agg=None)
            #     arprox_act.append(pa)
            # if flag1 != 0:
                # count = 0
                # for x in range(len(testset[:k])):
                #     flag = modify_testinstance(testset[x:x + 1], arcfs[x:x + 1], uf[0])
                #     if flag:
                #         count = count + 1
            #     cfcounts[i] = len(arcfs) / len(testset[:k]) * 100
            # else:
            #     cfcounts[i] = 0
            #print(f'\t\t\t\t Counterfactual \t:{arcfs.values}')
        elif method == 'UFCE1':
            step = {'Income':1, 'CCAvg':.1, 'Family':1, 'Education':1, 'Mortgage':1, 'Securities Account':1,'CD Account':1,'Online':1, 'CreditCard':1}
            onecfs, methodtimes[i], idx = sfexp(X, data_lab1, testset[:k], uf[0], step, f2changee, numf, catf, bb, desired_outcome, no_cf, features)
            onetestdata = testset.loc[idx]
            onetestdata = onetestdata.reset_index(drop=True)
            onecfs = onecfs.reset_index(drop=True)
            o_feas, oneout = ufc.feasibility(onetestdata, onecfs, Xtrain, features, f2change, bb, desired_outcome, uf[0], idx, method="ufc")
            one_actcfs, flag, idx1, dic = ufc.actionability(onecfs, onetestdata, features, f2change, idx, uf[0], method="ufc")
            # for i in range(len(onecfs)):
            #     print(onetestdata[i:i + 1].values)
            #     print(onecfs[i:i+1].values)
            # print("act", len(dicecfs), len(dice_actcfs))
            o_plaus = ufc.implausibility(onecfs, onetestdata, Xtrain, len(onecfs), idx)
            # o_plaus_act = ufc.implausibility(one_actcfs, testset[:k], Xtrain, len(one_actcfs), idx1)
            # print("plaus", plaus, plaus_act)
            o_spar_dic, o_spar = ufc.sparsity_count(onecfs, onetestdata, features, idx)
            # o_spar_act = ufc.sparsity_count(one_actcfs, testset[:k], features, idx1)
            # print("spar", len(spar), len(spar_act))
            oneprox = []
            oneprox_act = []
            for x in range(len(onecfs)):
                p = ufc.continuous_distance(onetestdata[x:x+1], onecfs[x:x+1], numf, metric='euclidean', agg=None)
                oneprox.append(p)
            # for x in idx1:
            #     pa = ufc.continuous_distance(testset[x:x+1], one_actcfs[x:x+1], numf, metric='euclidean', agg=None)
            #     oneprox_act.append(pa)
            # if len(onecfs) != 0:
            #     totest1 = testset.loc[foundidx1]
            #     totest1 = totest1.reset_index(drop=True)
            #     onecfs = onecfs.reset_index(drop=True)
            #     cfcounts[i] = len(onecfs)/len(testset[:k]) * 100
            # else:
            #     cfcounts[i] = 0
            # print("---------------1F idx---:", foundidx1)
            # for id in foundidx1:
            #     print(f'\t\t\t\t{id} Test instance \t:{testset[id:id+1].values}')
            #     print(f'\t\t\t\t UF with MC \t:{interval1[id]}')
            #     print(f'\t\t Counterfactual \t:{onecfs[id:id+1].values}')
        elif method == 'UFCE2':
            twocfs, methodtimes[i], idx = dfexp(X, data_lab1, testset[:k], uf[0], MI_FP[:8], numf, catf, f2changee, protectf, bb, desired_outcome, no_cf, features)
            print(len(twocfs))
            twotestdata = testset.loc[idx]
            twotestdata = twotestdata.reset_index(drop=True)
            twocfs = twocfs.reset_index(drop=True)
            # for i in range(len(twocfs)):
            #     print(twotestdata[i:i + 1].values)
            #     print(twocfs[i:i+1].values)
            t_feas, twoout = ufc.feasibility(twotestdata, twocfs, Xtrain, features, f2change, bb, desired_outcome, uf[0], idx, method="other")
            two_actcfs, flag, idx1, dic = ufc.actionability(twocfs, twotestdata, features, f2change, idx, uf[0], method="other")
            # print("act", len(dicecfs), len(dice_actcfs))
            t_plaus = ufc.implausibility(twocfs, twotestdata, Xtrain, len(twocfs), idx)
            # t_plaus_act = ufc.implausibility(two_actcfs, testset[:k], Xtrain, len(two_actcfs), idx1)
            # print("plaus", plaus, plaus_act)
            t_spar_dic, t_spar = ufc.sparsity_count(twocfs, twotestdata, features, idx)
            # t_spar_act = ufc.sparsity_count(two_actcfs, testset[:k], features, idx1)
            # print("spar", len(spar), len(spar_act))
            twoprox = []
            twoprox_act = []
            for x in range(len(twocfs)):
                p = ufc.continuous_distance(twotestdata[x:x+1], twocfs[x:x+1], numf, metric='euclidean', agg=None)
                twoprox.append(p)
            # for x in idx1:
            #     pa = ufc.continuous_distance(testset[x:x+1], two_actcfs[x:x+1], numf, metric='euclidean', agg=None)
            #     twoprox_act.append(pa)
            # if len(twocfs) != 0:
            #     totest2 = testset.loc[foundidx2]
            #     totest2 = totest2.reset_index(drop=True)
            #     twocfs = twocfs.reset_index(drop=True)
            #     cfcounts[i] = len(twocfs)/len(testset[:k]) * 100
            # else:
            #     cfcounts[i] = 0
            # print("---------------2F idx---:", foundidx2)
            
            # for id in foundidx2:
            #     print(f'\t\t\t\t{id} Test instance \t:{testset[id:id + 1].values}')
            #     print(f'\t\t\t\t UF with MC \t:{interval2[id]}')
            #     print(f'\t\t\t\t Counterfactual \t:{twocfs[id:id + 1].values}')
        else:
            threecfs, methodtimes[i], idx = tfexp(X, data_lab1, testset[:k], uf[0], MI_FP[:8], numf, catf, f2changee, protectf, bb, desired_outcome, no_cf, features)#features
            threetestdata = testset.loc[idx]
            threetestdata = threetestdata.reset_index(drop=True)
            threecfs = threecfs.reset_index(drop=True)
            # for i in range(len(threecfs)):
            #     print(threetestdata[i:i + 1].values)
            #     print(threecfs[i:i+1].values)
            th_feas, threeout = ufc.feasibility(threetestdata, threecfs, Xtrain, features, f2change, bb, desired_outcome, uf[0], idx, method="ufc")
            three_actcfs, flag, idx1, dic = ufc.actionability(threecfs, threetestdata, features, f2change, idx, uf[0], method="ufc")
            # print("act", len(dicecfs), len(dice_actcfs))
            th_plaus = ufc.implausibility(threecfs, threetestdata, Xtrain, len(threecfs), idx)
            # th_plaus_act = ufc.implausibility(three_actcfs, testset[:k], Xtrain, len(three_actcfs), idx1)
            # print("plaus", plaus, plaus_act)
            th_spar_dic, th_spar = ufc.sparsity_count(threecfs, threetestdata, features, idx)
            # th_spar_act = ufc.sparsity_count(one_actcfs, testset[:k], features, idx1)
            # print("spar", len(spar), len(spar_act))
            threeprox = []
            threeprox_act = []
            for x in range(len(threecfs)):
                p = ufc.continuous_distance(threetestdata[x:x+1], threecfs[x:x+1], numf, metric='euclidean', agg=None)
                threeprox.append(p)

    print("Testset length: ", k)
    print("\t Generated\tValid \t Proximity\t\t Sparsity\t Plausible\t Actionable\tFeasible\tPercentage_Feasible ")
    print(f"DiCE\t  {len(dicecfs)} \t\t {len(dicecfs)}\t {np.nanmean(diceprox):.2f}\t\t\t {d_spar:.2f} \t\t {d_plaus} \t\t {len(dice_actcfs)} \t\t  {d_feas} \t\t {(d_feas*100)/k}")
    print(f"DiCE_in\t  {len(dicecfs_in)} \t\t {len(dicecfs_in)}\t {np.nanmean(diceprox_in):.2f}\t\t\t {d_spar_in:.2f} \t\t {d_plaus_in} \t\t {len(dice_actcfs_in)} \t\t  {d_feas_in} \t\t {(d_feas_in*100)/k}")
    print(f"AR \t  {len(arcfs)}\t\t {len(arcfs)}\t {np.nanmean(arprox):.2f}\t\t\t {a_spar:.2f} \t\t {a_plaus} \t\t {len(ar_actcfs)} \t\t {a_feas} \t\t {(a_feas*100)/k}")
    print(f"UFCE-1\t  {len(onecfs)}\t\t {len(onecfs)}\t {np.nanmean(oneprox):.2f}\t\t\t {o_spar:.2f} \t\t {len(onecfs)} \t\t {len(one_actcfs)} \t\t {o_feas} \t\t {o_feas*100/k}")
    print(f"UFCE-2\t  {len(twocfs)}\t\t {len(twocfs)}\t {np.nanmean(twoprox):.2f}\t\t\t {t_spar:.2f} \t\t {len(twocfs)} \t\t {len(two_actcfs)} \t\t {t_feas} \t\t {t_feas*100/k}")
    print(f"UFCE-3\t  {len(threecfs)}\t\t {len(threecfs)}\t {np.nanmean(threeprox):.2f}\t\t\t {th_spar:.2f} \t\t {len(threecfs)} \t\t {len(three_actcfs)} \t\t {th_feas} \t\t {th_feas*100/k}")


    print("Individual User Feedback Results UF-{u}")
    data = {'Method': ['DiCE', 'DiCE_in', 'AR', 'UFCE-1', 'UFCE-2', 'UFCE-3'],
             'Generated': [len(dicecfs), len(dicecfs_in), len(arcfs), len(onecfs), len(twocfs), len(threecfs)],
             'Valid': [len(dicecfs), len(dicecfs_in), len(arcfs), len(onecfs), len(twocfs), len(threecfs)],
             'Plausible': [d_plaus, d_plaus_in, a_plaus, o_plaus, t_plaus, th_plaus],
             'Actionable': [len(dice_actcfs), len(dice_actcfs_in), len(ar_actcfs), len(one_actcfs), len(two_actcfs), len(three_actcfs)],
             'Feasible': [d_feas, d_feas_in, a_feas, o_feas, t_feas, th_feas],
             'Feasible-Percentage': [(d_feas*100)/k, (d_feas_in*100)/k, (a_feas*100)/k, (o_feas*100)/k, (t_feas*100)/k, (th_feas*100)/k]}
    df = pd.DataFrame(data)
    print(f'Evaluation Metrics for Feasibility:', df.to_latex(float_format="{:0.2f}".format))

    # # here storing the time and percentage of counterfactuals for each cfmethod.
    # temptime = pd.DataFrame([methodtimes])
    # time_cfs_all = pd.concat([time_cfs_all, temptime], ignore_index=True)
    # tempcount = pd.DataFrame([cfcounts])
    # percent_cfs_all = pd.concat([percent_cfs_all, tempcount], ignore_index=True)
    # # print("Time of all methods for this uf")
    # print(display(time_cfs_all))
    # print("Percentage of all methods for this uf")
    # print(display(percent_cfs_all))
    # print(f'\t\t\t\t-----fold_mean_values of all evaluation metrics----')

    # print(f'Mean and St.dev of Joint-Proximity:', jproxidf.to_latex(float_format="{:0.2f}".format))
    # print(f'Mean and St.dev of Cat-Proximity:', catproxidf.to_latex(float_format="{:0.2f}".format))
    # print(f'Mean and St.dev of Cont-Proximity:', contproxidf.to_latex(float_format="{:0.2f}".format))
    # print(f'Mean and St.dev of Sparsity:', spardf.to_latex(float_format="{:0.2f}".format))
    # print(f'Mean and St.dev of Actionability:', actdf.to_latex(float_format="{:0.2f}".format))
    # print(f'Mean and St.dev of Plausibility:', plausdf.to_latex(float_format="{:0.2f}".format))
    # print(f'Mean and St.dev of Feasibility:', feasidf.to_latex(float_format="{:0.2f}".format))
    break
