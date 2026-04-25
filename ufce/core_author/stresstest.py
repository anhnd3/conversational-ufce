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



def generate_random_values(feature_ranges, feature_types, num_samples=1):
    """
    Generate random values within specified ranges for each feature using Monte Carlo sampling.

    Returns:
    - A dictionary where keys are feature names and values are lists of random values within the specified ranges.
    """
    random_values = {}

    for feature, (lower, upper) in feature_ranges.items():
        feature_type = feature_types.get(feature, 'float')  # Default to float if type is not specified

        if feature_type == 'integer':
            random_samples = np.random.randint(low=lower, high=upper + 1, size=num_samples)
        elif feature_type == 'float':
            random_samples = np.random.uniform(low=lower, high=upper, size=num_samples)
        else:
            raise ValueError(f"Unsupported feature type for {feature}: {feature_type}")

        random_values[feature] = random_samples.tolist()[0]

    return random_values

feature_ranges = {'Income': (1,40),
    'CCAvg': (1.5,3.5),
     'Family': (2, 3),
    'Education': (2, 3),
     'Mortgage': (40, 80),
    'CDAccount': (1, 1),
     'Online': (1, 1),
     'SecuritiesAccount': (1, 1),
    'CreditCard': (1, 1)
    }
feature_types = {
    'Income': 'integer',
    'CCAvg': 'float',
     'Family': 'integer',
    'Education': 'integer',
     'Mortgage': 'integer',
    'CDAccount': 'integer',
     'Online': 'integer',
     'SecuritiesAccount': 'integer',
    'CreditCard': 'integer'
}
ufmc = []
for i in range(10):
    random_values = generate_random_values(feature_ranges, feature_types)
    ufmc.append(random_values)
print(ufmc)

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
print(f'\t Top-5 Mutually-informed feature paris:{MI_FP[:5]}')

# %%
f2change = ['Income', 'CCAvg', 'Mortgage', 'Education']
f2cat = ['CDAccount', 'Online']

# %%
# # different levels of values (specific percentage of MAD represents to a diferent user value)
# uf1 = compute_percentage_for_uf_analysis(df, f2change, f2cat, 0.20) # 20% of mad
# uf2 = compute_percentage_for_uf_analysis(df, f2change, f2cat, 0.40) # 40% of mad
# uf3 = compute_percentage_for_uf_analysis(df, f2change, f2cat, 0.60) # 60% of mad
# uf4 = compute_percentage_for_uf_analysis(df, f2change, f2cat, 0.80) # 80% of mad
# uf5 = compute_percentage_for_uf_analysis(df, f2change, f2cat, 1.0) # # 100% of mad
# ufs = [uf5] #uf1, uf2, uf3, uf4, 
# # print(uf1)
models = ['lr'] # the experiment is run for 'lr' (Logistic Regression) as AR method doesn't work on 'mlp'

# %%
mnames = ['ufce1','ufce2', 'ufce3', 'dice', 'dice-uf' 'ar']
percent_cfs_all = pd.DataFrame()
time_cfs_all=pd.DataFrame()
# for u, uf in enumerate(ufs):
#     print(f' user feedback{u}:', uf[0])

scaler = StandardScaler()  # check verify the scaler
scaler=scaler.fit(Xtrain[:])

dfmc = pd.DataFrame()

for u, uf in enumerate(ufmc):
    print(f' user feedback{u}:', uf)
    cfmethods = ['DiCE', 'DiCE-UF', 'AR', 'UFCE1', 'UFCE2', 'UFCE3'] # 
    methodtimes = dict()
    cfcounts = dict()
    no_cf = 5
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
        k = 50
        if method == 'DiCE':
            outcome_label = 'Personal Loan'
            dicecfs, idx, methodtimes[i], flag1 = dice_cfexp(df, testset[:k], numf, f2changee, no_cf, bb, uf, outcome_label)
            dicecfs = dicecfs.drop(['Personal Loan', 'proximity'], axis=1)
            # if flag1 != 0:
            #     dicecfs = dicecfs.drop('Personal Loan', axis=1)
            dicetestdata = testset.loc[idx]
            dicetestdata = dicetestdata.reset_index(drop=True)
            dicecfs = dicecfs.reset_index(drop=True)
            dice_actcfs, flag, idx1, dic = ufc.actionability(dicecfs, dicetestdata, features, f2change, idx, uf, method="other")
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
            d_feas, diceout = ufc.feasibility(dicetestdata, dicecfs, Xtrain, features, f2change, bb, desired_outcome, uf, idx, method="other")
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
        elif method == 'DiCE-UF':
            outcome_label = 'Personal Loan'
            dicecfs_in, idx_in, methodtimes[i], flag1 = dice_cfexp_in(df, testset[:k], numf, f2changee, no_cf, bb, uf, outcome_label)
            dicecfs_in = dicecfs_in.drop(['Personal Loan', 'proximity'], axis=1)
            # if flag1 != 0:
            #     dicecfs = dicecfs.drop('Personal Loan', axis=1)
            dicetestdata = testset.loc[idx_in]
            dicetestdata = dicetestdata.reset_index(drop=True)
            dicecfs_in = dicecfs_in.reset_index(drop=True)
            dice_actcfs_in, flag, idx1, dic = ufc.actionability(dicecfs_in, dicetestdata, features, f2change, idx_in, uf, method="ufc")
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
            d_feas_in, diceout_in = ufc.feasibility(dicetestdata, dicecfs_in, Xtrain, features, f2change, bb, desired_outcome, uf, idx_in, method="ufc")
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
            arcfs, artime, idx = ar_cfexp(X, numf, bb, testset[:k], uf, scaler, Xtrain, f2changee)
            artestdata = testset.loc[idx]
            artestdata = artestdata.reset_index(drop=True)
            arcfs = arcfs.reset_index(drop=True)
            a_feas, arout = ufc.feasibility(artestdata, arcfs, Xtrain, features, f2change, bb, desired_outcome, uf, idx, method="other")
            ar_actcfs, flag, idx1, dic = ufc.actionability(arcfs, artestdata, features, f2change, idx, uf, method="other")
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
            step = {'Age':1, 'Experience':1, 'Income':1, 'CCAvg':.1, 'Family':1, 'Education':1, 'Mortgage':1, 'Securities Account':1,'CD Account':1,'Online':1, 'CreditCard':1}
            onecfs, methodtimes[i], idx = sfexp(X, data_lab1, testset[:k], uf, step, f2changee, numf, catf, bb, desired_outcome, k, features)
            onetestdata = testset.loc[idx]
            onetestdata = onetestdata.reset_index(drop=True)
            onecfs = onecfs.reset_index(drop=True)
            o_feas, oneout = ufc.feasibility(onetestdata, onecfs, Xtrain, features, f2change, bb, desired_outcome, uf, idx, method="ufc")
            one_actcfs, flag, idx1, dic = ufc.actionability(onecfs, onetestdata, features, f2change, idx, uf, method="ufc")
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
            twocfs, methodtimes[i], idx = dfexp(X, data_lab1, testset[:k], uf, MI_FP[:5], numf, catf, f2changee, protectf, bb, desired_outcome, k, features)
            twotestdata = testset.loc[idx]
            twotestdata = twotestdata.reset_index(drop=True)
            twocfs = twocfs.reset_index(drop=True)
            # for i in range(len(twocfs)):
            #     print(twotestdata[i:i + 1].values)
            #     print(twocfs[i:i+1].values)
            t_feas, twoout = ufc.feasibility(twotestdata, twocfs, Xtrain, features, f2change, bb, desired_outcome, uf, idx, method="ufc")
            two_actcfs, flag, idx1, dic = ufc.actionability(twocfs, twotestdata, features, f2change, idx, uf, method="ufc")
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
            threecfs, methodtimes[i], idx = tfexp(X, data_lab1, testset[:k], uf, MI_FP[:5], numf, catf, f2changee, protectf, bb, desired_outcome, k, features)#features
            threetestdata = testset.loc[idx]
            threetestdata = threetestdata.reset_index(drop=True)
            threecfs = threecfs.reset_index(drop=True)
            # for i in range(len(threecfs)):
            #     print(threetestdata[i:i + 1].values)
            #     print(threecfs[i:i+1].values)
            th_feas, threeout = ufc.feasibility(threetestdata, threecfs, Xtrain, features, f2change, bb, desired_outcome, uf, idx, method="ufc")
            three_actcfs, flag, idx1, dic = ufc.actionability(threecfs, threetestdata, features, f2change, idx, uf, method="ufc")
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
            # for x in idx1:
            #     pa = ufc.continuous_distance(testset[x:x+1], three_actcfs[x:x+1], numf, metric='euclidean', agg=None)
            #     threeprox_act.append(pa)
            # if len(threecfs) != 0:
            #     totest3 = testset.loc[foundidx3]
            #     totest3 = totest3.reset_index(drop=True)
            #     threecfs = threecfs.reset_index(drop=True)
            #     cfcounts[i] = len(threecfs)/len(testset[:k]) * 100
            # else:
            #     cfcounts[i] = 0
            # print("---------------3F idx---:", foundidx3)
            # for id in foundidx3:
            #     print(f'\t\t{id} Test instance \t:{testset[id:id + 1].values}')
            #     print(f'\t\t UF with MC \t:{interval3[id]}')
            #     print(f'\t\t Counterfactual \t:{threecfs[id:id + 1].values}')

    # calling all 7 evaluation metrics (properties)
    # # categorical proximity
    # mmeans, mstds = [], []
    # mmeans, mstds = Catproximity(onecfs, totest1, twocfs, totest2, threecfs, totest3, dicecfs, arcfs, testset[:k], catf)
    # df = pd.DataFrame(data=[mmeans], columns=mnames)
    # meandf_catproximity = pd.concat([meandf_catproximity, df], ignore_index=True, axis=0)
    # df = pd.DataFrame(data=[mstds], columns=mnames)
    # stddf_catproximity = pd.concat([stddf_catproximity, df], ignore_index=True, axis=0)
    # mmeans.extend(mstds)
    # df = pd.DataFrame(data=[mmeans], columns=cols)
    # catproxidf = pd.concat([catproxidf, df], ignore_index=True, axis=0)
    # # continuous proximity
    # mmeans, mstds = [], []
    # mmeans, mstds = Contproximity(onecfs, totest1, twocfs, totest2, threecfs, totest3, dicecfs, arcfs, testset[:k], numf)#Xtest
    # df = pd.DataFrame(data=[mmeans], columns=mnames)
    # meandf_contproximity = pd.concat([meandf_contproximity, df], ignore_index=True, axis=0)
    # df = pd.DataFrame(data=[mstds], columns=mnames)
    # stddf_contproximity = pd.concat([stddf_contproximity, df], ignore_index=True, axis=0)
    # mmeans.extend(mstds)
    # df = pd.DataFrame(data=[mmeans], columns=cols)
    # contproxidf = pd.concat([contproxidf, df], ignore_index=True, axis=0)
    # # sparsity
    # mmeans, mstds = [], []
    # mmeans, mstds = Sparsity(onecfs, totest1, twocfs, totest2, threecfs, totest3, dicecfs, arcfs, testset[:k], numf)#Xtest
    # df = pd.DataFrame(data=[mmeans], columns=mnames)
    # meandf_sparsity = pd.concat([meandf_sparsity, df], ignore_index=True, axis=0)
    # df = pd.DataFrame(data=[mstds], columns=mnames)
    # stddf_sparsity = pd.concat([stddf_sparsity, df], ignore_index=True, axis=0)
    # mmeans.extend(mstds)
    # df = pd.DataFrame(data=[mmeans], columns=cols)
    # spardf = pd.concat([spardf, df], ignore_index=True, axis=0)
    # # actionability
    # mmeans, mstds = [], []
    # mmeans, mstds = Actionability(onecfs, totest1, twocfs, testout2, threecfs, testout3, dicecfs, arcfs, testset[:k], features, f2change)#Xtest
    # df = pd.DataFrame(data=[mmeans], columns=mnames)
    # meandf_actionability = pd.concat([meandf_actionability, df], ignore_index=True, axis=0)
    # df = pd.DataFrame(data=[mstds], columns=mnames)
    # stddf_actionability = pd.concat([stddf_actionability, df], ignore_index=True, axis=0)
    # mmeans.extend(mstds)
    # df = pd.DataFrame(data=[mmeans], columns=cols)
    # actdf = pd.concat([actdf, df], ignore_index=True, axis=0)
    # mmeans, mstds = [], []
    # mmeans, mstds = Plausibility(onecfs, testout1, twocfs, testout2, threecfs, testout3, dicecfs, arcfs, testset[:k], Xtrain)#Xtest
    # df = pd.DataFrame(data=[mmeans], columns=mnames)
    # meandf_plausibility = pd.concat([meandf_plausibility, df], ignore_index=True, axis=0)
    # df = pd.DataFrame(data=[mstds], columns=mnames)
    # stddf_plausibility = pd.concat([stddf_plausibility, df], ignore_index=True, axis=0)
    # mmeans.extend(mstds)
    # df = pd.DataFrame(data=[mmeans], columns=cols)
    # plausdf = pd.concat([plausdf, df], ignore_index=True, axis=0)
    # mmeans, mstds = [], []
    # mmeans, mstds, d_valid, d_plaus, d_act, ar_valid, ar_plaus, ar_act, o_valid, o_plaus, o_act, t_valid, t_plaus, t_act, th_valid, th_plaus, th_act = Feasibility(onecfs,twocfs, threecfs, dicecfs, dicecfs_v, arcfs, arcfs_v, testset[:k], Xtrain, features, f2changee, bb, desired_outcome, outcome_label, uf[0])#Xtest
    # df = pd.DataFrame(data=[mmeans], columns=mnames)
    # meandf_feasibility = pd.concat([meandf_feasibility, df], ignore_index=True, axis=0)
    # df = pd.DataFrame(data=[mstds], columns=mnames)
    # stddf_feasibility = pd.concat([stddf_feasibility, df], ignore_index=True, axis=0)
    # mmeans.extend(mstds)
    # df = pd.DataFrame(data=[mmeans], columns=cols)
    # feasidf = pd.concat([feasidf, df], ignore_index=True, axis=0)
    # diceout.to_csv(r'C:\Users\laboratorio\Documents\GitHub\UFCE\results\dice\dice_cfs5.csv', encoding='utf-8')
    # dicetestdata.to_csv(r'C:\Users\laboratorio\Documents\GitHub\UFCE\results\dice\dice_test5.csv', encoding='utf-8')
    # arout.to_csv(r'C:\Users\laboratorio\Documents\GitHub\UFCE\results\ar\ar_cfs5.csv', encoding='utf-8')
    # artestdata.to_csv(r'C:\Users\laboratorio\Documents\GitHub\UFCE\results\ar\ar_test5.csv', encoding='utf-8')
    # oneout.to_csv(r'C:\Users\laboratorio\Documents\GitHub\UFCE\results\one\one_cfs5.csv', encoding='utf-8')
    # onetestdata.to_csv(r'C:\Users\laboratorio\Documents\GitHub\UFCE\results\one\one_test5.csv', encoding='utf-8')
    # twoout.to_csv(r'C:\Users\laboratorio\Documents\GitHub\UFCE\results\two\two_cfs5.csv', encoding='utf-8')
    # twotestdata.to_csv(r'C:\Users\laboratorio\Documents\GitHub\UFCE\results\two\two_test5.csv', encoding='utf-8')
    # threeout.to_csv(r'C:\Users\laboratorio\Documents\GitHub\UFCE\results\three\three_cfs5.csv', encoding='utf-8')
    # threetestdata.to_csv(r'C:\Users\laboratorio\Documents\GitHub\UFCE\results\three\three_test5.csv', encoding='utf-8')
    print("Testset length: ", k)
    print("\t Generated\tValid \t Proximity\t\t Sparsity\t Plausible\t Actionable\tFeasible\tPercentage_Feasible ")
    print(f"DiCE\t  {len(dicecfs)} \t\t {len(dicecfs)}\t {np.nanmean(diceprox):.2f}\t\t\t {d_spar:.2f} \t\t {d_plaus} \t\t {len(dice_actcfs)} \t\t  {d_feas} \t\t {(d_feas*100)/k}")
    print(f"DiCE-UF\t  {len(dicecfs_in)} \t\t {len(dicecfs_in)}\t {np.nanmean(diceprox_in):.2f}\t\t\t {d_spar_in:.2f} \t\t {d_plaus_in} \t\t {len(dice_actcfs_in)} \t\t  {d_feas_in} \t\t {(d_feas_in*100)/k}")
    print(f"AR \t  {len(arcfs)}\t\t {len(arcfs)}\t {np.nanmean(arprox):.2f}\t\t\t {a_spar:.2f} \t\t {a_plaus} \t\t {len(ar_actcfs)} \t\t {a_feas} \t\t {(a_feas*100)/k}")
    print(f"UFCE-1\t  {len(onecfs)}\t\t {len(onecfs)}\t {np.nanmean(oneprox):.2f}\t\t\t {o_spar:.2f} \t\t {len(onecfs)} \t\t {len(one_actcfs)} \t\t {o_feas} \t\t {o_feas*100/k}")
    print(f"UFCE-2\t  {len(twocfs)}\t\t {len(twocfs)}\t {np.nanmean(twoprox):.2f}\t\t\t {t_spar:.2f} \t\t {len(twocfs)} \t\t {len(two_actcfs)} \t\t {t_feas} \t\t {t_feas*100/k}")
    print(f"UFCE-3\t  {len(threecfs)}\t\t {len(threecfs)}\t {np.nanmean(threeprox):.2f}\t\t\t {th_spar:.2f} \t\t {len(threecfs)} \t\t {len(three_actcfs)} \t\t {th_feas} \t\t {th_feas*100/k}")


    print("Individual User Feedback Results UF-{u}")
    data = {'Method': ['DiCE', 'DiCE-UF', 'AR', 'UFCE-1', 'UFCE-2', 'UFCE-3'],
             'Generated': [len(dicecfs),len(dicecfs_in), len(arcfs), len(onecfs), len(twocfs), len(threecfs)],
             'Valid': [len(dicecfs),len(dicecfs_in), len(arcfs), len(onecfs), len(twocfs), len(threecfs)],
             'Plausible': [d_plaus, d_plaus_in, a_plaus, o_plaus, t_plaus, th_plaus],
             'Actionable': [len(dice_actcfs), len(dice_actcfs_in), len(ar_actcfs), len(one_actcfs), len(two_actcfs), len(three_actcfs)],
             'Feasible': [d_feas, d_feas_in, a_feas, o_feas, t_feas, th_feas],
             'Feasible-Percentage': [(d_feas*100)/k, (d_feas_in*100)/k, (a_feas*100)/k, (o_feas*100)/k, (t_feas*100)/k, (th_feas*100)/k]}
    df2print = pd.DataFrame(data)
    print(f'Evaluation Metrics for Feasibility:', df2print.to_latex(float_format="{:0.2f}".format))

    data2store = {
    'UFC1_Plaus': [o_plaus],
    'UFC1_Act': [len(one_actcfs)],
    'UFC1_Feas': [o_feas],
    'UFC2_Plaus': [t_plaus],
    'UFC2_Act': [len(two_actcfs)],
    'UFC2_Feas': [t_feas],
    'UFC3_Plaus': [th_plaus],
    'UFC3_Act': [len(three_actcfs)],
    'UFC3_Feas': [th_feas],
    'AR_Plaus': [a_plaus],
    'AR_Act': [len(ar_actcfs)],
    'AR_Feas': [len(dice_actcfs)],
    'DiCE_Plaus': [d_plaus],
    'DiCE_Act': [len(dice_actcfs)],
    'DiCE_Feas': [d_feas],
    'DiCE-UF_Plaus': [d_plaus_in],
    'DiCE-UF_Act': [len(dice_actcfs_in)],
    'DiCE-UF_Feas': [d_feas_in]}
    df2store = pd.DataFrame(data2store)
    dfmc = pd.concat([dfmc, df2store], ignore_index=True, axis=0)
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
print(dfmc)
print(dfmc.to_latex(float_format="{:0.2f}".format))
