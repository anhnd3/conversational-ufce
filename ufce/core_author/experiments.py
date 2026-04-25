#!/usr/bin/env python
# coding: utf-8

# #### Import libraries

import os
import glob
import time
import gower
import random
import json
import pandas as pd
import numpy as np
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

plt.style.use("seaborn-whitegrid")
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000

import ufce
from ufce import UFCE
from goodness import *
from cfmethods import *
from evaluations import *
from data_processing import *
from generate_text_explanations import *

ufc = UFCE()


# ### Complete experiment

datasets = {'grad':'grad.csv'} #'bank':'bank.csv', 'grad':'grad.csv', 'bupa':'bupa.csv', 'wine':'wine.csv', 'movie':'movie.csv' 
models = ['lr'] # the experiment is run for 'lr' as AR method doesn't work on 'mlp'
for dataset in datasets.keys():
    # #### Read data
    #path = r'/home/~/data/'  # use this path on ubuntu. make sure you have correct path to UFCE folder.
    path = r'C:\Users\laboratorio\Documents\GitHub\UFCE\data' # use this path on windows system
    datafile = datasets[dataset]
    datasetdf = pd.read_csv(os.path.join(path, datafile))
    lr, lr_mean, lr_std, Xtest, Xtrain, X, Y, df = classify_dataset_getModel(datasetdf, data_name=dataset) # this method returns trained ML model's, cleaned dataframe, and etc.
    models = {lr: [lr_mean, lr_std]} #, mlp: [mlp_mean, mlp_std]}
    print(f'\t Dataset:{dataset}')
    if dataset == 'bank':
        readpath = r'C:\Users\laboratorio\Documents\GitHub\UFCE\folds\bank\\'
        writepath = r'C:\Users\laboratorio\Documents\GitHub\UFCE\folds\bank\totest\\'
        # readpath = r'/home/~/folds/bank/' # use this path on Ubuntu
        # writepath = r'/home/~/folds/bank/totest/' # use this path on Ubuntu
        features, catf, numf, uf, f2change, outcome_label, desired_outcome, nbr_features, protectf, data_lab0, data_lab1 = get_bank_user_constraints(datasetdf) # this will return user-constraints specific to data set.
        # print("global uf:", uf)
    elif dataset == 'grad':
        # readpath = r'/home/~/folds/grad/' # use this path on Ubuntu
        # writepath = r'/home/~/folds/totest/' # use this path on Ubuntu
        readpath = r'C:\Users\laboratorio\Documents\GitHub\UFCE\folds\grad\\'
        writepath = r'C:\Users\laboratorio\Documents\GitHub\UFCE\folds\grad\totest\\'
        features, catf, numf, uf, f2change, outcome_label, desired_outcome, nbr_features, protectf, data_lab0, data_lab1 = get_grad_user_constraints(datasetdf)
    elif dataset == 'wine':
        # readpath = r'/home/~/folds/wine/' # for ubuntu
        # writepath = r'/home/~/folds/wine/totest/'
        readpath = r'C:\Users\laboratorio\Documents\GitHub\UFCE\folds\wine\\'
        writepath = r'C:\Users\laboratorio\Documents\GitHub\UFCE\folds\wine\totest\\'
        features, catf, numf, uf, f2change, outcome_label, desired_outcome, nbr_features, protectf, data_lab0, data_lab1 = get_wine_user_constraints(datasetdf)
    elif dataset == 'bupa':
        # readpath = r'/home/~/folds/bupa/'
        # writepath = r'/home/~/folds/bupa/totest'
        readpath = r'C:\Users\laboratorio\Documents\Github\UFCE\folds\bupa\\'
        writepath = r'C:\Users\laboratorio\Documents\Github\UFCE\folds\bupa\totest\\'
        features, catf, numf, uf, f2change, outcome_label, desired_outcome, nbr_features, protectf, data_lab0, data_lab1 = get_bupa_user_constraints(datasetdf)
    else:
        # readpath = r'/home/~/folds/movie/' # for ubuntu
        # writepath = r'/home/~/folds/totest/' # for ubuntu
        readpath = r'C:\Users\laboratorio\Documents\Github\UFCE\folds\movie\\'
        writepath = r'C:\Users\laboratorio\Documents\Github\UFCE\folds\movie\totest\\'
        features, catf, numf, uf, f2change, outcome_label, desired_outcome, nbr_features, protectf, data_lab0, data_lab1 = get_movie_user_constraints(datasetdf)
    # Take top mutual information sharing pairs of features
    MI_FP = ufc.get_top_MI_features(X, features)
    # with open('feature_pairs.json', 'r') as file:
    #     MI_FP = json.load(file)
    print(f'\t Top-5 Mutually-informed feature paris:{MI_FP[:5]}')
    # Calling here the test-folds method from testfolds
    #create_folds(df, readpath)  # HPC may not allow write, use already generated testsets on cpu.system
    
    scaler = StandardScaler()  # check verify the scaler
    scaler=scaler.fit(Xtrain[:])
    
    for model in models.keys():
        print(f'\t\t Machine Learning blackbox:{model}')
        print(f'\t\t Cross Validation accuracy: %.3f +/- %.3f' % (models[model][0], models[model][1]))
        #predict_X_test_folds(model, readpath, writepath, outcome_label)  # if you dont have write permissions then comment out it.
        # path = r'/home/~/folds/' # use your path
        #path = r'C:\Users\~\folds\bank\totest'
        testfolds = glob.glob(os.path.join(writepath, "*.csv"))
        cfmethods = ['UFCE1', 'UFCE2', 'UFCE3', 'DiCE', 'DiCE-UF', 'AR']
        methodtimes = dict()
        no_cf = 5
        k = 1
        bb = model
        desired_outcome = desired_outcome
        protectf = protectf
        mnames = ['ufce1','ufce2', 'ufce3', 'dice', 'dice-uf', 'ar']
        meandf_jproximity = pd.DataFrame(columns=mnames)
        stddf_jproximity = pd.DataFrame(columns=mnames)
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
        for i, file in enumerate(testfolds[:]):
            print(f'\t\t\t Test Fold:{i} ')
            try:
                testset = pd.read_csv(file)
            except pd.io.common.EmptyDataError:
                print('File is empty or not found')
            else:
                for i, method in enumerate(cfmethods):
                    print(f'\t\t\t\t Method: {method}  --------------')
                    # k = 30
                    if method == 'DiCE':
                        dicecfs, idx, methodtimes[i], flag1 = dice_cfexp(datasetdf, testset[:], numf, f2change, no_cf, bb, uf, outcome_label)
                        if dicecfs.empty != True:
                            dicecfs = dicecfs.drop(['Chance of Admit', 'proximity'], axis=1)
                        dicetestdata = testset.loc[idx]
                        dicetestdata = dicetestdata.reset_index(drop=True)
                        dicecfs = dicecfs.reset_index(drop=True)
                        # print(f'\t\t\t\t Counterfactuals \t:{dicecfs.values}')
                    elif method == 'DiCE-UF':
                        dicecfs_in, idx_in, methodtimes[i], flag1 = dice_cfexp_in(datasetdf, testset[:], numf, f2change, no_cf, bb, uf, outcome_label)
                        if dicecfs_in.empty != True:
                            dicecfs_in = dicecfs_in.drop(['Chance of Admit', 'proximity'], axis=1)
                        dicetestdata_in = testset.loc[idx_in]
                        dicetestdata_in = dicetestdata_in.reset_index(drop=True)
                        dicecfs_in = dicecfs_in.reset_index(drop=True)
                        # print(f'\t\t\t\t Counterfactuals \t:{dicecfs.values}')
                    elif method == 'AR':
                        # if model == mlp:
                        #     pass
                        # else:
                        arcfs, methodtimes[i], idx = ar_cfexp(X, numf, bb, testset[:], uf, scaler, Xtrain, f2change)
                        artestdata = testset.loc[idx]
                        artestdata = artestdata.reset_index(drop=True)
                        arcfs = arcfs.reset_index(drop=True)
                        #     # print(f'\t\t\t\t Counterfactual \t:{arcfs.values}')
                    elif method == 'UFCE1':
                        # step = {'Age':1, 'Experience':1, 'Income':1, 'CCAvg':.1, 'Family':1, 'Education':1, 'Mortgage':1, 'Securities Account':1,'CD Account':1,'Online':1, 'CreditCard':1}
                        # step = {'Mcv': 1, 'Alkphos': 1, 'Sgpt': 1, 'Sgot': 1, 'Gammagt': 1, 'Drinks': 1}
                        step = {'GRE Score': 1, 'TOEFL Score': 1, 'University Rating': 1, 'SOP': 1, 'LOR': 1, 'CGPA': 0.1, 'Research': 1}
                        # step = {'fixed acidity': 0.5, 'residual sugar': 0.5, 'free sulfur dioxide': 1.0, 'total sulfur dioxide': 1.0,
                            # 'pH': 0.5, 'alcohol': 0.5, 'density': 0.1, 'volatile acidity': 0.10, 'citric acid': 0.1}
                        # step = {'Production expense': 3, 'Num_multiplex': 3, 'Multiplex coverage': 0.2, 'Movie_length': 5,
                            # 'Lead_ Actor_Rating': 1.0, 'Lead_Actress_rating': 1.0,
                            # 'Director_rating': 1.0, 'Producer_rating': 1.0, 'Genre': 1, 'Collection': 500, 'Budget':3000}
                        onecfs, methodtimes[i], idx = sfexp(X, data_lab1, testset[:], uf, step, f2change, numf, catf, bb, desired_outcome, no_cf, features)
                        onetestdata = testset.loc[idx]
                        onetestdata = onetestdata.reset_index(drop=True)
                        onecfs = onecfs.reset_index(drop=True)

                    elif method == 'UFCE2':
                        twocfs, methodtimes[i], idx = dfexp(X, data_lab1, testset[:], uf, MI_FP[:5], numf, catf, f2change, protectf, bb, desired_outcome, no_cf, features)
                        # print("two cfs", len(twocfs))
                        twotestdata = testset.loc[idx]
                        twotestdata = twotestdata.reset_index(drop=True)
                        twocfs = twocfs.reset_index(drop=True)

                    else:
                        threecfs, methodtimes[i], idx = tfexp(X, data_lab1, testset[:], uf, MI_FP[:5], numf, catf, f2change, protectf, bb, desired_outcome, no_cf, features)#features
                        threetestdata = testset.loc[idx]
                        threetestdata = threetestdata.reset_index(drop=True)
                        threecfs = threecfs.reset_index(drop=True)

                mmeans, mstds = [], []
                mmeans, mstds = Catproximity(onecfs, onetestdata, twocfs, twotestdata, threecfs, threetestdata, dicecfs, dicecfs_in, dicetestdata_in, arcfs, Xtest, catf)
                df = pd.DataFrame(data=[mmeans], columns=mnames)
                meandf_catproximity = pd.concat([meandf_catproximity, df], ignore_index=True, axis=0)
                df = pd.DataFrame(data=[mstds], columns=mnames)
                stddf_catproximity = pd.concat([stddf_catproximity, df], ignore_index=True, axis=0)
                mmeans.extend(mstds)
                df = pd.DataFrame(data=[mmeans], columns=cols)
                catproxidf = pd.concat([catproxidf, df], ignore_index=True, axis=0)
                # continuous proximity
                mmeans, mstds = [], []
                mmeans, mstds = Contproximity(onecfs, onetestdata, twocfs, twotestdata, threecfs, threetestdata, dicecfs, dicecfs_in, dicetestdata_in, arcfs, Xtest, numf)
                df = pd.DataFrame(data=[mmeans], columns=mnames)
                meandf_contproximity = pd.concat([meandf_contproximity, df], ignore_index=True, axis=0)
                df = pd.DataFrame(data=[mstds], columns=mnames)
                stddf_contproximity = pd.concat([stddf_contproximity, df], ignore_index=True, axis=0)
                mmeans.extend(mstds)
                df = pd.DataFrame(data=[mmeans], columns=cols)
                contproxidf = pd.concat([contproxidf, df], ignore_index=True, axis=0)
                # sparsity
                mmeans, mstds = [], []
                mmeans, mstds = Sparsity(onecfs, onetestdata, twocfs, twotestdata, threecfs, threetestdata, dicecfs, dicecfs_in, dicetestdata_in, arcfs, Xtest, numf)
                df = pd.DataFrame(data=[mmeans], columns=mnames)
                meandf_sparsity = pd.concat([meandf_sparsity, df], ignore_index=True, axis=0)
                df = pd.DataFrame(data=[mstds], columns=mnames)
                stddf_sparsity = pd.concat([stddf_sparsity, df], ignore_index=True, axis=0)
                mmeans.extend(mstds)
                df = pd.DataFrame(data=[mmeans], columns=cols)
                spardf = pd.concat([spardf, df], ignore_index=True, axis=0)
                # actionability
                # returns number of action cfs, yet to convert into percentage
                mmeans, mstds = [], []
                mmeans, mstds = Actionability(onecfs, onetestdata, twocfs, twotestdata, threecfs, threetestdata, dicecfs, dicecfs_in, dicetestdata_in, arcfs, Xtest, features, f2change, uf)
                df = pd.DataFrame(data=[mmeans], columns=mnames)
                meandf_actionability = pd.concat([meandf_actionability, df], ignore_index=True, axis=0)
                df = pd.DataFrame(data=[mstds], columns=mnames)
                stddf_actionability = pd.concat([stddf_actionability, df], ignore_index=True, axis=0)
                mmeans.extend(mstds)
                df = pd.DataFrame(data=[mmeans], columns=cols)
                actdf = pd.concat([actdf, df], ignore_index=True, axis=0)
                mmeans, mstds = [], []
                mmeans, mstds = Plausibility(onecfs, onetestdata, twocfs, twotestdata, threecfs, threetestdata, dicecfs, dicecfs_in, dicetestdata_in, arcfs, Xtest, Xtrain)
                df = pd.DataFrame(data=[mmeans], columns=mnames)
                meandf_plausibility = pd.concat([meandf_plausibility, df], ignore_index=True, axis=0)
                df = pd.DataFrame(data=[mstds], columns=mnames)
                stddf_plausibility = pd.concat([stddf_plausibility, df], ignore_index=True, axis=0)
                mmeans.extend(mstds)
                df = pd.DataFrame(data=[mmeans], columns=cols)
                plausdf = pd.concat([plausdf, df], ignore_index=True, axis=0)
                mmeans, mstds = [], []
                mmeans, mstds = Feasibility(onecfs, onetestdata, twocfs, twotestdata, threecfs, threetestdata, dicecfs, dicecfs_in, dicetestdata_in, arcfs, Xtest, Xtrain, features, f2change, bb, desired_outcome, uf)
                df = pd.DataFrame(data=[mmeans], columns=mnames)
                meandf_feasibility = pd.concat([meandf_feasibility, df], ignore_index=True, axis=0)
                df = pd.DataFrame(data=[mstds], columns=mnames)
                stddf_feasibility = pd.concat([stddf_feasibility, df], ignore_index=True, axis=0)
                mmeans.extend(mstds)
                df = pd.DataFrame(data=[mmeans], columns=cols)
                feasidf = pd.concat([feasidf, df], ignore_index=True, axis=0)

        print(f'\t\t\t\t-----fold_mean_values of all evaluation metrics----')

        print(f'Mean and St.dev of Joint-Proximity:', jproxidf.to_latex(float_format="{:0.2f}".format))
        print(f'Mean and St.dev of Cat-Proximity:', catproxidf.to_latex(float_format="{:0.2f}".format))
        print(f'Mean and St.dev of Cont-Proximity:', contproxidf.to_latex(float_format="{:0.2f}".format))
        print(f'Mean and St.dev of Sparsity:', spardf.to_latex(float_format="{:0.2f}".format))
        print(f'Mean and St.dev of Actionability:', actdf.to_latex(float_format="{:0.2f}".format))
        print(f'Mean and St.dev of Plausibility:', plausdf.to_latex(float_format="{:0.2f}".format))
        print(f'Mean and St.dev of Feasibility:', feasidf.to_latex(float_format="{:0.2f}".format))


        methods = ['UFCE1','UFCE2','UFCE3','DiCE', 'DiCE-UF', 'AR']
        x_pos = np.arange(len(methods))
        #time plot
        tt = [i for i in methodtimes.values()] #[dice, ar, one, two, three, cem]
        #ntt = [j / max(tt) for j in tt]
        serror = [0, 0, 0, 0, 0, 0] #[dice_std, ar_std, one_std, two_std, three_std, cem_std]

        # path = os.getcwd() # for general cases when workign directly is accessible
        # path = r'/home/~/'

        path = os.path.join(readpath, 'results')
        path1 = os.path.join(path, 'time.png')
        ufc.barplot(methods, tt, x_pos, serror, 'Time', 'lower is the better', path1, save=True)

        path1 = os.path.join(path, 'jproximity.png')
        ufc.barplot(methods, meandf_jproximity.mean().values, x_pos, np.std(meandf_jproximity).values, 'Joint-Proximity', 'lower is the better', path1, save=True)

        path1 = os.path.join(path, 'cont_proximity.png')
        ufc.barplot(methods, meandf_contproximity.mean().values, x_pos, np.std(meandf_contproximity).values, 'Cont-Proximity', 'lower is the better', path1, save=True)

        path1 = os.path.join(path, 'cat_proximity.png')
        ufc.barplot(methods, meandf_catproximity.mean().values, x_pos, np.std(meandf_catproximity).values, 'Cat-Proximity', 'lower is the better', path1, save=True)

        path1 = os.path.join(path, 'sparsity.png')
        ufc.barplot(methods, meandf_sparsity.mean().values, x_pos, np.std(meandf_sparsity).values, 'Sparsity', 'lower is the better', path1, save=True)

        path1 = os.path.join(path, 'actionability.png')
        ufc.barplot(methods, meandf_actionability.mean().values, x_pos, np.std(meandf_actionability).values, 'Actionability', 'higher is the better', path1, save=True)

        path1 = os.path.join(path, 'plausibility.png')
        ufc.barplot(methods, meandf_plausibility.mean().values, x_pos, np.std(meandf_plausibility).values, 'Plausibility', 'higher is the better', path1, save=True)

        path1 = os.path.join(path, 'feasibility.png')
        ufc.barplot(methods, meandf_feasibility.mean().values, x_pos, np.std(meandf_feasibility).values, 'Feasibility', 'higher is the better', path1, save=True)
