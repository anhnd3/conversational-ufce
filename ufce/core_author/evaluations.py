from sklearn.preprocessing import StandardScaler
import numpy as np
from ufce import UFCE
ufc = UFCE()


def _normalize_cont_metric(method='eucl'):
    metric = str(method).strip().lower()
    if metric in ('eucl', 'euclidean'):
        return 'euclidean'
    if metric == 'mad':
        return 'mad'
    raise ValueError(f"Unsupported Contproximity method: {method}")

# # joint-proximty
def Joint_proximity(oneF_cfdf, twoF_cfdf, threeF_cfdf, dice_cfs, ar_cfs, X_test, numf, catf):
    """
    :param oneF_cfdf:
    :param twoF_cfdf:
    :param threeF_cfdf:
    :param dice_cfs:
    :param ar_cfs:
    :param X_test:
    :param numf:
    :param catf:
    :return:
    """
    one_e2j = []
    for x in range(len(oneF_cfdf)):
        one_e2j.append(ufc.distance_e2j(X_test[x:x+1], oneF_cfdf[x:x+1], numf, catf, ratio_cont=None, agg=None))
    two_e2j = []
    for x in range(len(twoF_cfdf)):
        two_e2j.append(ufc.distance_e2j(X_test[x:x+1], twoF_cfdf[x:x+1], numf, catf, ratio_cont=None, agg=None))
    three_e2j = []
    for x in range(len(threeF_cfdf)):
        three_e2j.append(ufc.distance_e2j(X_test[x:x+1], threeF_cfdf[x:x+1], numf, catf, ratio_cont=None, agg=None))
    dice_e2j = []
    for x in range(len(dice_cfs)):
        dice_e2j.append(ufc.distance_e2j(X_test[x:x+1], dice_cfs[x:x+1], numf, catf, ratio_cont=None, agg=None))
    ar_e2j = []
    for x in range(len(ar_cfs)):
        ar_e2j.append(ufc.distance_e2j(X_test[x:x+1], ar_cfs[x:x+1], numf, catf, ratio_cont=None, agg=None))
    one = np.mean(one_e2j)
    two = np.mean(two_e2j)
    three = np.mean(three_e2j)
    dice = np.mean(dice_e2j)
    ar = np.mean(ar_e2j)
    means = [one, two, three, dice, ar]
    #mmeans = [i/max(means) for i in means]
    dice_std = np.std(dice_e2j) / np.sqrt(np.size(dice_e2j))
    ar_std = np.std(ar_e2j) / np.sqrt(np.size(ar_e2j))
    one_std = np.std(one_e2j) / np.sqrt(np.size(one_e2j))
    two_std = np.std(two_e2j) / np.sqrt(np.size(two_e2j))
    three_std = np.std(three_e2j) / np.sqrt(np.size(three_e2j))
    stds = [one_std, two_std, three_std, dice_std, ar_std]
    #mstds = [j / max(stds) for j in stds]
    return means, stds

def Catproximity(oneF_cfdf, testout1, twoF_cfdf, testout2, threeF_cfdf, testout3, dice_cfs, dicecfs_in, dicetestdata_in, ar_cfs, X_test, catf):
    """
    :param oneF_cfdf:
    :param testout1:
    :param twoF_cfdf:
    :param testout2:
    :param threeF_cfdf:
    :param testout3:
    :param dice_cfs:
    :param ar_cfs:
    :param X_test:
    :param catf:
    :return:
    """
    one_e2j = []
    for x in range(len(testout1)):
        one_e2j.append(ufc.categorical_distance(testout1[x:x+1], oneF_cfdf[x:x+1], catf, metric='jaccard', agg=None))
    two_e2j = []
    for x in range(len(testout2)):
        two_e2j.append(ufc.categorical_distance(testout2[x:x+1], twoF_cfdf[x:x+1], catf, metric='jaccard', agg=None))
    three_e2j = []
    for x in range(len(testout3)):
        three_e2j.append(ufc.categorical_distance(testout3[x:x+1], threeF_cfdf[x:x+1], catf, metric='jaccard', agg=None))
    dice_e2j = []
    for x in range(len(dice_cfs)):
        dice_e2j.append(ufc.categorical_distance(X_test[x:x+1], dice_cfs[x:x+1], catf, metric='jaccard', agg=None))
    dice_e2j_in = []
    for x in range(len(dicecfs_in)):
        dice_e2j_in.append(ufc.categorical_distance(dicetestdata_in[x:x+1], dicecfs_in[x:x+1], catf, metric='jaccard', agg=None))
    ar_e2j = []
    for x in range(len(ar_cfs)):
        ar_e2j.append(ufc.categorical_distance(X_test[x:x+1], ar_cfs[x:x+1], catf, metric='jaccard', agg=None))
    one = np.mean(one_e2j)
    two = np.mean(two_e2j)
    three = np.mean(three_e2j)
    dice = np.mean(dice_e2j)
    dice_in = np.mean(dice_e2j_in)
    ar = np.mean(ar_e2j)
    means = [one, two, three, dice, dice_in, ar]
    #mmeans = [i/max(means) for i in means]
    dice_std = np.std(dice_e2j) / np.sqrt(np.size(dice_e2j))
    dice_std_in = np.std(dice_e2j_in) / np.sqrt(np.size(dice_e2j_in))
    ar_std = np.std(ar_e2j) / np.sqrt(np.size(ar_e2j))
    one_std = np.std(one_e2j) / np.sqrt(np.size(one_e2j))
    two_std = np.std(two_e2j) / np.sqrt(np.size(two_e2j))
    three_std = np.std(three_e2j) / np.sqrt(np.size(three_e2j))
    stds = [one_std, two_std, three_std, dice_std, dice_std_in, ar_std]
    #mstds = [j / max(stds) for j in stds]
    return means, stds

def Contproximity(
    oneF_cfdf,
    testout1,
    twoF_cfdf,
    testout2,
    threeF_cfdf,
    testout3,
    dice_cfs,
    dicecfs_in,
    dicetestdata_in,
    ar_cfs,
    X_test,
    numf,
    method='eucl',
    X=None,
):
    """
    :param oneF_cfdf:
    :param testout1:
    :param twoF_cfdf:
    :param testout2:
    :param threeF_cfdf:
    :param testout3:
    :param dice_cfs:
    :param ar_cfs:
    :param X_test:
    :param numf:
    :return:
    """
    metric = _normalize_cont_metric(method)
    if metric == 'mad' and X is None:
        raise ValueError("X is required when method='mad'")

    one_e2j = []
    for x in range(len(testout1)):
        one_e2j.append(ufc.continuous_distance(testout1[x:x+1], oneF_cfdf[x:x+1], numf, metric=metric, X=X, agg=None))
    two_e2j = []
    for x in range(len(testout2)):
        two_e2j.append(ufc.continuous_distance(testout2[x:x+1], twoF_cfdf[x:x+1], numf, metric=metric, X=X, agg=None))
    three_e2j = []
    for x in range(len(testout3)):
        three_e2j.append(ufc.continuous_distance(testout3[x:x+1], threeF_cfdf[x:x+1], numf, metric=metric, X=X, agg=None))
    dice_e2j = []
    for x in range(len(dice_cfs)):
        dice_e2j.append(ufc.continuous_distance(X_test[x:x + 1], dice_cfs[x:x + 1], numf, metric=metric, X=X, agg=None))
    dice_e2j_in = []
    for x in range(len(dicecfs_in)):
        dice_e2j_in.append(ufc.continuous_distance(dicetestdata_in[x:x+1], dicecfs_in[x:x+1], numf, metric=metric, X=X, agg=None))
    ar_e2j = []
    for x in range(len(ar_cfs)):
        ar_e2j.append(ufc.continuous_distance(X_test[x:x+1], ar_cfs[x:x+1], numf, metric=metric, X=X, agg=None))
    one = np.mean(one_e2j)
    two = np.mean(two_e2j)
    three = np.mean(three_e2j)
    dice = np.mean(dice_e2j)
    dice_in = np.mean(dice_e2j_in)
    ar = np.mean(ar_e2j)
    means = [one, two, three, dice, dice_in, ar]
    #mmeans = [i / max(means) for i in means]
    dice_std = np.std(dice_e2j) / np.sqrt(np.size(dice_e2j))
    dice_std_in = np.std(dice_e2j_in) / np.sqrt(np.size(dice_e2j_in))
    ar_std = np.std(ar_e2j) / np.sqrt(np.size(ar_e2j))
    one_std = np.std(one_e2j) / np.sqrt(np.size(one_e2j))
    two_std = np.std(two_e2j) / np.sqrt(np.size(two_e2j))
    three_std = np.std(three_e2j) / np.sqrt(np.size(three_e2j))
    stds = [one_std, two_std, three_std, dice_std, dice_std_in, ar_std]
    #mstds = [j / max(stds) for j in stds]
    return means, stds

def Sparsity(oneF_cfdf, testout1, twoF_cfdf, testout2, threeF_cfdf, testout3, dice_cfs, dicecfs_in, dicetestdata_in, ar_cfs, X_test, numf):
    """
    :param oneF_cfdf:
    :param testout1:
    :param twoF_cfdf:
    :param testout2:
    :param threeF_cfdf:
    :param testout3:
    :param dice_cfs:
    :param ar_cfs:
    :param X_test:
    :param numf:
    :return:
    """

    one_sparsity_d, one_val = ufc.sparsity_count(oneF_cfdf, testout1, numf, numf) #last two arguments are not used
    two_sparsity_d, two_val = ufc.sparsity_count(twoF_cfdf, testout2, numf,  numf)
    three_sparsity_d, three_val = ufc.sparsity_count(threeF_cfdf, testout3, numf, numf)
    dice_sparsity_d, dice_val = ufc.sparsity_count(dice_cfs, X_test, numf, numf)
    dice_sparsity_d_in, dice_val_in = ufc.sparsity_count(dicecfs_in, dicetestdata_in, numf, numf)
    ar_sparsity_d, ar_val = ufc.sparsity_count(ar_cfs, X_test, numf, numf)
    dice = np.array(list(dice_sparsity_d.values())).mean()
    dice_in = np.array(list(dice_sparsity_d_in.values())).mean()
    ar = np.array(list(ar_sparsity_d.values())).mean()
    one = np.array(list(one_sparsity_d.values())).mean()
    two = np.array(list(two_sparsity_d.values())).mean()
    three = np.array(list(three_sparsity_d.values())).mean()
    means = [one, two, three, dice, dice_in, ar]
    #mmeans = [i / max(means) for i in means]
    dice_std = np.array(list(dice_sparsity_d.values())).std()
    dice_std_in = np.array(list(dice_sparsity_d_in.values())).std()
    ar_std = np.array(list(ar_sparsity_d.values())).std()
    one_std = np.array(list(one_sparsity_d.values())).std()
    two_std = np.array(list(two_sparsity_d.values())).std()
    three_std = np.array(list(three_sparsity_d.values())).std()
    stds = [one_std, two_std, three_std, dice_std, dice_std_in, ar_std]
    #mstds = [j / max(stds) for j in stds]
    return means, stds

def Actionability(oneF_cfdf, testout1, twoF_cfdf, testout2, threeF_cfdf, testout3, dice_cfs, dicecfs_in, dicetestdata_in, ar_cfs, X_test, features, f2change, uf):
    """
    :param oneF_cfdf:
    :param testout1:
    :param twoF_cfdf:
    :param testout2:
    :param threeF_cfdf:
    :param testout3:
    :param dice_cfs:
    :param ar_cfs:
    :param X_test:
    :param features:
    :param f2change:
    :return:
    """
    idx = 0
    dice, flag, ids, one_actionability_d = ufc.actionability(dice_cfs, X_test, features, f2change, idx, uf, method = "other")
    dice_in, flag, ids, one_actionability_d_in = ufc.actionability(dicecfs_in, dicetestdata_in, features, f2change, idx, uf, method="other")
    ar, flag, ids, two_actionability_d = ufc.actionability(ar_cfs, X_test, features, f2change, idx, uf, method="other")
    one, flag, ids, three_actionability_d = ufc.actionability(oneF_cfdf, testout1, features, f2change, idx, uf, method="other")
    # print("one actionable", one[:2])
    two, flag, ids, dice_actionability_d = ufc.actionability(twoF_cfdf, testout2, features, f2change, idx, uf, method="other")
    # print("two actionable", two[:2])
    three, flag, ids, ar_actionability_d = ufc.actionability(threeF_cfdf, testout3, features, f2change, idx, uf, method="other")
    # print("three actionable", three[:2])
    # dice = np.array(list(dice_actionability_d.values())).mean()
    # ar = np.array(list(ar_actionability_d.values())).mean()
    # one = np.array(list(one_actionability_d.values())).mean()
    # two = np.array(list(two_actionability_d.values())).mean()
    # three = np.array(list(three_actionability_d.values())).mean()
    means = [len(one), len(two), len(three), len(dice), len(dice_in), len(ar)] # divider provide equal impact
    #mmeans = [i / max(means) for i in means]
    # dice_std = np.array(list(dice_actionability_d.values())).std()
    # ar_std = np.array(list(ar_actionability_d.values())).std()
    # one_std = np.array(list(one_actionability_d.values())).std()
    # two_std = np.array(list(two_actionability_d.values())).std()
    # three_std = np.array(list(three_actionability_d.values())).std()
    # stds = [one_std, two_std, three_std, dice_std, ar_std]
    stds = [0, 0, 0, 0, 0, 0]
    #mstds = [j / max(stds) for j in stds]
    return means, stds

def Plausibility(oneF_cfdf, testout1, twoF_cfdf, testout2, threeF_cfdf, testout3, dice_cfs, dicecfs_in, dicetestdata_in, ar_cfs, X_test, X_train):
    """
    :param oneF_cfdf:
    :param twoF_cfdf:
    :param threeF_cfdf:
    :param dice_cfs:
    :param ar_cfs:
    :param X_test:
    :param X_train:
    :return:
    """
    idx = 0
    one_val = ufc.implausibility(oneF_cfdf, testout1, X_train[:], len(oneF_cfdf), idx)
    # print('in 2f call')
    two_val = ufc.implausibility(twoF_cfdf, testout2, X_train[:], len(twoF_cfdf), idx)
    # print('end 2f call')
    # print("in evaluationsssss", testout3, len(threeF_cfdf))
    three_val = ufc.implausibility(threeF_cfdf, testout3, X_train[:], len(threeF_cfdf), idx)
    dice_val = ufc.implausibility(dice_cfs, X_test, X_train[:], len(dice_cfs), idx)
    dice_val_in = ufc.implausibility(dicecfs_in, dicetestdata_in, X_train[:], len(dicecfs_in), idx)
    ar_val = ufc.implausibility(ar_cfs, X_test, X_train[:], len(ar_cfs), idx)
    # dice = np.array(list(dice_plausibility_d.values())).mean()
    # ar = np.array(list(ar_plausibility_d.values())).mean()
    # one = np.array(list(one_plausibility_d.values())).mean()
    # two = np.array(list(two_plausibility_d.values())).mean()
    # three = np.array(list(three_plausibility_d.values())).mean()
    means = [one_val, two_val, three_val, dice_val, dice_val_in, ar_val]
    #mmeans = [i / max(means) for i in means]
    # dice_std = np.array(list(dice_plausibility_d.values())).std()
    # ar_std = np.array(list(ar_plausibility_d.values())).std()
    # one_std = np.array(list(one_plausibility_d.values())).std()
    # two_std = np.array(list(two_plausibility_d.values())).std()
    # three_std = np.array(list(three_plausibility_d.values())).std()
    # stds = [one_std, two_std, three_std, dice_std, ar_std]
    stds = [0, 0, 0, 0, 0, 0]
    #mstds = [j / max(stds) for j in stds]
    return means, stds

def Feasibility(onecfs, testout1, twocfs, testout2, threecfs, testout3, dice_cfs, dicecfs_in, dicetestdata_in, ar_cfs, X_test, X_train, features, f2change, bb, desired_outcome, uf):
    """
    :param oneF_cfdf:
    :param twoF_cfdf:
    :param threeF_cfdf:
    :param dice_cfs:
    :param ar_cfs:
    :param X_test:
    :param X_train:
    :param features:
    :param f2change:
    :param bb: black box model
    :param desired_outcome:
    :param outcome_label:
    :return:
    """
    # print("DiCE caledddddd-------------------")
    idx = 0
    d_feas, temp = ufc.feasibility(X_test, dice_cfs, X_train, features, f2change, bb, desired_outcome, uf, idx, method = "other")
    d_feas_in, temp = ufc.feasibility(dicetestdata_in, dicecfs_in, X_train, features, f2change, bb, desired_outcome, uf, idx, method="other")
    a_feas, temp = ufc.feasibility(X_test, ar_cfs, X_train, features, f2change, bb, desired_outcome, uf, idx, method="other")
    o_feas, temp = ufc.feasibility(testout1, onecfs, X_train, features, f2change, bb, desired_outcome, uf, idx, method="other")
    t_feas, temp = ufc.feasibility(testout2, twocfs, X_train, features, f2change, bb, desired_outcome, uf, idx, method="other")
    th_feas, temp = ufc.feasibility(testout3, threecfs, X_train, features, f2change, bb, desired_outcome, uf, idx, method="other")
    means = [o_feas, t_feas, th_feas, d_feas, d_feas_in, a_feas]
    stds = [0, 0, 0, 0, 0, 0]
    return means, stds
