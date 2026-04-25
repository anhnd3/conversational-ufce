from sklearn.preprocessing import StandardScaler
import numpy as np
from ufce import UFCE
ufc = UFCE()

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

def Contproximity(oneF_cfdf, testout1,  twoF_cfdf, testout2, threeF_cfdf, testout3, dice_cfs, dicecfs_in, dicetestdata_in, ar_cfs, X_test, numf):
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
    one_e2j = []
    for x in range(len(testout1)):
        one_e2j.append(ufc.continuous_distance(testout1[x:x+1], oneF_cfdf[x:x+1], numf, metric='euclidean', agg=None))
    two_e2j = []
    for x in range(len(testout2)):
        two_e2j.append(ufc.continuous_distance(testout2[x:x+1], twoF_cfdf[x:x+1], numf, metric='euclidean', agg=None))
    three_e2j = []
    for x in range(len(testout3)):
        three_e2j.append(ufc.continuous_distance(testout3[x:x+1], threeF_cfdf[x:x+1], numf, metric='euclidean', agg=None))
    dice_e2j = []
    for x in range(len(dice_cfs)):
        dice_e2j.append(ufc.continuous_distance(X_test[x:x + 1], dice_cfs[x:x + 1], numf, metric='euclidean', agg=None))
    dice_e2j_in = []
    for x in range(len(dicecfs_in)):
        dice_e2j_in.append(ufc.continuous_distance(dicetestdata_in[x:x+1], dicecfs_in[x:x+1], numf, metric='euclidean', agg=None))
    ar_e2j = []
    for x in range(len(ar_cfs)):
        ar_e2j.append(ufc.continuous_distance(X_test[x:x+1], ar_cfs[x:x+1], numf, metric='euclidean', agg=None))
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

def Actionability(
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
    features,
    f2change,
    uf,
    return_details=False,
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
    :param features:
    :param f2change:
    :return:
    """
    idx = 0
    dice, flag, ids_dice, one_actionability_d = ufc.actionability(dice_cfs, X_test, features, f2change, idx, uf, method="other")
    dice_in, flag, ids_dice_in, one_actionability_d_in = ufc.actionability(dicecfs_in, dicetestdata_in, features, f2change, idx, uf, method="other")
    ar, flag, ids_ar, two_actionability_d = ufc.actionability(ar_cfs, X_test, features, f2change, idx, uf, method="other")
    one, flag, ids_one, three_actionability_d = ufc.actionability(oneF_cfdf, testout1, features, f2change, idx, uf, method="other")
    # print(one.attrs.get("debug_act"))

    # print("one actionable", one[:2])
    two, flag, ids_two, dice_actionability_d = ufc.actionability(twoF_cfdf, testout2, features, f2change, idx, uf, method="other")
    # print(two.attrs.get("debug_act"))
    
    # print("two actionable", two[:2])
    three, flag, ids_three, ar_actionability_d = ufc.actionability(threeF_cfdf, testout3, features, f2change, idx, uf, method="other")
    # print(three.attrs.get("debug_act"))
    
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
    details = {
        "legend": "A = passes actionability constraints",
        "UFCE1": {"count": int(len(one)), "passed_idx": [int(i) for i in ids_one], "n_pairs_input": int(len(testout1))},
        "UFCE2": {"count": int(len(two)), "passed_idx": [int(i) for i in ids_two], "n_pairs_input": int(len(testout2))},
        "UFCE3": {"count": int(len(three)), "passed_idx": [int(i) for i in ids_three], "n_pairs_input": int(len(testout3))},
        "DICE": {"count": int(len(dice)), "passed_idx": [int(i) for i in ids_dice], "n_pairs_input": int(len(X_test))},
        "DICE_IN": {"count": int(len(dice_in)), "passed_idx": [int(i) for i in ids_dice_in], "n_pairs_input": int(len(dicetestdata_in))},
        "AR": {"count": int(len(ar)), "passed_idx": [int(i) for i in ids_ar], "n_pairs_input": int(len(X_test))},
    }
    #mstds = [j / max(stds) for j in stds]
    if return_details:
        return means, stds, details
    return means, stds

def Plausibility(
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
    X_train,
    return_details=False,
    use_standard_scaler=True,
    lof_space_label="raw_lof",
):
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
    one_val = ufc.implausibility(
        oneF_cfdf,
        testout1,
        X_train[:],
        len(oneF_cfdf),
        idx,
        method_name="UFCE1",
        return_details=return_details,
        use_standard_scaler=use_standard_scaler,
        lof_space_label=lof_space_label,
    )
    # print('in 2f call')
    two_val = ufc.implausibility(
        twoF_cfdf,
        testout2,
        X_train[:],
        len(twoF_cfdf),
        idx,
        method_name="UFCE2",
        return_details=return_details,
        use_standard_scaler=use_standard_scaler,
        lof_space_label=lof_space_label,
    )
    # print('end 2f call')
    # print("in evaluationsssss", testout3, len(threeF_cfdf))
    three_val = ufc.implausibility(
        threeF_cfdf,
        testout3,
        X_train[:],
        len(threeF_cfdf),
        idx,
        method_name="UFCE3",
        return_details=return_details,
        use_standard_scaler=use_standard_scaler,
        lof_space_label=lof_space_label,
    )
    dice_val = ufc.implausibility(
        dice_cfs,
        X_test,
        X_train[:],
        len(dice_cfs),
        idx,
        method_name="DICE",
        return_details=return_details,
        use_standard_scaler=use_standard_scaler,
        lof_space_label=lof_space_label,
    )
    dice_val_in = ufc.implausibility(
        dicecfs_in,
        dicetestdata_in,
        X_train[:],
        len(dicecfs_in),
        idx,
        method_name="DICE_IN",
        return_details=return_details,
        use_standard_scaler=use_standard_scaler,
        lof_space_label=lof_space_label,
    )
    ar_val = ufc.implausibility(
        ar_cfs,
        X_test,
        X_train[:],
        len(ar_cfs),
        idx,
        method_name="AR",
        return_details=return_details,
        use_standard_scaler=use_standard_scaler,
        lof_space_label=lof_space_label,
    )
    if return_details:
        one_d = one_val
        two_d = two_val
        three_d = three_val
        dice_d = dice_val
        dice_in_d = dice_val_in
        ar_d = ar_val
        one_val = int(one_d.get("count", 0))
        two_val = int(two_d.get("count", 0))
        three_val = int(three_d.get("count", 0))
        dice_val = int(dice_d.get("count", 0))
        dice_val_in = int(dice_in_d.get("count", 0))
        ar_val = int(ar_d.get("count", 0))
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
    details = {
        "legend": "P = LOF inlier",
        "UFCE1": one_d if return_details else None,
        "UFCE2": two_d if return_details else None,
        "UFCE3": three_d if return_details else None,
        "DICE": dice_d if return_details else None,
        "DICE_IN": dice_in_d if return_details else None,
        "AR": ar_d if return_details else None,
    }
    #mstds = [j / max(stds) for j in stds]
    if return_details:
        return means, stds, details
    return means, stds

def Feasibility(onecfs, testout1, twocfs, testout2, threecfs, testout3,
                dice_cfs, dicecfs_in, dicetestdata_in, ar_cfs,
                X_test, X_train, features, f2change, bb, desired_outcome, uf,
                return_details=False,
                use_standard_scaler=True,
                lof_space_label="raw_lof"):
    """
    Wrapper around UFCE.feasibility() that is robust to None returns.
    Always returns (means, stds) with means length=6.
    """

    import pandas as pd

    def _safe_call(x_test_df, cf_df, label):
        out = ufc.feasibility(
            x_test_df,
            cf_df,
            X_train,
            features,
            f2change,
            bb,
            desired_outcome,
            uf,
            0,
            method="other",
            return_details=return_details,
            use_standard_scaler=use_standard_scaler,
            lof_space_label=lof_space_label,
        )
        if out is None:
            # Legacy safety: some implementations return None; normalize here
            return 0, pd.DataFrame(columns=list(features)), {}
        # Some implementations may return only feas (int) accidentally
        if isinstance(out, (int, float)):
            return int(out), pd.DataFrame(columns=list(features)), {}
        # Normal case: (feas, temp_df)
        if isinstance(out, tuple) and len(out) == 2:
            return out[0], out[1], {}
        if isinstance(out, tuple) and len(out) == 3:
            return out[0], out[1], out[2]
        # Unknown format
        return 0, pd.DataFrame(columns=list(features)), {}

    # Call in the same order you already use
    d_feas, _, d_det = _safe_call(X_test, dice_cfs, "dice")
    d_feas_in, _, d_in_det = _safe_call(dicetestdata_in, dicecfs_in, "dice_in")
    a_feas, _, a_det = _safe_call(X_test, ar_cfs, "ar")
    o_feas, _, o_det = _safe_call(testout1, onecfs, "ufce1")
    t_feas, _, t_det = _safe_call(testout2, twocfs, "ufce2")
    th_feas, _, th_det = _safe_call(testout3, threecfs, "ufce3")

    means = [o_feas, t_feas, th_feas, d_feas, d_feas_in, a_feas]
    stds = [0, 0, 0, 0, 0, 0]
    details = {
        "legend": "F = passes feasibility constraints",
        "UFCE1": o_det,
        "UFCE2": t_det,
        "UFCE3": th_det,
        "DICE": d_det,
        "DICE_IN": d_in_det,
        "AR": a_det,
    }
    if return_details:
        return means, stds, details
    return means, stds
