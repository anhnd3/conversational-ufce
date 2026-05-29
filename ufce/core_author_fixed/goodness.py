import numpy as np

from scipy.spatial.distance import cdist, pdist
from scipy.spatial.distance import _validate_vector
from scipy.stats import median_absolute_deviation
#%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import LocalOutlierFactor

# In following some functions are customised according to our needs, the orginal source of these functions belongs to:
#"Guidotti, R. Counterfactual explanations and how to find them: literature review and benchmarking. Data Min Knowl Disc (2022). https://doi.org/10.1007/s10618-022-00831-6
    
# Begin> 3rd party adapted ///////

def nbr_valid_cf(cf_list, b, y_val, y_desired=None):
    """
    :param cf_list:
    :param b: blackbox
    :param y_val:
    :param y_desired:
    :return:
    """
    y_cf = b.predict(cf_list)
    idx = y_cf != y_val if y_desired is None else y_cf == y_desired
    val = np.sum(idx)
    return val


def perc_valid_cf(cf_list, b, y_val, k=None, y_desired=None):
    """
    :param cf_list:
    :param b:
    :param y_val:
    :param k:
    :param y_desired:
    :return:
    """
    n_val = nbr_valid_cf(cf_list, b, y_val, y_desired)
    k = len(cf_list) if k is None else k
    res = n_val / k
    return res

def nbr_actionable_cf(x, cf_list, features, variable_features):
    """
    :param x:
    :param cf_list:
    :param features:
    :param variable_features:
    :return:
    """
    nbr_actionable = 0
    for i, cf in cf_list.iterrows():
        for j in features:
            if cf[j] != x[j].values and j in variable_features:
                nbr_actionable += 1
    return nbr_actionable/cf_list.shape[0]

def number_of_diff(df1, df2):
    differences = 0
    for i in range(len(df1)):
        if not df1.loc[i, :].equals(df2.loc[i, :]):
            differences += 1
    return differences

def perc_actionable_cf(x, cf_list, variable_features, k=None):
    n_val = nbr_actionable_cf(x, cf_list, variable_features)
    k = len(cf_list) if k is None else k
    res = n_val / k
    return res


def nbr_valid_actionable_cf(x, cf_list, b, y_val, variable_features, y_desidered=None):
    y_cf = b.predict(cf_list)
    idx = y_cf != y_val if y_desidered is None else y_cf == y_desidered

    nbr_valid_actionable = 0
    nbr_features = cf_list.shape[1]
    for i, cf in enumerate(cf_list):
        if not np.array(idx)[i]:
            continue
        constraint_violated = False
        for j in range(nbr_features):
            if cf[j] != x[j] and j not in variable_features:
                constraint_violated = True
                break
        if not constraint_violated:
            nbr_valid_actionable += 1

    return nbr_valid_actionable


def perc_valid_actionable_cf(x, cf_list, b, y_val, variable_features, k=None, y_desidered=None):
    n_val = nbr_valid_actionable_cf(x, cf_list, b, y_val, variable_features, y_desidered)
    k = len(cf_list) if k is None else k
    res = n_val / k
    return res


def nbr_violations_per_cf(x, cf_list, variable_features):
    nbr_features = cf_list.shape[1]
    nbr_violations = np.zeros(len(cf_list))
    for i, cf in enumerate(cf_list):
        for j in range(nbr_features):
            if cf[j] != x[j] and j not in variable_features:
                nbr_violations[i] += 1
    return nbr_violations


def avg_nbr_violations_per_cf(x, cf_list, variable_features):
    return np.mean(nbr_violations_per_cf(x, cf_list, variable_features))


def avg_nbr_violations(x, cf_list, variable_features):
    val = np.sum(nbr_violations_per_cf(x, cf_list, variable_features))
    nbr_cf, nbr_features = cf_list.shape
    return val / (nbr_cf * nbr_features)


def mad_cityblock(u, v, mad):
    u = _validate_vector(u)
    v = _validate_vector(v)
    l1_diff = abs(u - v)
    l1_diff_mad = l1_diff / mad
    return l1_diff_mad.sum()


def continuous_distance(x, cf_list, continuous_features, metric='euclidean', X=None, agg=None):
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
        mad = median_absolute_deviation(X[:, continuous_features], axis=0)
        mad = np.array([v if v != 0 else 1.0 for v in mad])

        def _mad_cityblock(u, v):
            return mad_cityblock(u, v, mad)

        dist = cdist(x.reshape(1, -1)[:, continuous_features], cf_list[:, continuous_features], metric=_mad_cityblock)
    else:
        dist = cdist(x.loc[:, continuous_features], cf_list.loc[:, continuous_features], metric=metric)

    if agg is None or agg == 'mean':
        return np.mean(dist)

    if agg == 'max':
        return np.max(dist)

    if agg == 'min':
        return np.min(dist)


def categorical_distance(x, cf_list, categorical_features, metric='jaccard', agg=None):
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


def distance_l2j(x, cf_list, continuous_features, categorical_features, ratio_cont=None, agg=None):
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
# End> 3rd party adapted ///////

def merge_dictionaries(dict1, dict2):
    """
    :param dict1:
    :param dict2:
    :return:
    """
    merged_dictionary = {}

    for key in dict1:
        if key in dict2:
            new_value = dict1[key] + dict2[key]
        else:
            new_value = dict1[key]

        merged_dictionary[key] = new_value

    return merged_dictionary

# Begin> 3rd party adapted ///////

def distance_mh(x, cf_list, continuous_features, categorical_features, X, ratio_cont=None, agg=None):
    """
    :param x:
    :param cf_list:
    :param continuous_features:
    :param categorical_features:
    :param X:
    :param ratio_cont:
    :param agg:
    :return:
    """
    nbr_features = cf_list.shape[1]
    dist_cont = continuous_distance(x, cf_list, continuous_features, metric='mad', X=X, agg=agg)
    dist_cate = categorical_distance(x, cf_list, categorical_features, metric='hamming', agg=agg)
    if ratio_cont is None:
        ratio_continuous = len(continuous_features) / nbr_features
        ratio_categorical = len(categorical_features) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist
# End> 3rd party adapted ///////

#cf feature changes and avg change
def nbr_changes_per_cf(x, cf_list, continuous_features):
    """
    :param x:
    :param cf_list:
    :param continuous_features:
    :return:
    """
    nbr_features = cf_list.shape[1]
    nbr_changes = np.zeros(len(cf_list))
    for i, cf in cf_list.iterrows():
        for j in continuous_features:
            if cf[j] != x[j].values:
                nbr_changes[i] += 1 if j in continuous_features else 0.5
    return nbr_changes

# Begin> 3rd party adapted ///////

def avg_nbr_changes_per_cf(x, cf_list, continuous_features):
    return np.mean(nbr_changes_per_cf(x, cf_list, continuous_features))


def avg_nbr_changes(x, cf_list, nbr_features, continuous_features):
    val = np.sum(nbr_changes_per_cf(x, cf_list, continuous_features))
    nbr_cf, _ = cf_list.shape
    return val / (nbr_cf * nbr_features)


def continuous_diversity(cf_list, continuous_features, metric='euclidean', X=None, agg=None):
    """
    :param cf_list:
    :param continuous_features:
    :param metric:
    :param X:
    :param agg:
    :return:
    """
    if metric == 'mad':
        mad = median_absolute_deviation(X[:, continuous_features], axis=0)
        mad = np.array([v if v != 0 else 1.0 for v in mad])

        def _mad_cityblock(u, v):
            return mad_cityblock(u, v, mad)

        dist = pdist(cf_list[:, continuous_features], metric=_mad_cityblock)
    else:
        dist = pdist(cf_list.loc[:, continuous_features], metric=metric)

    if agg is None or agg == 'mean':
        return np.mean(dist)

    if agg == 'max':
        return np.max(dist)

    if agg == 'min':
        return np.min(dist)


def categorical_diversity(cf_list, categorical_features, metric='jaccard', agg=None):
    """
    :param cf_list:
    :param categorical_features:
    :param metric:
    :param agg:
    :return:
    """
    dist = pdist(cf_list.loc[:, categorical_features], metric=metric)

    if agg is None or agg == 'mean':
        return np.mean(dist)

    if agg == 'max':
        return np.max(dist)

    if agg == 'min':
        return np.min(dist)


def diversity_l2j(cf_list, continuous_features, categorical_features, ratio_cont=None, agg=None):
    nbr_features = cf_list.shape[1]
    dist_cont = continuous_diversity(cf_list, continuous_features, metric='euclidean', X=None, agg=agg)
    dist_cate = categorical_diversity(cf_list, categorical_features, metric='jaccard', agg=agg)
    if ratio_cont is None:
        ratio_continuous = len(continuous_features) / nbr_features
        ratio_categorical = len(categorical_features) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist


def diversity_mh(cf_list, continuous_features, categorical_features, X, ratio_cont=None, agg=None):
    nbr_features = cf_list.shape[1]
    dist_cont = continuous_diversity(cf_list, continuous_features, metric='mad', X=X, agg=agg)
    dist_cate = categorical_diversity(cf_list, categorical_features, metric='hamming', agg=agg)
    if ratio_cont is None:
        ratio_continuous = len(continuous_features) / nbr_features
        ratio_categorical = len(categorical_features) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist

# End> 3rd party adapted ///////


def count_diversity(cf_list, features, nbr_features, continuous_features):
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
                if cf_list[i:i+1][k].values != cf_list[j:j+1][k].values:
                    nbr_changes += 1 if j in continuous_features else 0.5
                #print("value change is ", nbr_changes)
    return nbr_changes / (nbr_cf * nbr_cf * nbr_features) if nbr_changes != 0 else 0.0

# Begin> 3rd party adapted ///////

def count_diversity_all(cf_list, nbr_features, continuous_features):
    """
    :param cf_list:
    :param nbr_features:
    :param continuous_features:
    :return:
    """
    return count_diversity(cf_list, range(cf_list.shape[1]), nbr_features, continuous_features)

#cf feature changes and avg change
def nbr_changes_per_cf(x, cf_list, continuous_features):
    """
    :param x:
    :param cf_list:
    :param continuous_features:
    :return:
    """
    nbr_features = cf_list.shape[1]
    nbr_changes = np.zeros(len(cf_list))
    for i, cf in cf_list.iterrows():

        for j in continuous_features:
            if cf[j] != x[j].values:
                nbr_changes[i] += 1 if j in continuous_features else 0.5
    return nbr_changes


def avg_nbr_changes_per_cf(x, cf_list, continuous_features):
    return np.mean(nbr_changes_per_cf(x, cf_list, continuous_features))

def count_diversity_all(cf_list, nbr_features, continuous_features):
    return count_diversity(cf_list, range(cf_list.shape[1]), nbr_features, continuous_features)

def euclidean_jaccard(x, A, continuous_features, categorical_features, ratio_cont=None):
    nbr_features = A.shape[1]
    dist_cont = cdist(x.reshape(1, -1)[:, continuous_features], A[:, continuous_features], metric='euclidean')
    dist_cate = cdist(x.reshape(1, -1)[:, categorical_features], A[:, categorical_features], metric='jaccard')
    if ratio_cont is None:
        ratio_continuous = len(continuous_features) / nbr_features
        ratio_categorical = len(categorical_features) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist

def lof(x, cf_list, X, scaler):
    X_train = np.vstack([x.values.reshape(1, -1), X])

    nX_train = scaler.transform(X_train) #instead of X_train
    ncf_list = scaler.transform(cf_list)

    clf = LocalOutlierFactor(n_neighbors=3, novelty=True)
    clf.fit(nX_train)
    #print("here",nX_train.shape, ncf_list.shape)
    lof_values = clf.predict(ncf_list)
    print("lof here:", lof_values)
    return np.mean(np.abs(lof_values))
# End> 3rd party adapted ///////

