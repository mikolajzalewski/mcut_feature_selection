import numpy as np
from scipy.optimize import minimize_scalar
from itertools import combinations
from tqdm.auto import tqdm


import xgboost as xgb
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import (Lasso, LinearRegression)
from sklearn.neighbors import NearestNeighbors



# Correlation method

def correlation_feature_selection(data, how_many = 10):
    corr = data.corr()['marg']
    corr = corr[corr.index != 'marg']
    corr = corr.abs().sort_values(ascending=False)
    corr = corr.head(how_many)
    corr = corr.index
    return corr.tolist()


# Heuristic method

def heuristic(k, r_cf, r_ff):
    return (k * r_cf) / np.sqrt(k + k * (k - 1) * r_ff)

def mean_correlation(indices, correlation_matrix):
    n = len(indices)
    if n == 1:
        return 0  # nie można obliczyć r_ff dla jednej zmiennej
    suma = sum(correlation_matrix[i, j] for i in indices for j in indices if i != j)
    return suma / (n * (n - 1))

def maximize_heuristic(correlation_matrix):
    n = correlation_matrix.shape[0] - 1  # liczba zmiennych oprócz 'y'
    max_n = min(n, 10)  # maksymalna liczba zmiennych w modelu
    best_value = -np.inf
    best_k = None
    best_indices = None

    for k in tqdm(range(1, max_n)):
        for indices in tqdm(combinations(range(n), k)):
            r_cf = np.mean(correlation_matrix[indices, -1])
            r_ff = mean_correlation(indices, correlation_matrix)
            value = heuristic(k, r_cf, r_ff)
            if value > best_value:
                best_value = value
                best_k = k
                best_indices = indices

    return best_value, best_k, best_indices

def heuristic_feature_selection(data):
    correlation_matrix = np.corrcoef(data, rowvar=False)
    features_names = data.columns
    value, k, indices = maximize_heuristic(correlation_matrix)
    return [features_names[i] for i in indices]


# RFE method

def RFE_feature_selection(data, accepted_p_value = 0.1):
    X = data.drop(columns='marg')
    y = data['marg']
    searching = True
    while searching:
        model = LinearRegression()
        fmodel = model.fit(X, y)
        p_values = pd.Series(fmodel.coef_, index=X.columns)
        worst_feature = p_values.idxmax()
        if p_values.max() > accepted_p_value:
            X = X.drop(columns=worst_feature)
        else:
            searching = False
    return X.columns.tolist()

from sklearn.model_selection import cross_val_score
# Sequential Forward Selection

def sequential_feature_selection(X, y, max_features=None):
    """
    Function implementing Sequential Feature Selection (SFS) for feature selection.

    :param X: DataFrame containing the features.
    :param y: Series containing the dependent variable.
    :param max_features: Maximum number of features to select.
    :return: List of names of selected features.
    """
    if max_features is None:
        max_features = X.shape[1]
    
    # Initial set of feature indices
    selected_features = []
    remaining_features = list(X.columns)
    
    # Model to use for feature evaluation
    model = LinearRegression()
    
    while len(selected_features) < max_features:
        best_score = -np.inf
        best_feature = None
        
        for feature in remaining_features:
            # Add the tested feature to the already selected
            features_to_test = selected_features + [feature]
            
            # Perform cross-validation with the current set of features
            scores = cross_val_score(model, X[features_to_test], y, cv=5)
            mean_score = scores.mean()  # Use the mean of validation results
            
            # Check if we found a new best feature
            if mean_score > best_score:
                best_score = mean_score
                best_feature = feature
        
        # If we found a new best feature, add it to the set
        if best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
        else:
            break
    
    return selected_features

# XGBoost Feature Importance

def xgboost_feature_selection(X,y, number_of_features = 10):
    model = xgb.XGBRegressor()
    model.fit(X, y)
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    return X.columns[indices][:number_of_features].tolist()

# Lassso Feature Selection

def lasso_feature_selection(X, y, alpha=0.1):
    model = Lasso(alpha=alpha)
    model.fit(X, y)
    return X.columns[model.coef_ != 0].tolist()

# Relief Feature Selection

def relief_algorithm(X, y, n_features_to_select):
    # Konwersja DataFrame i Series do numpy array
    columns = X.columns
    X = X.values
    y = y.values
    
    # Inicjalizacja wagi cech
    feature_scores = np.zeros(X.shape[1])
    
    # Instancja NearestNeighbors do znajdowania najbliższych sąsiadów
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    
    # Główna pętla algorytmu Relief
    for index, instance in enumerate(X):
        # Znajdowanie najbliższych sąsiadów (1 z tej samej klasy i 1 z innej)
        _, neighbors = nn.kneighbors([instance])
        hit_index = neighbors[0][0] if y[neighbors[0][0]] == y[index] else neighbors[0][1]
        miss_index = neighbors[0][1] if y[neighbors[0][1]] != y[index] else neighbors[0][0]
        
        # Obliczenie różnicy między instancją a hit i miss
        hit_diff = np.abs(instance - X[hit_index])
        miss_diff = np.abs(instance - X[miss_index])
        
        # Aktualizacja wagi cech
        feature_scores += -hit_diff if y[hit_index] == y[index] else hit_diff
        feature_scores += miss_diff if y[miss_index] != y[index] else -miss_diff
    
    # Normalizacja wyników przez liczbę instancji
    feature_scores /= X.shape[0]
    
    # Wybór najważniejszych cech
    important_indices = np.argsort(-feature_scores)[:n_features_to_select]
    # return columns names
    return columns[important_indices].tolist()

from mcut import AutoMCUT, mcut, mtrcs

def mcut_feature_selection(
        data,
        number_of_time_folds,
        target_variable,
        score_function,
        numb_of_bins,
        binning_func,
        how_many_folds,
        numb_of_best_features,
        numb_of_features_in_each_group,
        verbose=True,
        not_known=None,):
    m = AutoMCUT(data, number_of_time_folds, target_variable, score_function, numb_of_bins, binning_func, verbose, not_known)
    m.find_best_features( how_many_folds, numb_of_best_features, broad_df = False, numb_of_features_in_each_group = 5)
    return m.find_optimal_variables_lasso(numb_of_features_in_each_group)
