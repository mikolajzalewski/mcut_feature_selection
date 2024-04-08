import datetime as dt
import os
import warnings
from itertools import combinations
from typing import Callable, List

import joblib
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
from czech_models.functions.functions import broader_df
from IPython.display import clear_output, display
from numba import jit
from tqdm.auto import tqdm

### METRICS #######################################################################################


def mean_std(x):
    """Function to calculate mean to std ratio"""
    if np.std(x) == 0:
        return 0
    x = x.copy()
    return np.mean(x) / np.std(x)


def winrate(x):
    """Function to calculate winrate"""
    if len(x) == 0:
        return 0
    x = x.copy()
    return len(x[x >= 0]) / len(x)


def effectiveness(df):
    if np.sum(abs(df)) == 0:
        return 0
    df = df.copy()
    return np.round(np.sum(df) / np.sum(abs(df)), 3)




mtrcs = ["mean", "sum", "count", mean_std, winrate, effectiveness]


################################################################################################

### RAW MCUT ###################################################################################


def add_diff(x):
    '''Function to calculate difference between two consecutive values'''
    x = x.copy()
    return x.diff()

def add_daily_diff(x):
    '''Function to calculate difference between two consecutive values from the same hour and minute'''
    x = x.copy()
    return x.groupby([x.index.hour, x.index.minute]).diff()

def add_rolling_mean(x, window=3):
    '''Function to calculate rolling mean, for the same hour and minute'''
    x = x.copy()
    return x.groupby([x.index.hour, x.index.minute]).rolling(window=window).mean().droplevel([0,1])

def add_rolling_std(x, window=3):
    '''Function to calculate rolling std, for the same hour and minute'''
    x = x.copy()
    return x.groupby([x.index.hour, x.index.minute]).rolling(window=window).std().droplevel([0,1])

def broader_df(df, not_known: list = []):
    df_ = df.copy()
    '''Function to add features to dataframe'''
    for col in tqdm(df_.columns):
        if col not in not_known:
            df_[col+'_diff'] = add_diff(df_[col])
        df_[col+'_daily_diff'] = add_daily_diff(df_[col])
        df_[col+'_rolling_3_mean'] = df_[col] - add_rolling_mean(df_[col], window=3)
        df_[col+'_rolling_7_mean'] = df_[col] - add_rolling_mean(df_[col], window=7)
        df_[col+'_rolling_14_mean'] = df_[col] - add_rolling_mean(df_[col], window=14)
    return df_

@jit()
def remove_outliers_iqr_and_return_normal_len(data, numb_of_bins):
    # Calculate the interquartile range (IQR)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    # Define the upper and lower bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Identify outliers
    outliers = (data < lower_bound) | (data > upper_bound)

    # Remove outliers
    cleaned_data = data[~outliers]

    return cleaned_data, np.round(
        (cleaned_data.max() - cleaned_data.min()) / numb_of_bins, 2
    )


def iterate_points(temporaly_list, len_of_typical_interval, numb_of_bins, long, short):
    """
    Base function to mcut algorithm. It iterates over points and adds new points if they are too far from each other and removes points if they are too close to each other.
    Args:
        temporaly_list: list of actual points
        len_of_typical_interval: length of typical interval
        numb_of_bins: number of bins
        long: how far as far can be points
        short: how close as close can be points
    Returns:
        list of points
    """
    temp = temporaly_list.copy()
    finish = False

    j_counter = 0
    while (not finish) & (j_counter < 2 * numb_of_bins):
        for i, val in enumerate(temp[:-1]):
            if i >= len(temp) - 2:
                finish = True
                break
            if temp[i + 1] - temp[i] > long * len_of_typical_interval:
                temp.append(temp[i] + (0.95 * long) * len_of_typical_interval)
                temp.sort()
                break
            j_counter += 1
    finish = False
    j_counter = 0
    while (not finish) & (j_counter < 2 * numb_of_bins):
        for i, val in enumerate(temp[:-1]):
            if (temp[i + 1] - temp[i] < short * len_of_typical_interval) & (
                i + 1 == len(temp) - 1
            ):
                # if the last interval, remove left instead of right
                temp.remove(temp[i])
                temp.sort()
                finish = True
                break
            if i + 1 >= len(temp) - 1:
                finish = True
                break
            if temp[i + 1] - temp[i] < short * len_of_typical_interval:
                temp.remove(temp[i + 1])
                temp.sort()
                break
            j_counter += 1

    return list(temp)


def expand_if_empty(dataframe, list_orig, threshhold):
    """
    Function to expand intervals if they have less observations than threshold.
    """
    # Initialize the start index, end index, and the list of new points
    start_idx, end_idx = 0, 1
    new_list = np.array([list_orig[start_idx] - 0.001])

    while end_idx < (len(list_orig) - 1):
        # Calculate the interval
        interval = dataframe[
            (dataframe >= list_orig[start_idx]) & (dataframe < list_orig[end_idx])
        ]

        # Check if there are at least 'threshold' observations in the interval
        if interval.shape[0] >= threshhold:
            # If so, add the right point to the new list
            new_list = np.append(new_list, list_orig[end_idx].round(3))
            # Move to the next interval
            start_idx += 1
            end_idx += 1
        else:
            # If not, remove the right point and restart from the beginning
            end_idx += 1

    interval = dataframe[
        (dataframe >= list_orig[start_idx]) & (dataframe < list_orig[end_idx])
    ]

    if interval.shape[0] >= threshhold:
        new_list = np.append(new_list, list_orig[end_idx])
    else:
        new_list = new_list[:-1]
        new_list = np.append(new_list, list_orig[end_idx])

    return new_list


@jit()
def preprocess(data, verbose):
    if len(data) == 0:
        return None
    changedata = False
    if np.quantile(data, 0.9) - np.quantile(data, 0.1) <= 2:
        data *= 100
        changedata = True

    abs_max = np.abs(data).max()
    if verbose:
        assert not np.isinf(abs_max)
    elif np.isinf(abs_max):
        return None

    max_value = data.max()
    min_value = data.min()
    return min_value, max_value, changedata, data


def raw_mcut(data, numb_of_bins, threshhold, long, short, verbose, duplicates):
    returned_preprocessed_data = preprocess(data, verbose)

    if returned_preprocessed_data is None:
        return None
    else:
        min_value, max_value, changedata, data = returned_preprocessed_data

    data, len_of_typical_interval = remove_outliers_iqr_and_return_normal_len(
        data, numb_of_bins
    )

    if verbose:
        assert (
            len_of_typical_interval != 0
        ), "Feature can NOT be constant / almost constant!!! At least 90% of the observations are the same."
    elif len_of_typical_interval == 0:
        return None

    percentiles = np.linspace(0, 1, numb_of_bins + 1)

    # Obliczanie kwantyli datala każdataej wartości procentyli
    quantiles = np.percentile(data, percentiles * 100)

    list_after_first_iter = iterate_points(
        quantiles.tolist(),
        len_of_typical_interval,
        numb_of_bins,
        long=long,
        short=short,
    )

    list_after_first_iter.insert(-1, max_value)
    list_after_first_iter.insert(0, min_value)

    points_list = np.sort(list_after_first_iter, kind="stable")

    if points_list.min() != min_value:
        points_list = np.append(min_value, points_list)
    if points_list.max() != max_value:
        points_list = np.append(points_list, max_value)

    points_list = np.unique(points_list)

    final_list = expand_if_empty(data, points_list, threshhold)

    if final_list.shape[0] < np.round(numb_of_bins / 2) + 1:
        if verbose:
            print(
                "Your feature is almost constant. Try to change long andata short parameters to adatajust the algorithm."
            )
            print(f"Returning qcut with {numb_of_bins} bins.")
        pd.qcut(data, numb_of_bins, duplicates=duplicates)

    return final_list / 100 if changedata else final_list


def mcut(
    d,
    numb_of_bins: int = 7,
    threshhold: int = 50,
    long: float = 1.5,
    short: float = 0.5,
    retbins: bool = 0,
    duplicates="raise",
    precision=3,
    verbose=True,
):
    assert len(d.shape) == 1, "Enter only one column!"
    assert d.dtype == float or d.dtype == int

    if "values" in dir(d):
        d = d.values
    data_to_raw_mcut = d[~np.isnan(d)]

    raw_list = raw_mcut(
        data_to_raw_mcut, numb_of_bins, threshhold, long, short, verbose, duplicates
    )
    if raw_list is None:
        return None

    return pd.cut(
        d,
        np.sort(raw_list),
        duplicates=duplicates,
        retbins=retbins,
        precision=precision,
    )


################################################################################################

### MCUT #######################################################################################


def prepare_cols_comb(
    cols,
    symetrical_operations=["+"],
    asymetrical_operations=["-"],
    left_forbidden_cols={},
    right_forbidden_cols={},
):
    """Get all 2 element combinations from given columns

    Args:
        cols: list of columns
        symetrical_operations: list of symetrical operations to use, default ['+', '-', '*']
        asymetrical_operations: list of asymetrical operations to use, default ['/']
        left_forbidden_cols: dict with keys being operations and values being lists of columns to not use in the left position for given operation
        right_forbidden_cols: dict with keys being operations and values being lists of columns to not use in the right position for given operation
    Returns:
        list of all combinations of columns with operations
        length of returned list is equal to len(cols) * (len(cols) - 1) * (len(symetrical_operations) + len(asymetrical_operations)/2
    """
    cols_comb = cols.copy()

    for operation in symetrical_operations:
        for left_id in range(len(cols)):
            for right_id in range(left_id + 1, len(cols)):
                if (
                    operation in left_forbidden_cols.keys()
                    and cols[left_id] in left_forbidden_cols[operation]
                ) or (
                    operation in right_forbidden_cols.keys()
                    and cols[right_id] in right_forbidden_cols[operation]
                ):
                    continue
                cols_comb.append(f"{cols[left_id]} {operation} {cols[right_id]}")

    for operation in asymetrical_operations:
        for left_id in range(len(cols)):
            for right_id in range(len(cols)):
                if left_id != right_id:
                    if (
                        operation in left_forbidden_cols.keys()
                        and cols[left_id] in left_forbidden_cols[operation]
                    ) or (
                        operation in right_forbidden_cols.keys()
                        and cols[right_id] in right_forbidden_cols[operation]
                    ):
                        continue
                    cols_comb.append(f"{cols[left_id]} {operation} {cols[right_id]}")

    return cols_comb


def common_member(a_list, b_list):
    """Returns common elements from two lists
    Args:
        a_list: list
        b: list
    Returns:
        list with common elements
    """
    a_set = set(a_list)
    b_set = set(b_list)

    if a_set & b_set:
        return a_set & b_set
    return []


def iterations(col, data, y_col, score_function, n_bins, func=pd.qcut, threshhold=50):
    """
    Function to calculate the monotonicity of a variable
    """
    if isinstance(col, str):
        primary_sum = abs(data[y_col]).sum()

        if func is not mcut:
            bins = func(data.eval(col).values, n_bins, duplicates="drop")
        else:
            bins = func(
                data.eval(col).values,
                n_bins,
                duplicates="drop",
                threshhold=threshhold,
                verbose=False,
            )

        if bins is None:
            return []

        raw = data[y_col].groupby([bins])

        qcut_vals = raw.agg(score_function)

        achieved_sum = abs(raw.sum()).sum()

        n_bins_after = qcut_vals.shape[0]
        if n_bins_after < n_bins / 2:
            return []

        ref_x = list(range(1, n_bins_after + 1))
        temp_array = np.array([ref_x, [1] * n_bins_after])
        slope = sm.OLS(qcut_vals, temp_array.T).fit().params.x1
        mono = scipy.stats.spearmanr(ref_x, qcut_vals)
        n_features = len(col.split(" "))
        n_features = n_features - n_features // 2
        efficiency = achieved_sum / primary_sum

        return [
            col,
            n_features,
            slope,
            mono[0],
            qcut_vals.min(),
            qcut_vals.max(),
            qcut_vals[qcut_vals > 0].sum(),
            qcut_vals[qcut_vals < 0].sum(),
            efficiency,
        ]


def selection(
    data: pd.DataFrame,
    y_col: str,
    cols_to_check: List[str],
    score_function: Callable,
    n_bins: int,
    func=pd.qcut,
    threshhold=None,
) -> pd.DataFrame:
    """Returns dataframe with features and their parameters for monotonicity check, such as slope and spearman correlation, due to qcut method.

    Args:
        data: dataframe with data
        y_col: name of column with target variable
        cols_to_check: list of columns to check
        score_function: function to calculate score for each bin
        n_bins: number of bins to use
    Returns:
        dataframe with features and their parameters for monotonicity check
    """
    if threshhold is None:
        args = [data, y_col, score_function, n_bins, func]
    else:
        args = [data, y_col, score_function, n_bins, func, threshhold]
    results = []
    #     results = joblib.Parallel(n_jobs=-1)(joblib.delayed(iterations)(item, *args)
    #                                          for item in tqdm(cols_to_check))
    for item in tqdm(cols_to_check):
        to_append = iterations(item, *args)
        if to_append is not None:
            results.append(to_append)

    return_dataframe = pd.DataFrame(results)
    if not return_dataframe.empty:
        return_dataframe.columns = [
            "feature",
            "n_features",
            "slope",
            "mono",
            "min_val",
            "max_val",
            "sum_pos",
            "sum_neg",
            "efficiency",
        ]
        return_dataframe = return_dataframe.assign(
            min_max_ratio=lambda x: x.max_val / abs(x.min_val)
        )
        return_dataframe = return_dataframe.sort_values("mono")
    else:
        return_dataframe = pd.DataFrame(
            columns=[
                "feature",
                "n_features",
                "slope",
                "mono",
                "min_val",
                "max_val",
                "sum_pos",
                "sum_neg",
                "efficiency",
                "min_max_ratio",
            ]
        )

    return return_dataframe.dropna()


def fold_validation(
    data_set: pd.DataFrame,
    k_number: int,
    y_col: str,
    cols_to_check: List[str],
    score_function: Callable,
    n_bins: int,
    func=pd.qcut,
    condition: str = "abs(mono) > 0.8",
    threshhold=None,
    additional_info = '',
) -> pd.DataFrame:
    """Function is used to select features for monotonicity check. It splits data into k_number
       of parts and checks monotonicity for each part. Then it check if common feature is monotonic on whole dataset and returns it.
    Args:
        data_set: dataframe with data
        k_number: number of parts to split data
        y_col: name of column with target variable
        cols_to_check: list of columns to check
        score_function: function to calculate score for each bin
        n_bins: number of bins to use
        condition: condition to select features
    Returns:
        dataframe with features and their parameters for monotonicity check
    """
    assert (
        type(data_set) == pd.core.frame.DataFrame
    ), "data_set must be a pandas dataframe"
    assert type(y_col) == str, "y_col must be a string"
    assert type(cols_to_check) == list, "cols_to_check must be a list"
    assert type(n_bins) == int, "n_bins must be an integer"
    assert n_bins > 0, "n_bins must be greater than 0"
    assert type(condition) == str, "condition must be a string"
    assert (
        type(threshhold) == int or type(threshhold) == float or threshhold is None
    ), "threshhold must be a number or None"
    assert threshhold is None or threshhold > 0, "threshhold must be greater than 0"
    assert type(func) == type(pd.qcut), "func must be a function"
    assert callable(score_function), "score_function must be a function"
    assert type(additional_info) == str, "additional_info must be a string"

    results = cols_to_check

    if type(k_number) == int:
        assert type(k_number) == int, "k_number must be an integer"
        assert k_number > 0, "k_number must be greater than 0"

        leng = int(data_set.shape[0] / k_number)
        timeseries = True

    elif type(k_number) == tuple:
        assert type(k_number[0]) == str, "k_number[0] must be a string"
        assert type(k_number[1]) == int, "k_number[1] must be an integer"
        assert (
            k_number[0] in data_set.columns
        ), "choosen variable k_number[0] must be in the df"

        dep_var = k_number[0]
        k_number = k_number[1]
        timeseries = False

    else:
        raise Exception(
            """ k_number must be int if you want to check your variable in time periods, """
            + """ or pass tuple('var_name', n) to simulate variable influence"""
            + """ in n baskets of var_name variable """
        )

    for i in range(k_number):
        clear_output()
        print("Round ", i + 1, "/", k_number, " " + str(additional_info))

        if timeseries:
            data_frame = data_set[leng * i : leng * (i + 1)].copy()
        else:
            q1 = np.quantile(data_set[dep_var], i / k_number)
            q2 = np.quantile(data_set[dep_var], (i + 1) / k_number)
            data_frame = data_set.query(f"{dep_var} >= {q1} and {dep_var} <= {q2}")

        if threshhold is None:
            args = [data_frame, y_col, results, score_function, n_bins, func]
        else:
            args = [
                data_frame,
                y_col,
                results,
                score_function,
                n_bins,
                func,
                threshhold,
            ]

        res = selection(*args).query(condition)
        if res.feature.shape[0] == 0:
            clear_output()
            print("Zero features :(")
            return pd.DataFrame([])
        results = common_member(results, list(res.feature))
    clear_output()
    data_frame = data_set.copy()
    if threshhold is None:
        args = [data_frame, y_col, results, score_function, n_bins, func]
    else:
        args = [data_frame, y_col, results, score_function, n_bins, func, threshhold]
    res = selection(*args).query(condition)
    clear_output()
    return res


################################################################################################

### FEATURE SELECTION ##########################################################################


def evaluate_strategy(df_: pd.Series):
    """
    Function takes series and return metrics that describes performance of the strategy.
    """
    assert type(df_) == pd.Series, "Data must be a pd.Series."
    # per mwh
    mean_ = np.round(np.mean(df_), 2)
    sum_ = np.round(np.sum(df_), 2)
    mean_std_ = np.round(mean_std(df_), 2)
    winrate_ = np.round(winrate(df_), 2)
    min_ = np.round(np.min(df_), 2)
    max_ = np.round(np.max(df_), 2)
    # per week
    percent_notrade = np.round(
        (
            1
            - (df_[df_ != 0].resample("d").count() > 0).sum()
            / (df_.resample("d").count() > 0).sum()
        ),
        2,
    )
    max_drawdown = np.round(-(df_.cumsum().cummax() - df_.cumsum()).max().round(2), 2)
    d_agg = df_.resample("w").sum().reset_index(drop=True)
    slope = np.round(sm.OLS(d_agg.cumsum(), d_agg.index).fit().params.x1, 2)
    mono = np.round(scipy.stats.spearmanr(d_agg.index, d_agg.cumsum())[0], 2)
    efficiency = df_.sum() / abs(df_).sum()
    return [
        mean_,
        sum_,
        mean_std_,
        winrate_,
        min_,
        max_,
        percent_notrade,
        max_drawdown,
        slope,
        mono,
        efficiency,
    ]


def backtest(df, feature, margin, freq, allow_pos_values=True, allow_neg_values=True):
    assert type(df) == pd.DataFrame
    if type(feature) is not str:
        feature = feature.astype(str)
    if type(margin) is not str:
        margin = margin.astype(str)
    df["feat"] = df.eval(feature)
    df["marg"] = df.eval(margin)
    df["const"] = 1
    d = df[["feat", "const", "marg"]].resample(freq).mean().dropna()
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    model = sm.OLS(d.marg, d[["feat", "const"]]).fit()
    d["pred"] = model.predict(d[["feat", "const"]])
    if allow_pos_values and allow_neg_values:
        d["pnl"] = np.where(d.pred > 0, d.marg, -d.marg)
    elif allow_pos_values and not allow_neg_values:
        d["pnl"] = np.where(d.pred > 0, d.marg, 0)
    elif not allow_pos_values and allow_neg_values:
        d["pnl"] = np.where(d.pred < 0, -d.marg, 0)
    return d.pnl


def compare_time_units(time1, time2):
    # Funkcja do zamiany jednostek czasu na liczbę minut
    def time_to_minutes(time_str):
        unit = time_str[-1]
        value = time_str[:-1]
        if value == "":
            value = 1
        else:
            value = int(value)
        if unit == "h":
            return value * 60
        elif unit == "d":
            return value * 1440  # 1 dzień = 24 godziny * 60 minut
        elif unit == "M":
            return value * 43200  # 1 miesiąc = 30 dni * 24 godziny * 60 minut
        elif unit == "t":
            return value  # minuty (1 minuta = 1 minuta)
        else:
            raise ValueError("Nieobsługiwana jednostka czasu")


def score_found_features_in_time(
    df_,
    agg_results,
    marg="marg",
    freq="h",
    allow_pos_values=True,
    allow_neg_values=True,
):
    df_in = agg_results.copy()
    df_sum = pd.DataFrame()
    df_in["marg"] = marg
    df_in["freq"] = freq
    for idx, val in tqdm(df_in.iterrows(), total=len(df_in)):
        temp_df = evaluate_strategy(
            backtest(
                df_,
                val["feature"],
                val["marg"],
                val["freq"],
                allow_pos_values,
                allow_neg_values,
            )
        )
        temp_df = pd.DataFrame([val["feature"], val["marg"], val["freq"]] + temp_df).T
        ret_cols = [
            "mean",
            "sum",
            "mean_std",
            "winrate",
            "min",
            "max",
            "percent_notrade",
            "max_drawdown",
            "week_slope",
            "week_mono",
            "efficiency",
        ]
        temp_df.columns = ["feature", "marg", "freq"] + ret_cols
        df_sum = pd.concat([df_sum, temp_df]).reset_index(drop=True)

    return df_sum


################################################################################################

### AUTOMATIZATION TOOLS #######################################################################


class AutoMCUT:
    def __init__(
        self,
        data,
        number_of_folds,
        target_variable,
        score_function,
        numb_of_bins,
        binning_func,
        verbose=True,
        not_known=None,
    ):
        self.data = data
        self.k = number_of_folds
        self.marg = target_variable
        self.func = score_function
        self.numb_of_bins = numb_of_bins
        self.cut = binning_func
        self.verbose = verbose

        cols = self.data.columns.tolist()
        cols.remove(self.marg)
        self.cols = cols

        if not_known is not None:
            self.not_known = not_known

        self.preprocessed = False

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def find_best_features(self, how_many_folds=4, numb_of_best_features=200, broad_df = False):
        """
        Function takes dataframe and returns best features for given target variable.
        It allows to start from small score of correlation and then automatically increase it.
        """
        data = self.data.copy()
        k = self.k
        marg = self.marg
        func = self.func
        numb_of_bins = self.numb_of_bins
        cut = self.cut
        cols_accumulated_folds = 0

        if broad_df:
            if "not_known" in dir(self):
                marg_raw = data[marg].copy()
                data = data.drop(marg, axis=1)
                data = broader_df(data, self.not_known)
                data[marg] = marg_raw
            else:
                marg_raw = data[marg].copy()
                data = data.drop(marg, axis=1)
                data = broader_df(data)
                data[marg] = marg_raw
            data = data.dropna()
            
        assert not data.empty
        cols = data.drop(columns=[marg]).columns.tolist()
        results = selection(data, marg, cols, func, numb_of_bins, cut)
        self.results = results
        assert max(abs(results.mono)) > 0.5, "Your data are not correlated at all!!!"

        condition = np.quantile(abs(results.mono).clip(0.5, 0.9), 0.5)
        basic_cols = results.query(f"abs(mono)>{condition}").feature.tolist()
        data = data[basic_cols + [marg]].copy()
        self.data = data
        temp_cols = prepare_cols_comb(basic_cols)
        cols_accumulated_folds += 1
        try:
            while (~results.empty) & (cols_accumulated_folds <= (how_many_folds)):
                if condition + 0.1 < 1:
                    condition += 0.1
                warnings.warn('Now processing "abs(mono)>' + str(condition) + '", and folds are {cols_accumulated_folds}/{how_many_folds}')
                temp = fold_validation(
                    data,
                    k,
                    marg,
                    temp_cols,
                    func,
                    numb_of_bins,
                    cut,
                    f"abs(mono)>{condition}",
                    additional_info=f'Folds: {cols_accumulated_folds}/{how_many_folds}'
                )
                if temp.empty:
                    break
                temp["abs_mono"] = abs(temp.mono)

                results = temp
                that_iter_cols = (
                    temp.sort_values(["mono", "efficiency"]).tail(numb_of_best_features).feature.tolist()
                )
                temp_cols = prepare_cols_comb(that_iter_cols)
                cols_accumulated_folds += 1

        except KeyboardInterrupt:
            clear_output()
            print("KeyboardInterrupt caught! Task was interrupted by the user.")
            print("Returning the last df with scored features.")
            print('I saved it as "results".')
            self.results = pd.concat([results, self.results]).drop_duplicates()
            self.results["abs_mono"] = abs(self.results.mono)
        clear_output()
        print("Max condition is " + f"abs(mono)>{condition}")
        print("Number of features is " + str(results.shape[0]))
        print("I saved it as 'results'.")
        self.results = pd.concat([results, self.results]).drop_duplicates()
        self.results["abs_mono"] = abs(self.results.mono)

    def find_optimal_variables(self, numb_of_features_in_each_group = 2):
        """
        Find the set of independent variables that maximizes the heuristic.

        Parameters:
        - df (DataFrame): DataFrame containing the variables.
        - margin_variable (str): Name of the dependent variable (margin).

        Returns:
        - tuple: (best_set, max_heuristic, dict) where best_set is the set of variables that maximizes the heuristic
                and max_heuristic is the value of the heuristic and dict is a dictionary mapping the names of the
                    variables in best_set to the names of the variables in df.
        """
        data = self.data.copy()
        margin_variable = self.marg

        if "results" not in dir(self):
            self.find_best_features()
        agg_results = self.results.groupby(self.results.n_features).apply(lambda x: x.sort_values(["abs_mono", "efficiency"]).tail(numb_of_features_in_each_group)).droplevel(0)

        # create df
        df = pd.DataFrame(index=data.index)
        dc = {}
        for i, col in enumerate(agg_results.feature.tolist()):
            df[f"zm_{i}"] = data.eval(col)
            dc[f"zm_{i}"] = col
        df["margin"] = data[margin_variable]

        # Compute the correlation matrix
        corr_matrix = df.corr()

        # Initialize variables for tracking the best result
        max_heuristic = -np.inf
        best_set = None

        # Iterate over all possible combinations of independent variables
        for k in range(1, len(df.columns)):
            clear_output()
            print(f"Round {k}/{len(df.columns)-1}:")
            for subset in tqdm(combinations(df.columns[df.columns != "margin"], k)):
                # Calculate r_cf - average correlation between independent variables and dependent variable
                r_cf = abs(corr_matrix.loc[list(subset), "margin"]).mean()

                # Calculate r_ff - average correlation between independent variables
                r_ff = abs(corr_matrix.loc[subset, subset]).mean()
                if type(r_ff) == pd.Series:
                    r_ff = r_ff.values[0]

                # Calculate heuristic
                heuristic = (k * r_cf) / np.sqrt(k + k * (k - 1) * r_ff)
                #             heuristic = (r_cf) / (np.sqrt(k) * r_ff)

                # Check if this set of variables is the best so far
                if heuristic > max_heuristic:
                    max_heuristic = heuristic
                    best_set = subset

        print("Best set of variables:")
        print(best_set)
        print('I saved it as "best_set".')
        self.best_set = best_set
        print("Dict to map variables is saved as 'dc'.")
        self.dc = dc

    def generate_report(self, err=0, train_period_in_days=365, return_signals = False, side = None):
        """
        Function return basic metrics and cumsum of pnl for OLS model.
        It uses best_set and dc from find_optimal_variables.
        Model is retrain every day, train data is 1 year before test data.
        PnL is margin * sign(pred)
        return_signals: return model predictions as a series
        side: if you have two prices, you can choose only one side of predictions as a taken trades
        """
        assert side in [None, 'short', 'long']
        data = self.data.copy()
        margin_variable = self.marg
        best_set = self.best_set
        dc = self.dc

        if "own_set" in dir(self):
            best_set = self.own_set

        t = pd.DataFrame(index=data.index)
        for k in best_set:
            if str(k) in dc.keys():
                t[k] = data.eval(dc[k])
            else:
                t[k] = data.eval(k)
        t["const"] = 1
        t["margin"] = data[margin_variable]

        cols = t.columns.tolist()[:-1]

        d = t.dropna()

        d["pred_OLS"] = np.nan

        for date in pd.date_range(
            d.index[0] + dt.timedelta(train_period_in_days),
            d.index[-1],
            freq="D",
            tz="CET",
        ):
            start_date = date - dt.timedelta(days=train_period_in_days)

            train = d.loc[start_date : date - dt.timedelta(1) - dt.timedelta(hours=1)]
            if train.empty:
                break
            test = d.loc[
                date
                + dt.timedelta(days=1) : date
                + dt.timedelta(days=1)
                + dt.timedelta(hours=23)
            ]

            model = sm.OLS(train.margin, train[cols]).fit()
            d.loc[test.index, "pred_OLS"] = model.predict(test[cols])

        d["pnl_OLS"] = np.where(
            d.pred_OLS > 0, d.margin, np.where(d.pred_OLS < 0, -d.margin, np.nan)
        )
        d["pnl_OLS"] = np.where(abs(d.pred_OLS) > err, d["pnl_OLS"], np.nan)

        if side == 'long':
            d["pnl_OLS"] = np.where(d.pred_OLS > 0, d["pnl_OLS"], np.nan)
        elif side == 'short':
            d["pnl_OLS"] = np.where(d.pred_OLS < 0, d["pnl_OLS"], np.nan)

        model.summary()

        if return_signals:
            return d.pred_OLS
