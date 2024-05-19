import datetime as dt
import statsmodels.api as sm
import xgboost as xgb
import pandas as pd
import numpy as np

from tqdm.auto import tqdm
from sklearn.linear_model import Lasso


def backtest_OLS(df, cols, type = 'test'):
    
    if type == 'test':
        dates = pd.date_range('2022-12-31', '2023-04-01', freq='D', tz = 'CET')
    elif type == 'train':
        dates = pd.date_range('2022-01-01', '2022-12-31', freq='D', tz = 'CET')
        
    d = df[cols + ['marg']].dropna()
    d['pred'] = np.nan
    pvs = pd.DataFrame(columns=cols)
    
    for date in tqdm(dates):

        start_date = date - dt.timedelta(days = 365)

        train = d.loc[(start_date) : (date - dt.timedelta(1) - dt.timedelta(hours=1))][cols + ['marg']].dropna()
        test = d.loc[(date+dt.timedelta(days=1)):(date +dt.timedelta(days=1) +dt.timedelta(hours=23))]
        
        if (train.empty) | (test.empty):
            continue
        
        model = sm.OLS(train.marg, train[cols]).fit()
        d.loc[test.index,'pred'] = model.predict(test[cols])
        pvs.loc[date] = model.pvalues
        
    return d, pvs

def backtest_Lasso(df, cols, type = 'test'):
    
    if type == 'test':
        dates = pd.date_range('2022-12-31', '2023-04-01', freq='D', tz = 'CET')
    elif type == 'train':
        dates = pd.date_range('2022-01-01', '2022-12-31', freq='D', tz = 'CET')
    
    d = df[cols + ['marg']].dropna()
    d['pred'] = np.nan
    pvs = pd.DataFrame(columns=cols)
    
    for date in tqdm(dates):
        
        start_date = date - dt.timedelta(days = 365)

        train = d.loc[(start_date) : (date - dt.timedelta(1) - dt.timedelta(hours=1))][cols + ['marg']].dropna()
        test = d.loc[(date+dt.timedelta(days=1)):(date +dt.timedelta(days=1) +dt.timedelta(hours=23))]
        
        if (train.empty) | (test.empty):
            continue
        
        model = Lasso().fit(train[cols], train.marg)
        d.loc[test.index,'pred'] = model.predict(test[cols])
        pvs.loc[date] = model.coef_
        
    return d, pvs

def backtest_XGBoost(df, cols, type = 'test'):
    
    if type == 'test':
        dates = pd.date_range('2022-12-31', '2023-04-01', freq='D', tz = 'CET')
    elif type == 'train':
        dates = pd.date_range('2022-01-01', '2022-12-31', freq='D', tz = 'CET')
    
    d = df[cols + ['marg']].dropna()
    d['pred'] = np.nan
    pvs = pd.DataFrame(columns=cols)
    
    for date in tqdm(dates):

        start_date = date - dt.timedelta(days = 365)

        train = d.loc[(start_date) : (date - dt.timedelta(1) - dt.timedelta(hours=1))][cols + ['marg']].dropna()
        test = d.loc[(date+dt.timedelta(days=1)):(date +dt.timedelta(days=1) +dt.timedelta(hours=23))]
        
        if (train.empty) | (test.empty):
            continue
        
        model = xgb.XGBRegressor().fit(train[cols], train.marg)
        d.loc[test.index,'pred'] = model.predict(test[cols])
        pvs.loc[date] = model.feature_importances_
        
    return d, pvs