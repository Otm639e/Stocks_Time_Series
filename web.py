import taipy as tp
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from taipy import Config, Scope, Gui
import os
from collections import Counter
# from statsmodels.tools.sm_exceptions import ValueWarning, HessianInversionWarning, ConvergenceWarning
import warnings

# #in practice do not supress these warnings, they carry important information about the status of your model
# warnings.filterwarnings('ignore', category=ValueWarning)
# warnings.filterwarnings('ignore', category=HessianInversionWarning)
# warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Functions
def best_fitted_stocks(symbol_list):
    symbol_dict = {}
    for symbol in symbol_list:
        tickerData = yf.Ticker(symbol)
        tickerDf = tickerData.history(interval='1d', start='2023-1-1')
        
        first_diff = tickerDf[['Close']].diff()[1:]
        
        lag_count = 0
        a, ci = acf(first_diff, nlags=40, alpha=0.05, fft=False)
        # Key step
        centered_ci = ci - a[:,None]
        outside = np.abs(a) >= centered_ci[:,1]
        for i, lag in enumerate(outside):
            if lag:
                lag_count+=abs(list(a)[i])
        
        a2, ci2 = pacf(first_diff, nlags=13, alpha=0.05)
        centered_ci2 = ci2 - a2[:,None]
        outside2 = np.abs(a2) >= centered_ci2[:,1]
        for i, lag in enumerate(outside2):
            if lag:
                lag_count+=abs(list(a2)[i])
        
        symbol_dict[symbol] = lag_count
    return symbol_dict

def stock_history(ticker):
    #get data on this ticker
    tickerData = yf.Ticker(ticker)
    #get the historical prices for this ticker
    tickerDf = tickerData.history(interval='1d', start='2023-1-1')
    return tickerDf

def get_stock_data(ticker):
    tickerDf = stock_history(ticker)
    return tickerDf.reset_index()

def stock_closing(ticker):
    tickerDf = get_stock_data(ticker)
    return tickerDf[["Date","Close"]]

def make_stationary(ticker):
    tickerDf = stock_history(ticker)
    first_diff = tickerDf[['Close']].diff()[1:].reset_index()
    return first_diff

def get_acf(ticker):
    tickerDf = stock_history(ticker)
    first_diff = tickerDf[['Close']].diff()[1:]
    a, ci = acf(first_diff, nlags=40, alpha=0.05, fft=False)
    return pd.DataFrame(list(enumerate(a))).rename({0:"Lag",1:"Significance"}, axis=1)

def get_pacf(ticker):
    tickerDf = stock_history(ticker)
    first_diff = tickerDf[['Close']].diff()[1:]
    a, ci = pacf(first_diff, nlags=40, alpha=0.05)
    return pd.DataFrame(list(enumerate(a))).rename({0:"Lag",1:"Significance"}, axis=1)

def significant_ma_lags(ticker):
    tickerDf = stock_history(ticker)
    first_diff = tickerDf[['Close']].diff()[1:]
    a, ci = acf(first_diff, nlags=40, alpha=0.05, fft=False)
    # Key step
    centered_ci = ci - a[:,None]
    outside = np.abs(a) >= centered_ci[:,1]
    lag_lst = []
    for i, lag in enumerate(outside):
        if lag and i!=0:
            lag_lst.append(str(i))
    return lag_lst

def significant_ar_lags(ticker):
    tickerDf = stock_history(ticker)
    first_diff = tickerDf[['Close']].diff()[1:]
    a, ci = pacf(first_diff, nlags=40, alpha=0.05)
    # Key step
    centered_ci = ci - a[:,None]
    outside = np.abs(a) >= centered_ci[:,1]
    lag_lst = []
    for i, lag in enumerate(outside):
        if lag and i!=0:
            lag_lst.append(str(i))
    return lag_lst

def get_test_and_train_data(ticker):
    tickerDf = stock_history(ticker)
    train_end = tickerDf.Close.index[int(len(tickerDf.Close)*0.9)]
    test_start = tickerDf.Close.index[int(len(tickerDf.Close)*0.9)+1]
    test_end = tickerDf.Close.index[-1]
    train_data = tickerDf.Close[:train_end]
    test_data = tickerDf.Close[test_start:test_end]
    return (train_data, test_data, tickerDf)

def get_forcasts(ticker, ar_lag, ma_lag, diff):
    ar_lag, ma_lag, diff  = [int(ar_lag), int(ma_lag), int(diff)]
    train_data, test_data, tickerDf = get_test_and_train_data(ticker)
    my_order = (ar_lag,diff,ma_lag)
    model = ARIMA(train_data.reset_index().drop('Date',axis=1), order=my_order)
    model_fit = model.fit()
    num_days = len(test_data)
    forecasts = model_fit.forecast(steps=num_days).reset_index()
    forecasts['years'] = [x for x in test_data.index]
    forecasts = forecasts.set_index('years').drop('index', axis=1)
    copy = tickerDf[["Close"]].copy()
    copy['Forcasts'] = tickerDf.Close
    copy.loc[copy.index.isin(list(forecasts.index)),"Forcasts"] = forecasts.predicted_mean
    return copy.loc[copy.index.isin(list(forecasts.index)),:].reset_index()

def get_rolling_forecasts(ticker, ar_lag, ma_lag, diff):
    # Take out model Warnings
    warnings.simplefilter("ignore", category=UserWarning)

    ar_lag, ma_lag, diff = [int(ar_lag), int(ma_lag), int(diff)]
    train_data, test_data, tickerDf = get_test_and_train_data(ticker)
    my_order = (ar_lag,diff,ma_lag)
    rolling_predictions = test_data.copy()
    train_data2 = train_data.copy()
    num=0
    for train_end in test_data.index:
        num+=1
        # print(num)
        train_data2 = tickerDf.Close[:train_end-timedelta(days=1)]
        model = ARIMA(train_data2.reset_index().drop('Date',axis=1), order=my_order)
        model_fit = model.fit(transformed=False,method='innovations_mle')
        pred = float(model_fit.forecast().iloc[0])
        rolling_predictions[train_end] = pred
    forecasts = rolling_predictions.reset_index()
    forecasts = forecasts.rename({"Close":"Forcasts"}, axis=1)
    forecasts['Close'] = list(test_data)
    return forecasts

def get_new_rolling_forcast(state):
    print("Updating Forecast ...")
    state.rolling_forcast_data = get_rolling_forecasts(state.assets_sel, state.max_ar_lag, state.max_ma_lag, state.diff_param)
    print("Obtained New Forecasts ...")
    return

def adf_test(ticker):
    tickerDf = stock_history(ticker)
    diff = tickerDf[['Close']]
    p_val = 1
    num = 0
    while p_val > 0.05:
        diff = diff.diff()[1:]
        p_val = adfuller(diff)[1]
        num+=1
    return str(num)

def tomorrow_forcast(ticker, ar_lag, ma_lag, diff):
    # Take out model Warnings
    warnings.simplefilter("ignore", category=UserWarning)
    ar_lag, ma_lag, diff  = [int(ar_lag), int(ma_lag), int(diff)]
    my_order = (ar_lag,diff,ma_lag)
    all_data = stock_history(ticker).Close
    model = ARIMA(all_data.reset_index().drop('Date',axis=1), order=my_order)
    model_fit = model.fit(transformed=False,method='innovations_mle')
    num_days = 1
    forecast = model_fit.forecast(steps=num_days).iloc[0]
    return str(round(forecast, 2))

def update_ticker1(state):
    state.assets_sel = state.first_best_ticker
    return
def update_ticker2(state):
    state.assets_sel = state.second_best_ticker
    return
def update_ticker3(state):
    state.assets_sel = state.third_best_ticker
    return

# Variables
sp_assets = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
assets = sp_assets['Symbol'].str.replace('.','-').tolist()
assets_sel = assets[0] 

# stocks_lags = best_fitted_stocks(assets)
# k = Counter(stocks_lags)
# # Finding 3 highest values
# high = k.most_common(3)
# first_best_ticker = high[0][0]
# second_best_ticker = high[1][0]
# third_best_ticker = high[2][0]

diff_param = adf_test(assets_sel)
max_ma_lag = max(significant_ma_lags(assets_sel))
max_ar_lag = max(significant_ar_lags(assets_sel))
rolling_forcast_data = pd.DataFrame(columns=['Date','Close','Forcasts'])

# ##### Stocks with highest ACF & PACF Significance Values 
# <|{first_best_ticker}|button|on_action=update_ticker1|> <|{second_best_ticker}|button|on_action=update_ticker2|> <|{third_best_ticker}|button|on_action=update_ticker3|>
page='''
# Stock *Forecasting* (ARIMA Model)

<|{assets_sel}|selector|lov={assets}|dropdown|>
<|{get_stock_data(assets_sel)}|table|width={"100%"}|height={"50vh"}|>

#### Closing Prices
<|{stock_closing(assets_sel)}|chart|x=Date|y=Close|width={"100%"}|height={"50vh"}|>
#### ACF Plot
<|{get_acf(assets_sel)}|chart|type=bar|x=Lag|y=Significance|width={"100%"}|height={"50vh"}|>
#### PACF Plot
<|{get_pacf(assets_sel)}|chart|type=bar|x=Lag|y=Significance|width={"100%"}|height={"50vh"}|>
###### Significant MA Lags: <|{max_ma_lag}|selector|lov={significant_ma_lags(assets_sel)}|dropdown|>
###### Significant AR Lags: <|{max_ar_lag}|selector|lov={significant_ar_lags(assets_sel)}|dropdown|>

#### Tommorrow's Forecast: *<|{'$'+tomorrow_forcast(assets_sel, max_ar_lag, max_ma_lag, diff_param)}|>*

#### Rolling Forecast Origin
<|Forecast|button|on_action=get_new_rolling_forcast|hover_text='Update Forecast'|>
<|{rolling_forcast_data}|chart|mode=lines|x=Date|y[1]=Close|y[2]=Forcasts|line[2]=dash|color[2]=orange|id="forecast_chart"|>


'''


def on_change(state, var_name, var_value):
    if var_name == "assets_sel":
        state.max_ma_lag = significant_ma_lags(var_value)[-1]
        state.max_ar_lag = significant_ar_lags(var_value)[-1]
        return
    if var_name == "rolling_forcast_data":
        print("Updated Param")
    return
    
rest = tp.Rest()
gui = tp.Gui(page)
tp.run(
    rest,
    gui,
    title="Stock Forecasting",
    port=os.environ.get('PORT', '5000'),
    dark_mode=True,
    ) 
