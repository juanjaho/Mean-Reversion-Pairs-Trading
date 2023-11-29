
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (15, 6)
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.stattools import jarque_bera
from pykalman import KalmanFilter
import multiprocess as mp
import time
import pmdarima as pm
import yfinance as yf
from math import floor

# ignore warnings
import warnings
warnings.filterwarnings('ignore')


LEVERAGE = 1

# Invert Strategy: Should be set to False for most cases
# False: Long the spread when the spread is NEGATIVE and short the spread when the spread is POSITIVE
# True: Long the spread when the spread is POSITIVE and short the spread when the spread is NEGATIVE
INVERT_STRATEGY = False

# Intrady Trading: Buy and sell on the same day
# True: Buy on Open, Sell on Close
# False: Buy on Open, Sell on Next Open
INTRADAY_TRADING = True

# Carry Forward Existing Positions: Carry forward existing positions to the next day
# Positions signal refers to the signal to buy or sell the spread
# True: Carry forward existing positions to the next day when there is no new positions signal
# False: Do not carry forward existing positions to the next day
CARRY_FORWARD_EXISTING_POSITIONS_SIGNAL = False

# TICKER = [x, y]
TICKER = ["NWS", "NWSA"]

# Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
INTERVAL = '1d'
START_DATE = '2023-01-01'
END_DATE = '2024-01-01'


prices = yf.download(TICKER+["SPY"], start=START_DATE, end=END_DATE, interval=INTERVAL)

prices_close = prices['Close']
prices_close = prices_close.dropna()
prices_close = prices_close[TICKER]


prices = prices['Open'] 
prices = prices.dropna()

# split into selected ticker and spy
spy = prices['SPY']
prices = prices.drop('SPY', axis=1)
# arrange data in order of ticker
prices = prices[TICKER]

# calculate daily return of SPY
spy_daily_return = spy.pct_change()



prices.isna().sum()


# split into train and test
train_size = floor(len(prices)*0.7)
# train_size = len(prices.loc[:'2023-04-01'])
train, test = prices.iloc[:train_size, :], prices.iloc[train_size:, :]
x_train, x_test = train.iloc[:, 0], test.iloc[:, 0]
y_train, y_test = train.iloc[:, 1], test.iloc[:, 1]


# Use AR model to find the hedge ratio
log_x_train = np.log(x_train)
log_y_train = np.log(y_train)
log_x_test = np.log(x_test)
log_y_test = np.log(y_test)

# log(x)-log(y) = log(beta) + s, where s noise
log_beta_train = log_y_train-log_x_train
log_beta_test = log_y_test-log_x_test

# https://alkaline-ml.com/pmdarima/tips_and_tricks.html
# test if ARIMA is appropriate to estimate beta
model = pm.auto_arima(log_beta_train.values, start_p=1, start_q=1, max_p=5, max_q=5, m=1, seasonal=True, trace=True)
# model = pm.auto_arima(log_beta, m=12, seasonal=True, trace=True) 
    
def define_arima_class(model) -> pm.arima.arima.ARIMA:
    """
    Define ARIMA class based on the model
    """
    return model

model = define_arima_class(model)

# get the p, d, q from the model
p = model.order[0]
d = model.order[1]
q = model.order[2]


# model_coeffs = dict(zip(model.arima_res_.param_names, model.params()))

pred_log_beta_train = model.predict_in_sample()
# set the first few predictions to be NaN
to_skip = max(p, d, q)
pred_log_beta_train[:to_skip] = np.nan

# get the variance of the residuals sigma^2
pred_log_sigma2_train = np.ones(len(log_beta_train))*model.params()[-1]

# predict the test set
pred_log_beta_test = np.zeros(len(log_beta_test))
pred_log_sigma2_test = np.zeros(len(log_beta_test))
for i in range(len(log_beta_test)):
    
    pred_log_beta_test[i] = model.predict(n_periods=1)[0]
    pred_log_sigma2_test[i] = model.params()[-1]
    
    # update the model after prediction to prevent forward-looking bias
    model.update(log_beta_test[i])

# convert logs to original prices
pred_beta_train = np.exp(pred_log_beta_train)
pred_beta_test = np.exp(pred_log_beta_test)
pred_sigma2_train = np.exp(pred_log_sigma2_train)
pred_sigma2_test = np.exp(pred_log_sigma2_test)


from scipy import poly1d

log_spread_train = log_y_train - log_x_train - pred_log_beta_train
log_spread_test = log_y_test - log_x_test - pred_log_beta_test

# plot the predicted beta and the actual beta
pred_beta = np.append(pred_beta_train, pred_beta_test)
pred_beta = pd.Series(pred_beta, index=prices.index)
beta = np.exp(np.append(log_beta_train, log_beta_test))
beta = pd.Series(beta, index=prices.index)


# state_vars = spread_train.ewm(span=30).var()
pred_log_vars = np.append(pred_log_sigma2_train, pred_log_sigma2_test)
pred_log_sigma = np.sqrt(pred_log_vars)
log_spread = np.append(log_spread_train, log_spread_test)


# Set initial positions to 0
columns = ['positions_'+TICKER[0]+'_long', 'positions_'+TICKER[0]+'_short', 'positions_'+TICKER[1]+'_long', 'positions_'+TICKER[1]+'_short']

# set initial positions to None if you want to carry forward existing positions signal even if there is no entry/exit signal on current day
if CARRY_FORWARD_EXISTING_POSITIONS_SIGNAL:
    prices[columns] = None
else:
    prices[columns] = 0

# Entry conditions based on sqrt_Qt
log_entry_threshold = 0*pred_log_sigma  # Replace with your specific calculation of sqrt_Qt

# Short entry
prices.loc[log_spread > log_entry_threshold , ('positions_'+TICKER[0]+'_short', 'positions_'+TICKER[1]+'_short')] = [1, -1]

# Long entry
prices.loc[log_spread < -log_entry_threshold, ('positions_'+TICKER[0]+'_long', 'positions_'+TICKER[1]+'_long')] = [-1, 1]


# Exit conditions based on sqrt_Qt
log_exit_threshold = 0*pred_log_sigma  # Set your desired exit threshold

# let entry threshold be 1 standard deviations from rolling standard deviation
# exit_threshold = 0*spread.rolling(lookback).std()

# Close short position
prices.loc[log_spread < log_exit_threshold, ('positions_'+TICKER[0]+'_short', 'positions_'+TICKER[1]+'_short')] = 0

# Close long position
prices.loc[log_spread > -log_exit_threshold, ('positions_'+TICKER[0]+'_long', 'positions_'+TICKER[1]+'_long')] = 0


# ensure existing positions are carried forward
prices.fillna(method='ffill', inplace=True)



positions_long_signal = prices.loc[:,('positions_'+TICKER[0]+'_long', 'positions_'+TICKER[1]+'_long')]
positions_short_signal = prices.loc[:,('positions_'+TICKER[0]+'_short', 'positions_'+TICKER[1]+'_short')]

positions_signal = np.array(positions_long_signal) + np.array(positions_short_signal)
if INVERT_STRATEGY:
    positions_signal = -positions_signal
positions_signal = pd.DataFrame(positions_signal, index=prices.index, columns=['positions_'+TICKER[0], 'positions_'+TICKER[1]])

unit_positions = positions_signal.copy()
# multiply sposition of TickX by hedge ratio to get amount of TickX to equalize TickY
unit_positions['positions_'+TICKER[0]] = positions_signal['positions_'+TICKER[0]] * np.append(pred_beta_train, pred_beta_test)

dollar_positions = unit_positions.copy()
# multiply positions by price to get dollar position
dollar_positions["positions_"+TICKER[0]] = unit_positions['positions_'+TICKER[0]] * prices[TICKER[0]]
dollar_positions["positions_"+TICKER[1]] = unit_positions['positions_'+TICKER[1]] * prices[TICKER[1]]


# Intraday Trading: Buy and sell on the same day
# In this scenario: Buy on open, sell on close
if INTRADAY_TRADING:
    intraday_daily_returns = (prices_close - prices[TICKER])/prices[TICKER]

    # calculate weighted returns based on positions
    pnl = (np.array(dollar_positions) * np.array(intraday_daily_returns)).sum(axis=1)

    # calculate the spread at close
    # log_spread_close = np.log(prices_close[TICKER[1]]) - np.log(prices_close[TICKER[0]]) - np.log(pred_beta)

    daily_used_capital = np.sum(np.abs(dollar_positions), axis=1)/2/LEVERAGE
    transaction_pct_cost = 0.0005
    transaction_costs = np.sum(np.abs(dollar_positions), axis=1)*transaction_pct_cost

    # Intraday Trading is always daily rebalanced
    # DAILY_REBALANCE = False
    # if not DAILY_REBALANCE:
    #     # ignore transaction costs if positions_sign is not changed from previous day
    #     transaction_costs_bool = (positions_sign == positions_sign.shift()).any(axis=1)
    #     transaction_costs[transaction_costs_bool] = 0

    # subtract transaction costs
    pnl_with_cost = (pnl - transaction_costs)/daily_used_capital
    pnl_with_cost = pnl_with_cost.fillna(0)

    # calculate the percentage returns
    pct_pnl = pnl / daily_used_capital

    # if pnl is nan, set to 0
    pct_pnl = pct_pnl.fillna(0)

else:
    # get daily returns
    daily_returns = prices.pct_change()[TICKER]

    # calculate weighted returns based on positions
    pnl = (np.array(dollar_positions.shift()) * np.array(daily_returns)).sum(axis=1)

    daily_used_capital = np.sum(np.abs(dollar_positions), axis=1)/2/LEVERAGE
    transaction_pct_cost = 0.0005
    transaction_costs = np.sum(np.abs(dollar_positions), axis=1)*transaction_pct_cost

    # TODO: Daily Rebalance is not complete as it has to accoutn for the unchanged positions
    DAILY_REBALANCE = False
    if not DAILY_REBALANCE:
        # ignore transaction costs if positions_sign is not changed from previous day
        transaction_costs_bool = (positions_signal == positions_signal.shift()).any(axis=1)
        transaction_costs[transaction_costs_bool] = 0

    # subtract transaction costs
    pnl_with_cost = (pnl - transaction_costs.shift())/daily_used_capital.shift()
    pnl_with_cost = pnl_with_cost.fillna(0)

    # calculate the percentage returns
    pct_pnl = pnl / daily_used_capital.shift()

    # if pnl is nan, set to 0
    pct_pnl = pct_pnl.fillna(0)
