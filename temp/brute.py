
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

# Function to calculate sharpe ratio for a given pair of stocks
def calculate_sharpe(pair_data):
    stock_pair, data = pair_data
    sharpe, stock_pair = arima_brute(stock_pair, data.copy())
    return stock_pair, sharpe

def arima_brute(TICKER,prices):

    LEVERAGE = 1
    INVERT_STRATEGY = False

    prices = prices[TICKER]

    # split into train and test
    
    # train_size = len(prices.loc[:'2023-04-01'])
    train = prices
    x_train = train.iloc[:, 0]
    y_train = train.iloc[:, 1]


    # Use AR model to find the hedge ratio
    log_x_train = np.log(x_train)
    log_y_train = np.log(y_train)

    # log(x)-log(y) = log(beta) + s, where s noise
    log_beta_train = log_y_train-log_x_train
    # https://alkaline-ml.com/pmdarima/tips_and_tricks.html
    # test if ARIMA is appropriate to estimate beta
    model = pm.auto_arima(log_beta_train.values, start_p=1, start_q=1, max_p=5, max_q=5, m=1, seasonal=True)
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

    # get the intercept
    # if model.with_intercept:
    #     intercept = 
    # else:
    #     intercept = 0

    # predict the test set
  
    # convert logs to original prices
    pred_beta_train = np.exp(pred_log_beta_train)



  

    log_spread_train = log_y_train - log_x_train - pred_log_beta_train




    # state_vars = spread_train.ewm(span=30).var()
    pred_vars = pred_log_sigma2_train
    pred_sigma = np.sqrt(pred_vars)
    e = log_spread_train

    # Set initial positions to 0
    columns = ['positions_'+TICKER[0]+'_long', 'positions_'+TICKER[0]+'_short', 'positions_'+TICKER[1]+'_long', 'positions_'+TICKER[1]+'_short']
    for column in columns:
        prices[column] = 0

    # Entry conditions based on sqrt_Qt
    entry_threshold = 0*pred_sigma  # Replace with your specific calculation of sqrt_Qt
    # Short entry
    prices.loc[e > entry_threshold, ('positions_'+TICKER[0]+'_short', 'positions_'+TICKER[1]+'_short')] = [1, -1]
    # Long entry
    prices.loc[e < -entry_threshold, ('positions_'+TICKER[0]+'_long', 'positions_'+TICKER[1]+'_long')] = [-1, 1]

    # Exit conditions based on sqrt_Qt
    exit_threshold = 0*pred_sigma  # Set your desired exit threshold

    # let entry threshold be 1 standard deviations from rolling standard deviation
    # exit_threshold = 0*spread.rolling(lookback).std()

    # Close short position
    prices.loc[e < exit_threshold, ('positions_'+TICKER[0]+'_short', 'positions_'+TICKER[1]+'_short')] = 0

    # Close long position
    prices.loc[e > -exit_threshold, ('positions_'+TICKER[0]+'_long', 'positions_'+TICKER[1]+'_long')] = 0


    # ensure existing positions are carried forward
    prices.fillna(method='ffill', inplace=True)



    positions_long = prices.loc[:,('positions_'+TICKER[0]+'_long', 'positions_'+TICKER[1]+'_long')]
    positions_short = prices.loc[:,('positions_'+TICKER[0]+'_short', 'positions_'+TICKER[1]+'_short')]

    positions = np.array(positions_long) + np.array(positions_short)
    if INVERT_STRATEGY:
        positions = -positions
    positions = pd.DataFrame(positions, index=prices.index, columns=['positions_'+TICKER[0], 'positions_'+TICKER[1]])

    positions_sign = positions.copy()

    # multiply sposition of Tick1 by hedge ratio
    positions['positions_'+TICKER[0]] = positions['positions_'+TICKER[0]] * pred_beta_train

    # multiply positions by price to get dollar position
    positions["positions_"+TICKER[0]] = positions['positions_'+TICKER[0]] * prices[TICKER[0]]
    positions["positions_"+TICKER[1]] = positions['positions_'+TICKER[1]] * prices[TICKER[1]]




    # get daily returns
    daily_returns = prices.pct_change()[TICKER]

    # calculate weighted returns based on positions
    pnl = (np.array(positions.shift()) * np.array(daily_returns)).sum(axis=1)

    daily_used_capital = np.sum(np.abs(positions), axis=1)/2/LEVERAGE
    transaction_pct_cost = 0.0005
    transaction_costs = np.sum(np.abs(positions), axis=1)*transaction_pct_cost


    DAILY_REBALANCE = False

    if not DAILY_REBALANCE:
        # ignore transaction costs if positions_sign is not changed from previous day
        transaction_costs_bool = (positions_sign == positions_sign.shift()).any(axis=1)
        transaction_costs[transaction_costs_bool] = 0

    # subtract transaction costs
    pnl_with_cost = (pnl - transaction_costs.shift())/daily_used_capital.shift()
    pnl_with_cost = pnl_with_cost.fillna(0)

    # calculate the percentage returns
    pct_pnl = pnl / daily_used_capital.shift()

    # if pnl is nan, set to 0
    pct_pnl = pct_pnl.fillna(0)




    # def rolling_sharpe(pnl, rolling_period=100):
    #     return np.sqrt(252) * pnl.rolling(rolling_period).mean() / pnl.rolling(rolling_period).std()


    # def rolling_sortino(pnl, rolling_period=100):
    #     return np.sqrt(252) * pnl.rolling(rolling_period).mean() / pnl[pnl<0].rolling(rolling_period).std()

    # sharpe_list = [0]
    # for i in range(1, 100):
    #     # test optimal lookback period
    #     lookback = i
    #     copy = pnl_with_cost.copy()
    #     rolling_sharpe_strat = rolling_sortino(copy,lookback)
    #     copy[rolling_sharpe_strat.shift(2)<1] = 0
    #     sharpe = np.sqrt(252) * copy[1:].mean() / copy[1:].std()
    #     sharpe_list.append(sharpe)

    # # plot sharpe ratio
    # plt.plot(sharpe_list)

    # print(max(sharpe_list), np.argmax(sharpe_list))
        


    # plot rolling sharpe ratio

    def rolling_sharpe(pnl, rolling_period=100):
        return np.sqrt(252) * pnl.rolling(rolling_period).mean() / pnl.rolling(rolling_period).std()

    def rolling_sortino(pnl, rolling_period=100):
        return np.sqrt(252) * pnl.rolling(rolling_period).mean() / pnl[pnl<0].rolling(rolling_period).std()

    # set to 0 to ignore sharpe ratio
    rolling_period = 0

    rolling_sortino(pct_pnl,rolling_period).plot()

    # set the pnl to 0 when the rolling sharpe ratio is below 1
    rolling_sharpe_strat = rolling_sortino(pct_pnl,rolling_period)

    # .shift() to shift the rolling sharpe ratio by 1 day to avoid look ahead bias
    # yesterday's rolling sharpe ratio affect today's position decision
    positions[rolling_sharpe_strat.shift(1) < 1]= 0

    # .shift(2) to shift the rolling sharpe ratio by 2 days to avoid look ahead bias of pnl 
    # shift(2) as rolling sharpe affect today's position which affect tomorrow's pnl
    pct_pnl[rolling_sharpe_strat.shift(2)<1] = 0
    pnl_with_cost[rolling_sharpe_strat.shift(2)<1] = 0



    strats = pct_pnl 
    strats.columns = ['Strategy']

    def calculate_sharpe_ratio(returns :pd.Series) -> float:
        return np.sqrt(252) * returns.mean() / returns.std()

    def calculate_sortino_ratio(returns :pd.Series) -> float:
        return np.sqrt(252) * returns.mean() / returns[returns<0].std()

    def calculate_drawdown(returns: pd.Series) -> tuple:
        # calculate a cumulative wealth index
        wealth_index = (1 + returns).cumprod()

        # calculate previous peaks
        previous_peaks = wealth_index.cummax()

        # calculate drawdown
        drawdown = (wealth_index - previous_peaks) / previous_peaks
        max_drawdown = drawdown.min()
        max_drawdown_period = drawdown.idxmin()

        return drawdown, max_drawdown, max_drawdown_period

    def calculate_annualized_return(returns: pd.Series) -> float:
        cumulative_returns=(1+returns).cumprod()
        n_years = len(cumulative_returns)/252
        return (cumulative_returns.iloc[-1])**(1/n_years)-1

    def calmar_ratio(returns: pd.Series) -> float:
        _, max_drawdown, _ = calculate_drawdown(returns)
        annualized_return = calculate_annualized_return(returns)
        return annualized_return / abs(max_drawdown)
        


    strat_sets  = {'All': strats}


    # calculate sharpe ratio, sortino ratio, drawdown, max drawdown, max drawdown period, annualized return, calmar ratio
    sharpe_ratios = {}

    for key, value in strat_sets.items():
        sharpe_ratio = calculate_sharpe_ratio(value)
        sharpe_ratios[key] = sharpe_ratio
    return sharpe_ratios['All'], tuple(TICKER)
      

