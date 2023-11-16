# Mean Reversion Pairs Trading

### Introduction

Pairs trading is a market neutral trading strategy that involves buying and selling two highly correlated stocks. The strategy assumes that the two stocks will move in a similar direction and the prices will converge after a certain period of time. The strategy is based on the mean reversion principle, which states that the price of an asset will eventually revert back to its mean price. The strategy is often used by hedge funds and proprietary traders.

### Methodology

The methodology of this project is as follows:

1. Find a pair of stocks that are cointegrated
2. Calculate the spread between the two stocks
3. Calculate the standard deviation of the spread
4. Buy the spread when it is below certain threshold and sell the spread when it is above certain threshold

### Data

The data used in this project is from [Yahoo Finance](https://finance.yahoo.com/). The data is downloaded using the `yfinance` package.

### Model

The models used in this project are as follows:

1. Linear regression model
2. Kalman filter model
3. ARIMA model

### Backtesting

The results of the project are as follows:

1. The strategy is not profitable for majority of the pairs
2. The strategy is profitable for some pairs where they have a high cointegration score due to underlying economic reasons. For example, BRK-A and BRK-B are two stocks of the same company, so they are highly cointegrated. The strategy is profitable for this pair.

### Paper Trading

The strategy is paper traded using the [Alpaca](https://alpaca.markets/) API. 

