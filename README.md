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

### Sample Images

![Cumulative Returns](./image/cumret.png?raw=true "Cumulative Returns")
![Drawdowns](./image/drawdown.png?raw=true "Drawdowns")

#### Sharpe Ratios

```
All set:
Strategy                           10.035522
Strategy with transaction costs     5.084462
SPY                                 1.262039

Train set:
Strategy                           10.620371
Strategy with transaction costs     5.291418
SPY                                 1.236930

Test set:
Strategy                           9.825243
Strategy with transaction costs    5.009323
SPY                                1.313724
```

#### Sortino Ratios

```
All set:
Strategy                           25.980593
Strategy with transaction costs    12.125281
SPY                                 2.160946

Train set:
Strategy                           29.553379
Strategy with transaction costs    12.778654
SPY                                 2.196900

Test set:
Strategy                           24.759964
Strategy with transaction costs    11.843611
SPY                                 2.333066
```

#### Max Drawdowns

```
All set:
Strategy                          -0.012960
Strategy with transaction costs   -0.019864
SPY                               -0.099037

Train set:
Strategy                          -0.005251
Strategy with transaction costs   -0.013484
SPY                               -0.079665

Test set:
Strategy                          -0.012960
Strategy with transaction costs   -0.019864
SPY                               -0.099037
```

#### Annualized Returns

```
All set:
Strategy                           0.659216
Strategy with transaction costs    0.291579
SPY                                0.180609

Train set:
Strategy                           0.634298
Strategy with transaction costs    0.275890
SPY                                0.232550

Test set:
Strategy                           0.669036
Strategy with transaction costs    0.297748
SPY                                0.160954
```

#### Calmar Ratios

```
All set:
Strategy                           50.866411
Strategy with transaction costs    14.678749
SPY                                 1.823650

Train set:
Strategy                           120.785411
Strategy with transaction costs     20.460122
SPY                                  2.919084

Test set:
Strategy                           51.624085
Strategy with transaction costs    14.989350
SPY                                 1.625185
```

### Reflection

The strategy does not seem to yield profits during paper trading, despite the observable mean-reverting characteristics in live data. The effectiveness of the strategy appears to be heavily influenced by the initial market spread at the opening. However, these spreads are swiftly arbitraged away as the market opens, resulting in the strategy's lack of profitability during paper trading due to latency in trade execution. Thus, the strategy slowly eats away at the initial capital in addition to the transaction costs.

![Alpaca Paper Trade](./image/alpaca_paper_trade.png?raw=true "Alpaca Paper Trade")
