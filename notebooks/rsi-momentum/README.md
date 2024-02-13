This folder contains examples for RSI-based momentum trading strategies for BTC and ETH.

- The evolution of the back test is labelled v1, v2, etc. from the simplest to most complex
- The backtest was performed with 2019-2024 Binance BTCUSDT and ETHUSDT data, daily time frame

The backtest evolution

- [v1](./v1-btc-spot-only.ipynb): The initial thesis of using RSI as a momentum indicator, trading BTC only. The starting parameters for technical indicators picked based on the earlier public studies from others.
    - This strategy is not losing money, but drags the performance of BTC
- [v2](./v2-eth-spot-only.ipynb): The same as above, but trading ETH. This validates whether or not the same strategy parameters can be used when trading ETH price action.
    - This strategy is not losing money, but drags the performance of BTC severely and has a massive drawdown.
- [v3](./v3-btc-eth.ipynb): The first multipair strategy version trading a portfolio of BTC, ETH and USDT. Any portfolio balancing is binary: you can have 100% BTC, 100% ETH or 50% BTC and 50% ETH.
    - This strategy profit is more than BTC, less than ETH, with some less drawdown than buy and hold ETH
- [v4](./v4-scaling-momentum-signal.ipynb): The multipair strategy, but instead of using a fixed binary allocation between assets, the allocation is dynamic based on ETH/BTC price.
    - This strategy profit is closer to buy and hold ETH, but with less drawdown.
- [v5](./v5-grid-search.ipynb): The grid search above to find optimised values for RSI low, RSI high and RSI length (days).
    - The optimal parameters outperform BTC and ETH
    - Based on the heatmap, there is a clear indication of parameter clustering and strategy parameters are not just random noise
- [v6](./v6-optimised.ipynb): Running the backtest results for the best grid searched pick
    - You can benchmark the optimal strategy against buy and hold ETH.