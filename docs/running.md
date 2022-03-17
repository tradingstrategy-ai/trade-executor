# Running Trade Executor on production 

These instructions are for the `master` branch bleeding edge testing.

First create a tmux session where you perform the work:

```shell
EXECUTOR_ID=pancake_8h_momentum_tick
tmux -CC -L $EXECUTOR_ID new    
```

# Create the environment

Clone the source code:

```shell
mkdir ~/trade-executor
cd ~/trade-executor
git clone --recurse-submodules https://github.com/tradingstrategy-ai/eth-hentai.git 
git clone --recurse-submodules https://github.com/tradingstrategy-ai/client.git 
git clone --recurse-submodules https://github.com/tradingstrategy-ai/trade-executor.git 
```

Create the Python environment for the executor:

```shell
cd trade-executor
poetry env use `which python3.9`
poetry shell
poetry install        
```

Create secrets file:

```shell
nano ~/$EXECUTOR_ID.secrets.env
```

Add required secrets:

```shell
export DISCORD_WEBHOOK_URL=
export PRIVATE_KEY=
export TRADING_STRATEGY_API_KEY=
```

You can combine this with generic secrets of JSON-RPC endpoints:

```shell
source ~/$EXECUTOR_ID.secrets.env
source ~/secrets.env
```

# Preflight check up

Check that the trading universe downloads correctly:

```shell
bootstraps/pancake_8h_momentum.sh check-universe --max-data-delay-minutes=1440   
```

```
Universe constructed.                    

Time periods
- Time frame 4h
- Candle data: 2021-04-23 08:00:00+00:00 - 2022-03-17 12:00:00+00:00

The size of our trading universe is
- 1 exchanges
- 3,730 pairs
- 1,230,381 candles
- 827,067 liquidity samples       
```


Then check your wallet balance looks correct:

```shell
bootstraps/pancake_8h_momentum.sh check-wallet
```

```
Balances of 0x4154fd0058a55dfDBBAEB7C340d8d42b21614Ed2
BUSD Token: 200 BUSD
```

# First run

Do the initial run. This run will try to make a trade immediately, assuming `MAX_DATA_DELAY_MINUTES` allows.
The first trade might not make sense, but you will that the algorithm is up'n'running.

```shell
bootstraps/pancake_8h_momentum.sh start --trade-immediately
```

# Subsequent runs

Start the executor with:

``shell
bootstraps/pancake_8h_momentum.sh start 
```

