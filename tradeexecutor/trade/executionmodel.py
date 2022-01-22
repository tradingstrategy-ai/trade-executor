import enum


class TradeExecutionModel(enum.Enum):
    """What kind of trade instruction execution model the strategy does"""

    #: Does not make any trades, just captures and logs them
    dummy = "dummy"

    #: Server-side normal Ethereum private eky account
    hot_wallet = "hot_wallet"

    #: Trading using Enzyme Protocol pool, single oracle mode
    single_oracle_pooled = "single_oracle_pooled"

    #: Trading using oracle network, oracles form a consensus using a judge smart contract
    multi_oracle_judged = "multi_oracle_judged"