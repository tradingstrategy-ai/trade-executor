import enum


class TradeInstructionExecutionModel(enum.Enum):
    """What kind of trade instruction execution model the strategy does"""

    #: Server-side normal Ethereum private eky account
    hot_wallet = "hot_wallet"

    #: Trading using Enzyme Protocol pool, single oracle mode
    single_oracle_pooled = "single_oracle_pooled"

    #: Trading using oracle network, oracles form a consensus using a judge smart contract
    multi_oracle_judged = "multi_oracle_judged"