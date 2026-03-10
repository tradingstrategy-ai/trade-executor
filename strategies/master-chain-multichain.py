"""Cross-chain vault of vaults strategy.

Based on `14-cross-hyperliquid.ipynb` notebook.
Cross-chain trading across Ethereum, Base, Arbitrum, Avalanche, Hypercore, HyperEVM, and Monad
to maximise yield diversification.

This is a multi-vault allocation strategy that:
- Selects from a universe of DeFi vaults across multiple chains
- Rebalances weekly based on rolling returns
- Caps individual position sizes and concentration
- Uses TVL-based filtering for vault inclusion

Backtest results (2025-01-08 to 2025-12-01)
=============================================

Last backtest run: 2026-03-10

================================  =========  ======  ======
Metric                            Strategy   BTC     ETH
================================  =========  ======  ======
Start period                      2025-01-06 2025-01-06 2025-01-06
End period                        2025-11-24 2025-11-24 2025-11-24
Risk-free rate                    0.0%       0.0%    0.0%
Time in market                    15.0%      98.0%   98.0%
Cumulative return                 36.63%     -6.09%  -9.54%
CAGR﹪                             42.45%     -6.87%  -10.75%
Sharpe                            4.7        0.04    0.23
Probabilistic Sharpe ratio        100.0%     51.47%  58.6%
Smart Sharpe                      4.43       0.04    0.22
Sortino                           45.21      0.06    0.35
Smart Sortino                     42.59      0.05    0.33
Sortino/√2                        31.97      0.04    0.25
Smart Sortino/√2                  30.12      0.04    0.23
Omega                             34.65      34.65   34.65
Max drawdown                      -0.7%      -32.2%  -57.64%
Longest DD days                   14         116     180
Volatility (ann.)                 7.57%      41.91%  76.74%
Calmar                            60.43      -0.21   -0.19
================================  =========  ======  ======
"""

#
# Imports
#

import datetime
import logging

import pandas as pd
from eth_defi.token import USDC_NATIVE_TOKEN
from eth_defi.vault.vaultdb import DEFAULT_RAW_PRICE_DATABASE, DEFAULT_VAULT_DATABASE
from plotly.graph_objects import Figure
from tradingstrategy.alternative_data.vault import load_vault_database
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.utils.token_filter import filter_for_selected_pairs

from tradeexecutor.analysis.vault import display_vaults
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.alpha_model import AlphaModel
from tradeexecutor.strategy.chart.definition import (ChartInput, ChartKind,
                                                     ChartRegistry)
from tradeexecutor.strategy.chart.standard.alpha_model import \
    alpha_model_diagnostics
from tradeexecutor.strategy.chart.standard.equity_curve import (
    equity_curve, equity_curve_with_drawdown)
from tradeexecutor.strategy.chart.standard.interest import (
    lending_pool_interest_accrued, vault_statistics)
from tradeexecutor.strategy.chart.standard.performance_metrics import \
    performance_metrics
from tradeexecutor.strategy.chart.standard.position import positions_at_end
from tradeexecutor.strategy.chart.standard.profit_breakdown import \
    trading_pair_breakdown
from tradeexecutor.strategy.chart.standard.signal import (price_vs_signal,
                                                          signal_comparison)
from tradeexecutor.strategy.chart.standard.single_pair import (
    trading_pair_positions, trading_pair_price_and_trades)
from tradeexecutor.strategy.chart.standard.thinking import last_messages
from tradeexecutor.strategy.chart.standard.trading_metrics import \
    trading_metrics
from tradeexecutor.strategy.chart.standard.trading_universe import (
    available_trading_pairs, inclusion_criteria_check)
from tradeexecutor.strategy.chart.standard.vault import all_vault_positions
from tradeexecutor.strategy.chart.standard.vault import \
    all_vaults_share_price_and_tvl as _all_vaults_share_price_and_tvl
from tradeexecutor.strategy.chart.standard.vault import vault_position_timeline
from tradeexecutor.strategy.chart.standard.volatility import \
    volatility_benchmark
from tradeexecutor.strategy.chart.standard.weight import (
    equity_curve_by_asset, volatile_and_non_volatile_percent,
    volatile_weights_by_percent, weight_allocation_statistics)
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import (ExecutionContext,
                                                      ExecutionMode)
from tradeexecutor.strategy.pandas_trader.indicator import (
    IndicatorDependencyResolver, IndicatorSource)
from tradeexecutor.strategy.pandas_trader.indicator_decorator import \
    IndicatorRegistry
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.pandas_trader.trading_universe_input import \
    CreateTradingUniverseInput
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.tag import StrategyTag
from tradeexecutor.strategy.trading_strategy_universe import (
    TradingStrategyUniverse, load_partial_data)
from tradeexecutor.strategy.tvl_size_risk import USDTVLSizeRiskModel
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.strategy.weighting import weight_passthrouh
from tradeexecutor.utils.dedent import dedent_any

logger = logging.getLogger(__name__)

#
# Trading universe constants
#

trading_strategy_engine_version = "0.5"

CHAIN_ID = ChainId.cross_chain
PRIMARY_CHAIN_ID = ChainId.arbitrum

# Extend ChainId with hypercore (chain_id=9999) as it is not yet in the enum
_hypercore = int.__new__(ChainId, 9999)
_hypercore._name_ = "hypercore"
_hypercore._value_ = 9999
ChainId._value2member_map_[9999] = _hypercore
ChainId._member_map_["hypercore"] = _hypercore
type.__setattr__(ChainId, "hypercore", _hypercore)
HYPERCORE_CHAIN_ID = ChainId.hypercore

EXCHANGES = ("uniswap-v2", "uniswap-v3")

SUPPORTING_PAIRS = [
    (ChainId.arbitrum, "uniswap-v3", "WETH", "USDC", 0.0005),
    (ChainId.ethereum, "uniswap-v3", "WETH", "USDC", 0.0005),
    (ChainId.ethereum, "uniswap-v3", "WBTC", "USDC", 0.003),
]

LENDING_RESERVES = None

PREFERRED_STABLECOIN = USDC_NATIVE_TOKEN[PRIMARY_CHAIN_ID].lower()

# Data source: https://top-defi-vaults.tradingstrategy.ai/top_vaults_by_chain.json
# Denomination filter: USDC, USDC.e, crvUSD, USDT (and variants USD₮0, USDt, USDT0)
# Excluded: risk=Blacklisted or risk=Dangerous
# Min TVL: $100k (except Hypercore $500k)
# Must-include: Ostium (Arbitrum), Growi HF (Hypercore)
# Sorted by 1y CAGR, top 10 per chain
VAULTS = [

    # Ethereum (10 vaults, sorted by 1y CAGR)
    (ChainId.ethereum, "0x0e297de4005883c757c9f09fdf7cf1363c20e626"),  # [USDC] Morpho Yearn OG USDC Compounder (age=0.4y, cagr_1y=414.28%, cagr_all=414.28%)
    (ChainId.ethereum, "0x66dcb62da5430d800a1c807822c25be17138fda8"),  # [USDC] Unity Trust (age=0.2y, cagr_1y=225.89%, cagr_all=225.89%)
    (ChainId.ethereum, "0x6d2981ff9b8d7edbb7604de7a65bac8694ac849f"),  # [USDT] Morpho Gauntlet USDT Prime Compounder (age=1.2y, cagr_1y=141.58%, cagr_all=150.86%)
    (ChainId.ethereum, "0x77a63952572dd0eaa7fd21a3fdeaa80b4071a5e8"),  # [USDC] BordelMortgageVaultShare (age=0.2y, cagr_1y=43.57%, cagr_all=43.57%)
    (ChainId.ethereum, "0x438982ea288763370946625fd76c2508ee1fb229"),  # [USDC] cSuperior Quality Private Credit USDC (age=1.3y, cagr_1y=39.29%, cagr_all=35.13%)
    (ChainId.ethereum, "0x056b269eb1f75477a8666ae8c7fe01b64dd55ecc"),  # [USDC] USD3 (age=0.5y, cagr_1y=31.66%, cagr_all=31.66%)
    (ChainId.ethereum, "0x09c4c7b1d2e9aa7506db8b76f1dbbd61c08c114b"),  # [USDC] Everstone (age=0.3y, cagr_1y=22.82%, cagr_all=22.82%)
    (ChainId.ethereum, "0x01f461a0bbb218bc1943aa027c5bbc424391e541"),  # [USDT] DeltaUSD HyperLiquid USDN Funding Arb (age=0.6y, cagr_1y=20.60%, cagr_all=20.60%)
    (ChainId.ethereum, "0x64c8159af38aa87d2fc8bca1e4f22076dd752558"),  # [USDT] Leveraged sUSDD Vault (age=0.0y, cagr_1y=19.38%, cagr_all=19.38%)
    (ChainId.ethereum, "0xbeefff4716a49418d69c251cab8759bb107e57c8"),  # [USDC] Steakhouse High Yield Turbo (age=0.2y, cagr_1y=17.87%, cagr_all=17.87%)

    # Base (10 vaults, sorted by 1y CAGR)
    (ChainId.base, "0xf7e26fa48a568b8b0038e104dfd8abdf0f99074f"),  # [USDC] Muscadine USDC Vault (age=0.7y, cagr_1y=172.17%, cagr_all=172.17%)
    (ChainId.base, "0x67b93f6676bd1911c5fae7ffa90fff5f35e14dcd"),  # [USDC] Base-USDC Yield-DynaVault v3 (age=0.1y, cagr_1y=53.68%, cagr_all=53.68%)
    (ChainId.base, "0x3094b241aade60f91f1c82b0628a10d9501462f9"),  # [USDC] Mo Earn Max USDC (age=0.4y, cagr_1y=22.89%, cagr_all=22.89%)
    (ChainId.base, "0xb99b6df96d4d5448cc0a5b3e0ef7896df9507cf5"),  # [USDC] Vault (age=1.1y, cagr_1y=14.95%, cagr_all=18.23%)
    (ChainId.base, "0x70fffbacb53ef74903ac074aae769414a70970d1"),  # [USDC] USDC Fluid Lender (age=0.5y, cagr_1y=13.47%, cagr_all=13.47%)
    (ChainId.base, "0x3ec4a293fb906dd2cd440c20decb250def141df1"),  # [USDC] ArcadiaV2 USD Coin Debt (age=2.0y, cagr_1y=12.50%, cagr_all=13.02%)
    (ChainId.base, "0x61a8606e04d350dfa1d1aaa68b37260746ae47d4"),  # [USDC] Tulipa Credit Vault (age=0.1y, cagr_1y=12.08%, cagr_all=12.08%)
    (ChainId.base, "0x8092ca384d44260ea4feaf7457b629b8dc6f88f0"),  # [USDC] DeTrade Core USDC (age=1.0y, cagr_1y=12.06%, cagr_all=12.06%)
    (ChainId.base, "0xc777031d50f632083be7080e51e390709062263e"),  # [USDC] Harvest: USDC Vault (0xC777) (age=0.8y, cagr_1y=10.86%, cagr_all=10.86%)
    (ChainId.base, "0xad20523a7dc37babc1cc74897e4977232b3d02e5"),  # [USDC] gTrade (Gains Network USDC) (age=1.4y, cagr_1y=10.34%, cagr_all=12.30%)

    # Arbitrum (10 vaults, sorted by 1y CAGR)
    # Must-include: Ostium Liquidity Pool Vault
    (ChainId.arbitrum, "0x75288264fdfea8ce68e6d852696ab1ce2f3e5004"),  # [USDC] HYPE++ (age=1.3y, cagr_1y=26.30%, cagr_all=32.04%)
    (ChainId.arbitrum, "0x58bfc95a864e18e8f3041d2fcd3418f48393fe6a"),  # [USDC] Plutus Hedge Token (age=1.1y, cagr_1y=21.21%, cagr_all=19.19%)
    (ChainId.arbitrum, "0xf63b7f49b4f5dc5d0e7e583cfd79dc64e646320c"),  # [USDC] Tokemak arbUSD (age=0.5y, cagr_1y=20.92%, cagr_all=20.92%)
    (ChainId.arbitrum, "0x1723cb57af58efb35a013870c90fcc3d60174a4e"),  # [USDC] Angmar Capital (age=0.4y, cagr_1y=19.33%, cagr_all=19.33%)
    (ChainId.arbitrum, "0x20d419a8e12c45f88fda7c5760bb6923cee27f98"),  # [USDC] Ostium Liquidity Pool Vault (age=1.7y, cagr_1y=12.73%, cagr_all=9.67%)
    (ChainId.arbitrum, "0xc8248953429d707c6a2815653eca89846ffaa63b"),  # [crvUSD] Llama Lend asdCRV / crvUSD (age=1.7y, cagr_1y=11.74%, cagr_all=13.33%)
    (ChainId.arbitrum, "0x4739e2c293bdcd835829aa7c5d7fbdee93565d1a"),  # [USD₮0] Steakhouse High Yield USDT0 (age=0.5y, cagr_1y=11.34%, cagr_all=11.34%)
    (ChainId.arbitrum, "0xd3443ee1e91af28e5fb858fbd0d72a63ba8046e0"),  # [USDC] gTrade (Gains Network USDC) (age=2.1y, cagr_1y=9.71%, cagr_all=12.46%)
    (ChainId.arbitrum, "0x9fa306b1f4a6a83fec98d8ebbabedff78c407f6b"),  # [USDC.e] USDC-2 yVault (age=1.7y, cagr_1y=8.48%, cagr_all=8.09%)
    (ChainId.arbitrum, "0xa7781f1d982eb9000bc1733e29ff5ba2824cdbe5"),  # [crvUSD] Yield Chasing crvUSD (age=1.3y, cagr_1y=8.28%, cagr_all=10.44%)

    # Avalanche (10 vaults, sorted by 1y CAGR)
    (ChainId.avalanche, "0xbed7c02887efd6b5eb9a547ac1a4d5e582791647"),  # [USDC] 40avax-USDC-VAULT (age=0.8y, cagr_1y=10000.00%, cagr_all=10000.00%)
    (ChainId.avalanche, "0xc0485c4bafb594ae1457820fb6e5b67e8a04bcfd"),  # [USDC] 40BH-USDC-VAULT (age=0.4y, cagr_1y=20.35%, cagr_all=20.35%)
    (ChainId.avalanche, "0x124d00b1ce4453ffc5a5f65ce83af13a7709bac7"),  # [USDC] 40avax-USDC-VAULT (age=0.7y, cagr_1y=14.83%, cagr_all=14.83%)
    (ChainId.avalanche, "0x606fe9a70338e798a292ca22c1f28c829f24048e"),  # [USDC] Borrowable USDC Deposit, SiloId: 142 (age=0.5y, cagr_1y=9.84%, cagr_all=9.84%)
    (ChainId.avalanche, "0x4af3abe954259fb70b97c57ebd7ac1eb822028ef"),  # [USDC] 9Summits USDC (age=0.4y, cagr_1y=7.87%, cagr_all=7.87%)
    (ChainId.avalanche, "0x37ca03ad51b8ff79aad35fadacba4cedf0c3e74e"),  # [USDC] 9Summits USDC (age=0.4y, cagr_1y=7.81%, cagr_all=7.81%)
    (ChainId.avalanche, "0x3048925b3ea5a8c12eecccb8810f5f7544db54af"),  # [USDC] Turtle Avalanche USDC (age=0.1y, cagr_1y=6.43%, cagr_all=6.43%)
    (ChainId.avalanche, "0x39de0f00189306062d79edec6dca5bb6bfd108f9"),  # [USDC] Re7 Labs USDC (age=0.9y, cagr_1y=5.54%, cagr_all=5.54%)
    (ChainId.avalanche, "0x9fd32fd5e32c6b95483d36c5e724c5c5250ce010"),  # [USDC] ygamiUSDC (age=0.3y, cagr_1y=4.46%, cagr_all=4.46%)
    (ChainId.avalanche, "0x8f23da78e3f31ab5deb75dc3282198bed630ffde"),  # [USDC] Keyring USDC (age=0.6y, cagr_1y=4.45%, cagr_all=4.45%)

    # Hypercore (10 vaults, sorted by 1y CAGR)
    # Must-include: Growi HF
    (HYPERCORE_CHAIN_ID, "0x4cb5f4d145cd16460932bbb9b871bb6fd5db97e3"),  # [USDC] Not In Employment Education or Training (age=0.3y, cagr_1y=10000.00%, cagr_all=10000.00%)
    (HYPERCORE_CHAIN_ID, "0xca230e816bdb34a46960c2f978a30a563d1ae9e0"),  # [USDC] Hyperrr (age=1.2y, cagr_1y=10000.00%, cagr_all=10000.00%)
    (HYPERCORE_CHAIN_ID, "0xe67dbf2d051106b42104c1a6631af5e5a458b682"),  # [USDC] Overdose (age=0.2y, cagr_1y=3236.70%, cagr_all=3236.70%)
    (HYPERCORE_CHAIN_ID, "0xd6e56265890b76413d1d527eb9b75e334c0c5b42"),  # [USDC] Systemic Strategies HyperGrowth (age=0.5y, cagr_1y=197.84%, cagr_all=197.84%)
    (HYPERCORE_CHAIN_ID, "0xbbf7d7a9d0eaeab4115f022a6863450296112422"),  # [USDC] Satori Quantum HF Vault (age=1.0y, cagr_1y=187.02%, cagr_all=187.02%)
    (HYPERCORE_CHAIN_ID, "0x61b1cf5c2d7c4bf6d5db14f36651b2242e7cba0a"),  # [USDC] OnlyShorts (age=0.4y, cagr_1y=183.34%, cagr_all=183.34%)
    (HYPERCORE_CHAIN_ID, "0xac26cf5f3c46b5e102048c65b977d2551b72a9c7"),  # [USDC] Long HYPE & BTC | Short Garbage (age=0.9y, cagr_1y=150.13%, cagr_all=150.13%)
    (HYPERCORE_CHAIN_ID, "0xda51323fe9800c8365646ad5c7ade0dd17fdc167"),  # [USDC] Citadel (age=2.3y, cagr_1y=148.57%, cagr_all=422.89%)
    (HYPERCORE_CHAIN_ID, "0x1e37a337ed460039d1b15bd3bc489de789768d5e"),  # [USDC] Growi HF (age=1.7y, cagr_1y=53.07%, cagr_all=69.97%)

    # HyperEVM (10 vaults, sorted by 1y CAGR)
    (ChainId.hyperliquid, "0xd0ee0cf300dfb598270cd7f4d0c6e0d8f6e13f29"),  # [USD₮0] Altura Vault Tokens (age=0.2y, cagr_1y=22.40%, cagr_all=22.40%)
    (ChainId.hyperliquid, "0x06fd9d03b3d0f18e4919919b72d30c582f0a97e5"),  # [USD₮0] Wrapped HLP (age=0.7y, cagr_1y=12.30%, cagr_all=12.30%)
    (ChainId.hyperliquid, "0x1c5164a764844356d57654ea83f9f1b72cd10db5"),  # [USD₮0] hyUSD₮0 (Looped HYPE) - 9 (age=0.7y, cagr_1y=11.76%, cagr_all=11.76%)
    (ChainId.hyperliquid, "0x2c910f67dbf81099e6f8e126e7265d7595dc20ad"),  # [USD₮0] hyUSD₮0 (hwHLP) - 11 (age=0.7y, cagr_1y=9.86%, cagr_all=9.86%)
    (ChainId.hyperliquid, "0x195eb4d088f222c982282b5dd495e76dba4bc7d1"),  # [USD₮0] HYPE++ Hyperliquid (age=0.5y, cagr_1y=8.34%, cagr_all=8.34%)
    (ChainId.hyperliquid, "0xe5add96840f0b908ddeb3bd144c0283ac5ca7ca0"),  # [USD₮0] Hyperithm USDT0 (age=0.6y, cagr_1y=8.27%, cagr_all=8.27%)
    (ChainId.hyperliquid, "0x9896a8605763106e57a51aa0a97fe8099e806bb3"),  # [USD₮0] Felix USDT0 (Frontier) (age=0.6y, cagr_1y=8.13%, cagr_all=8.13%)
    (ChainId.hyperliquid, "0x08c00f8279dff5b0cb5a04d349e7d79708ceadf3"),  # [USDC] Gauntlet USDC (age=0.4y, cagr_1y=7.37%, cagr_all=7.37%)
    (ChainId.hyperliquid, "0xfc5126377f0efc0041c0969ef9ba903ce67d151e"),  # [USD₮0] Felix USDT0 (age=0.8y, cagr_1y=7.25%, cagr_all=7.25%)
    (ChainId.hyperliquid, "0x8a862fd6c12f9ad34c9c2ff45ab2b6712e8cea27"),  # [USDC] Felix USDC (age=0.5y, cagr_1y=6.75%, cagr_all=6.75%)

    # Monad (10 vaults, sorted by 1y CAGR)
    (ChainId.monad, "0x3a2c4aaae6776dc1c31316de559598f2f952e2cb"),  # [USDC] Yuzu Money Vault (age=0.2y, cagr_1y=8.78%, cagr_all=8.78%)
    (ChainId.monad, "0x7cd231120a60f500887444a9baf5e1bd753a5e59"),  # [USDC] Hyperithm Delta Neutral Vault (age=0.1y, cagr_1y=8.43%, cagr_all=8.43%)
    (ChainId.monad, "0x58ba69b289de313e66a13b7d1f822fc98b970554"),  # [USDC] sUSN Delta Neutral Yield Vault (age=0.3y, cagr_1y=8.32%, cagr_all=8.32%)
    (ChainId.monad, "0x8d5c2df3eef09088fcccf3376d8ecd0dd505f642"),  # [USDC] Wrapped Neverland USDC (age=0.1y, cagr_1y=4.37%, cagr_all=4.37%)
    (ChainId.monad, "0x78999cc96d2ba0341588c60ccb0e91c6c33cf371"),  # [USDC] Hyperithm USDC Degen (age=0.1y, cagr_1y=4.29%, cagr_all=4.29%)
    (ChainId.monad, "0xa8665084d8cd6276c00ca97cbc0bf4bc9ae94c79"),  # [USDC] Hyperithm USDC Degen (age=0.1y, cagr_1y=4.29%, cagr_all=4.29%)
    (ChainId.monad, "0x4e8aaecce10ad9394e96fe5f2bd4e587a7b04298"),  # [USDT0] Wrapped Neverland USDT0 (age=0.0y, cagr_1y=4.04%, cagr_all=4.04%)
    (ChainId.monad, "0x4c0d041889281531ff060290d71091401caa786d"),  # [USDC] Asia Credit Yield Vault (age=0.3y, cagr_1y=3.84%, cagr_all=3.84%)
    (ChainId.monad, "0x8ee9fc28b8da872c38a496e9ddb9700bb7261774"),  # [USDC] Curvance USDC (age=0.3y, cagr_1y=3.76%, cagr_all=3.76%)
    (ChainId.monad, "0x0da39b740834090c146dc48357f6a435a1bb33b3"),  # [USDC] MuDigital Tulipa USDC (age=0.3y, cagr_1y=3.57%, cagr_all=3.57%)
]

BENCHMARK_PAIRS = [
    (ChainId.ethereum, "uniswap-v3", "WBTC", "USDC", 0.003),
]

# Exclude Euro vaults, etc.
ALLOWED_VAULT_DENOMINATION_TOKENS = {"USDC", "USDT", "USDC.e", "crvUSD", "USD₮0", "USDt", "USDT0"}

#
# Strategy parameters
#


class Parameters:

    id = "master-chain-multichain"

    # We trade 1d candle
    candle_time_bucket = TimeBucket.d1
    cycle_duration = CycleDuration.cycle_7d

    chain_id = CHAIN_ID
    primary_chain_id = PRIMARY_CHAIN_ID
    exchanges = EXCHANGES

    #
    # Basket size, risk and balancing parameters.
    #
    min_asset_universe = 5  # How many assets we need in the asset universe to start running the index
    max_assets_in_portfolio = 10  # How many assets our basket can hold once
    allocation = 0.95  # Allocate all cash to volatile pairs
    individual_rebalance_min_threshold_usd = 500.0  # Don't make buys less than this amount
    sell_rebalance_min_threshold = 100.0
    sell_threshold = 0.05  # Sell if asset is more than 5% of the portfolio
    per_position_cap_of_pool = 0.20  # Never own more than % of the lit liquidity of the trading pool
    max_concentration = 0.20  # How large % can one asset be in a portfolio once
    min_portfolio_weight = 0.0050  # Close position / do not open if weight is less than 50 BPS

    # How long
    # Needed to calculate weights
    rolling_returns_bars = 7

    min_tvl = 50_000  # Minimum TVL in the vault before it can be considered investable

    #
    # Backtesting only
    #
    backtest_start = datetime.datetime(2025, 1, 1)
    backtest_end = datetime.datetime(2025, 12, 1)
    initial_cash = 100_000

    #
    # Live only
    #
    routing = TradeRouting.default
    required_history_period = datetime.timedelta(days=365 * 3)
    slippage_tolerance = 0.0060  # 0.6%
    assummed_liquidity_when_data_missings = 10_000


#
# Universe creation
#


def create_trading_universe(
    input: CreateTradingUniverseInput,
) -> TradingStrategyUniverse:
    """Create the trading universe.

    - Load Trading Strategy full pairs dataset

    - Load built-in Coingecko top 1000 dataset

    - Get all DEX tokens for a certain Coigecko category

    - Load OHCLV data for these pairs

    - Load also BTC and ETH price data to be used as a benchmark
    """

    execution_context = input.execution_context
    client = input.client
    timestamp = input.timestamp
    parameters = input.parameters or Parameters  # Some CLI commands do not support yet passing this
    universe_options = input.universe_options

    if execution_context.live_trading:
        # Live trading, send strategy universe formation details
        # to logs
        debug_printer = logger.info
    else:
        # Jupyter notebook inline output
        debug_printer = print

    all_chain_names = sorted(set(c.get_name() for c, _ in VAULTS))
    debug_printer(f"Preparing trading universe on chains: {', '.join(all_chain_names)}")

    # Pull out our benchmark pairs ids.
    # We need to construct pair universe object for the symbolic lookup.
    all_pairs_df = client.fetch_pair_universe().to_pandas()
    pairs_df = filter_for_selected_pairs(
        all_pairs_df,
        SUPPORTING_PAIRS,
    )

    debug_printer(f"We have total {len(all_pairs_df)} pairs in dataset and going to use {len(pairs_df)} pairs for the strategy")

    # Check which vaults we can include based on allowed deposit tokens for this backtest
    # Use local vault database (~/.tradingstrategy/vaults/) instead of
    # the bundled one, as the bundled one may not have all chains
    vault_universe = load_vault_database(path=DEFAULT_VAULT_DATABASE)
    total_vaults = vault_universe.get_vault_count()
    vault_universe = vault_universe.limit_to_vaults(VAULTS, check_all_vaults_found=True)
    vault_universe = vault_universe.limit_to_denomination(ALLOWED_VAULT_DENOMINATION_TOKENS, check_all_vaults_found=True)
    debug_printer(f"Loaded total {vault_universe.get_vault_count()} vaults from the total of {total_vaults} in vault database, source vaults count: {len(VAULTS)}")

    # Default vault data bundle path for backtesting
    vault_bundled_price_data = DEFAULT_RAW_PRICE_DATABASE
    debug_printer(f"Using vault price data for backtesting from {vault_bundled_price_data}")

    dataset = load_partial_data(
        client=client,
        time_bucket=parameters.candle_time_bucket,
        pairs=pairs_df,
        execution_context=execution_context,
        universe_options=universe_options,
        liquidity=True,
        liquidity_time_bucket=TimeBucket.d1,
        lending_reserves=LENDING_RESERVES,
        vaults=vault_universe,
        vault_bundled_price_data=vault_bundled_price_data if not execution_context.live_trading else None,
        check_all_vaults_found=True,
    )

    debug_printer("Creating strategy universe with price feeds and vaults")
    strategy_universe = TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset=PREFERRED_STABLECOIN,
        forward_fill=True,  # We got very gappy data from low liquid DEX coins
        forward_fill_until=timestamp,
        primary_chain=parameters.primary_chain_id,
    )

    # crvUSD etc. do not have backtesting paths yet
    strategy_universe.ignore_routing = True

    # Dump our vault data and check for data errors
    display_vaults(
        vault_universe,
        strategy_universe,
        execution_mode=execution_context.mode,
        printer=debug_printer,
    )

    return strategy_universe


#
# Strategy logic
#


_cached_start_times: dict[int, pd.Timestamp] = {}


def decide_trades(
    input: StrategyInput
) -> list[TradeExecution]:
    """For each strategy tick, generate the list of trades."""
    parameters = input.parameters
    position_manager = input.get_position_manager()
    state = input.state
    timestamp = input.timestamp
    indicators = input.indicators
    strategy_universe = input.strategy_universe

    portfolio = position_manager.get_current_portfolio()
    equity = portfolio.get_total_equity()

    # All gone, stop doing decisions
    if input.execution_context.mode == ExecutionMode.backtesting:
        if equity < parameters.initial_cash * 0.10:
            return []

    # Build signals for each pair
    alpha_model = AlphaModel(
        timestamp,
        close_position_weight_epsilon=parameters.min_portfolio_weight,  # 10 BPS is our min portfolio weight
    )

    tvl_included_pair_count = indicators.get_indicator_value(
        "tvl_included_pair_count",
    )

    # Get pairs included in this rebalance cycle.
    # This includes pair that have been pre-cleared in inclusion_criteria()
    # with volume, volatility and TVL filters
    included_pairs = indicators.get_indicator_value(
        "inclusion_criteria",
        na_conversion=False,
    )
    if included_pairs is None:
        included_pairs = []

    # Set signal for each pair
    signal_count = 0
    for pair_id in included_pairs:
        pair = strategy_universe.get_pair_by_id(pair_id)

        if not state.is_good_pair(pair):
            # Tradeable flag set to False, etc.
            continue

        pair_signal = indicators.get_indicator_value("signal", pair=pair)
        if pair_signal is None:
            continue

        weight = pair_signal

        if weight < 0:
            continue

        alpha_model.set_signal(
            pair,
            weight,
        )

        # Diagnostics reporting
        signal_count += 1

    # Calculate how much dollar value we want each individual position to be on this strategy cycle,
    # based on our total available equity
    portfolio = position_manager.get_current_portfolio()
    equity = portfolio.get_total_equity()
    portfolio_target_value = equity * parameters.allocation

    # Select max_assets_in_portfolio assets in which we are going to invest
    # Calculate a weight for ecah asset in the portfolio using 1/N method based on the raw signal
    alpha_model.select_top_signals(count=parameters.max_assets_in_portfolio)
    alpha_model.assign_weights(method=weight_passthrouh)

    #
    # Normalise weights and cap the positions
    #
    size_risk_model = USDTVLSizeRiskModel(
        pricing_model=input.pricing_model,
        per_position_cap=parameters.per_position_cap_of_pool,  # This is how much % by all pool TVL we can allocate for a position
        missing_tvl_placeholder_usd=0.0,  # Placeholder for missing TVL data until we get the data off the chain
    )

    alpha_model.normalise_weights(
        investable_equity=portfolio_target_value,
        size_risk_model=size_risk_model,
        max_weight=parameters.max_concentration,
    )

    # Load in old weight for each trading pair signal,
    # so we can calculate the adjustment trade size
    alpha_model.update_old_weights(
        state.portfolio,
        ignore_credit=False,
    )
    alpha_model.calculate_target_positions(position_manager)

    # Shift portfolio from current positions to target positions
    # determined by the alpha signals (momentum)

    rebalance_threshold_usd = parameters.individual_rebalance_min_threshold_usd

    assert rebalance_threshold_usd > 0.1, "Safety check tripped - something like wrong with strat code"
    trades = alpha_model.generate_rebalance_trades_and_triggers(
        position_manager,
        min_trade_threshold=rebalance_threshold_usd,  # Don't bother with trades under XXXX USD
        invidiual_rebalance_min_threshold=parameters.individual_rebalance_min_threshold_usd,
        sell_rebalance_min_threshold=parameters.sell_rebalance_min_threshold,
        execution_context=input.execution_context,
    )

    # Add verbal report about decision made/not made,
    # so it is much easier to diagnose live trade execution.
    # This will be readable in Discord/Telegram logging.
    if input.is_visualisation_enabled():
        try:
            top_signal = next(iter(alpha_model.get_signals_sorted_by_weight()))
            if top_signal.normalised_weight == 0:
                top_signal = None
        except StopIteration:
            top_signal = None

        rebalance_volume = sum(t.get_value() for t in trades)

        report = dedent_any(f"""
        Cycle: #{input.cycle}
        Rebalanced: {'👍' if alpha_model.is_rebalance_triggered() else '👎'}
        Open/about to open positions: {len(state.portfolio.open_positions)}
        Max position value change: {alpha_model.max_position_adjust_usd:,.2f} USD
        Rebalance threshold: {alpha_model.position_adjust_threshold_usd:,.2f} USD
        Trades decided: {len(trades)}
        Pairs total: {strategy_universe.data_universe.pairs.get_count()}
        Pairs meeting inclusion criteria: {len(included_pairs)}
        Pairs meeting TVL inclusion criteria: {tvl_included_pair_count}
        Signals created: {signal_count}
        Total equity: {portfolio.get_total_equity():,.2f} USD
        Cash: {position_manager.get_current_cash():,.2f} USD
        Investable equity: {alpha_model.investable_equity:,.2f} USD
        Accepted investable equity: {alpha_model.accepted_investable_equity:,.2f} USD
        Allocated to signals: {alpha_model.get_allocated_value():,.2f} USD
        Discarted allocation because of lack of lit liquidity: {alpha_model.size_risk_discarded_value:,.2f} USD
        Rebalance volume: {rebalance_volume:,.2f} USD
        """)

        if top_signal:
            assert top_signal.position_size_risk
            report += dedent_any(f"""
            Top signal pair: {top_signal.pair.get_ticker()}
            Top signal value: {top_signal.signal}
            Top signal weight: {top_signal.raw_weight}
            Top signal weight (normalised): {top_signal.normalised_weight * 100:.2f} % (got {top_signal.position_size_risk.get_relative_capped_amount() * 100:.2f} % of asked size)
            """)

        for flag, count in alpha_model.get_flag_diagnostics_data().items():
            report += f"Signals with flag {flag.name}: {count}\n"

        state.visualisation.add_message(
            timestamp,
            report,
        )

        state.visualisation.set_discardable_data("alpha_model", alpha_model)

    return trades  # Return the list of trades we made in this cycle


#
# Indicators
#

empty_series = pd.Series([], index=pd.DatetimeIndex([]))

indicators = IndicatorRegistry()


@indicators.define()
def rolling_returns(
    close: pd.Series,
    rolling_returns_bars: int = 60,
) -> pd.Series:
    """Calculate rolling returns over a period"""

    windowed = close.rolling(
        window=rolling_returns_bars,
        min_periods=2,
    ).max()
    series = (close / windowed)
    return series


@indicators.define(source=IndicatorSource.tvl)
def tvl(
    close: pd.Series,
    execution_context: ExecutionContext,
    timestamp: pd.Timestamp,
) -> pd.Series:
    """Get TVL series for a pair.

    - Because TVL data is 1d and we use 1h everywhere else, we need to forward fill

    - Use previous hourly close as the value
    """
    if execution_context.live_trading:
        # TVL is daily data.
        # We need to forward fill until the current hour.
        # Use our special ff function.
        assert isinstance(timestamp, pd.Timestamp), f"Live trading needs forward-fill end time, we got {timestamp}"
        from tradingstrategy.utils.forward_fill import forward_fill
        df = pd.DataFrame({"close": close})
        df_ff = forward_fill(
            df,
            Parameters.candle_time_bucket.to_frequency(),
            columns=("close",),
            forward_fill_until=timestamp,
        )
        series = df_ff["close"]
        return series
    else:
        return close.resample("1h").ffill()


@indicators.define(dependencies=(tvl,), source=IndicatorSource.dependencies_only_universe)
def tvl_inclusion_criteria(
    min_tvl: USDollarAmount,
    dependency_resolver: IndicatorDependencyResolver,
) -> pd.Series:
    """The pair must have min XX,XXX USD one-sided TVL to be included.

    - If the Uniswap pool does not have enough ETH or USDC deposited, skip the pair as a scam

    :return:
        Series where each timestamp is a list of pair ids meeting the criteria at that timestamp
    """

    series = dependency_resolver.get_indicator_data_pairs_combined(tvl)
    mask = series >= min_tvl
    # Turn to a series of lists
    mask_true_values_only = mask[mask == True]
    series = mask_true_values_only.groupby(level='timestamp').apply(lambda x: x.index.get_level_values('pair_id').tolist())
    return series


@indicators.define(
    source=IndicatorSource.strategy_universe
)
def trading_availability_criteria(
    strategy_universe: TradingStrategyUniverse,
) -> pd.Series:
    """Is pair tradeable at each hour.

    - The pair has a price candle at that
    - Mitigates very corner case issues that TVL/liquidity data is per-day whileas price data is natively per 1h
      and the strategy inclusion criteria may include pair too early hour based on TVL only,
      leading to a failed attempt to rebalance in a backtest
    - Only relevant for backtesting issues if we make an unlucky trade on the starting date
      of trading pair listing

    :return:
        Series with with index (timestamp) and values (list of pair ids trading at that hour)
    """
    # Trading pair availability is defined if there is a open candle in the index for it.
    # Because candle data is forward filled, we should not have any gaps in the index.
    candle_series = strategy_universe.data_universe.candles.df["open"]
    pairs_per_timestamp = candle_series.groupby(level='timestamp').apply(lambda x: x.index.get_level_values('pair_id').tolist())
    return pairs_per_timestamp


@indicators.define(
    dependencies=[
        tvl_inclusion_criteria,
        trading_availability_criteria
    ],
    source=IndicatorSource.strategy_universe
)
def inclusion_criteria(
    strategy_universe: TradingStrategyUniverse,
    min_tvl: USDollarAmount,
    dependency_resolver: IndicatorDependencyResolver
) -> pd.Series:
    """Pairs meeting all of our inclusion criteria.

    - Give the tradeable pair set for each timestamp

    :return:
        Series where index is timestamp and each cell is a list of pair ids matching our inclusion criteria at that moment
    """

    # Filter out benchmark pairs like WETH in the tradeable pair set
    benchmark_pair_ids = set(strategy_universe.get_pair_by_human_description(desc).internal_id for desc in SUPPORTING_PAIRS)

    tvl_series = dependency_resolver.get_indicator_data(
        tvl_inclusion_criteria,
        parameters={
            "min_tvl": min_tvl,
        },
    )

    trading_availability_series = dependency_resolver.get_indicator_data(trading_availability_criteria)

    #
    # Process all pair ids as a set and the final inclusion
    # criteria is union of all sub-criterias
    #

    df = pd.DataFrame({
        "tvl_pair_ids": tvl_series,
        "trading_availability_pair_ids": trading_availability_series,
    })

    # https://stackoverflow.com/questions/33199193/how-to-fill-dataframe-nan-values-with-empty-list-in-pandas
    df = df.fillna("").apply(list)

    def _combine_criteria(row):
        final_set = set(row["tvl_pair_ids"]) & \
                    set(row["trading_availability_pair_ids"])
        return final_set - benchmark_pair_ids

    union_criteria = df.apply(_combine_criteria, axis=1)

    # Inclusion criteria data can be spotty at the beginning when there is only 0 or 1 pairs trading,
    # so we need to fill gaps to 0
    full_index = pd.date_range(
        start=union_criteria.index.min(),
        end=union_criteria.index.max(),
        freq=Parameters.candle_time_bucket.to_frequency(),
    )
    reindexed = union_criteria.reindex(full_index, fill_value=[])
    return reindexed


@indicators.define(dependencies=(tvl_inclusion_criteria,), source=IndicatorSource.dependencies_only_universe)
def tvl_included_pair_count(
        min_tvl: USDollarAmount,
        dependency_resolver: IndicatorDependencyResolver
) -> pd.Series:
    """Calculate number of pairs in meeting volatility criteria on each timestamp"""
    series = dependency_resolver.get_indicator_data(
        tvl_inclusion_criteria,
        parameters={"min_tvl": min_tvl},
    )
    series = series.apply(len)

    # TVL data can be spotty at the beginning when there is only 0 or 1 pairs trading,
    # so we need to fill gaps to 0
    full_index = pd.date_range(
        start=series.index.min(),
        end=series.index.max(),
        freq=Parameters.candle_time_bucket.to_frequency(),
    )
    # Reindex and fill NaN with zeros
    reindexed = series.reindex(full_index, fill_value=0)
    return reindexed


@indicators.define(dependencies=(inclusion_criteria,), source=IndicatorSource.dependencies_only_universe)
def all_criteria_included_pair_count(
    min_tvl: USDollarAmount,
    dependency_resolver: IndicatorDependencyResolver
) -> pd.Series:
    """Series where each timestamp is the list of pairs meeting all inclusion criteria.

    :return:
        Series with pair count for each timestamp
    """
    series = dependency_resolver.get_indicator_data(
        "inclusion_criteria",
        parameters={
            "min_tvl": min_tvl,
        },
    )
    return series.apply(len)


@indicators.define(source=IndicatorSource.strategy_universe)
def trading_pair_count(
    strategy_universe: TradingStrategyUniverse,
) -> pd.Series:
    """Get number of pairs that trade at each timestamp.

    - Pair must have had at least one candle before the timestamp to be included

    - Exclude benchmarks pairs we do not trade

    :return:
        Series with pair count for each timestamp
    """

    benchmark_pair_ids = {strategy_universe.get_pair_by_human_description(desc).internal_id for desc in SUPPORTING_PAIRS}

    # Get pair_id, timestamp -> timestamp, pair_id index
    series = strategy_universe.data_universe.candles.df["open"]
    swap_index = series.index.swaplevel(0, 1)

    seen_pairs = set()
    seen_data = {}

    for timestamp, pair_id in swap_index:
        if pair_id in benchmark_pair_ids:
            continue
        seen_pairs.add(pair_id)
        seen_data[timestamp] = len(seen_pairs)

    series = pd.Series(seen_data.values(), index=list(seen_data.keys()))
    return series


@indicators.define(
    source=IndicatorSource.dependencies_only_per_pair,
    dependencies=[
        rolling_returns,
    ]
)
def signal(
    rolling_returns_bars: int,
    candle_time_bucket: TimeBucket,
    pair: TradingPairIdentifier,
    dependency_resolver: IndicatorDependencyResolver
) -> pd.Series:
    """Calculate weighting criteria ("signal") as the past returns of the rolling returns window."""

    rolling_returns = dependency_resolver.get_indicator_data(
            "rolling_returns",
        parameters={
            "rolling_returns_bars": rolling_returns_bars,
        },
        pair=pair,
    )
    return rolling_returns


def create_indicators(
    timestamp: datetime.datetime | None,
    parameters: StrategyParameters,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext,
):
    """Create indicators for the strategy."""
    return indicators.create_indicators(
        timestamp=timestamp,
        parameters=parameters,
        strategy_universe=strategy_universe,
        execution_context=execution_context,
    )


#
# Charts
#


def equity_curve_with_benchmark(input: ChartInput) -> list[Figure]:
    """Add our benchmark token"""
    return equity_curve(
        input,
        benchmark_token_symbols=["ETH"],
    )


def all_vaults_share_price_and_tvl(input: ChartInput) -> list[Figure]:
    """Limit max_count"""
    return _all_vaults_share_price_and_tvl(
        input,
        max_count=2,
    )


def inclusion_criteria_check_with_chain(input: ChartInput) -> pd.DataFrame:
    """Inclusion criteria check with chain id shown"""
    return inclusion_criteria_check(
        input,
        show_chain=True,
    )


def create_charts(
    timestamp: datetime.datetime | None,
    parameters: StrategyParameters,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext,
) -> ChartRegistry:
    """Define charts we use in backtesting and live trading."""
    charts = ChartRegistry(default_benchmark_pairs=BENCHMARK_PAIRS)
    charts.register(available_trading_pairs, ChartKind.indicator_all_pairs)
    charts.register(inclusion_criteria_check, ChartKind.indicator_all_pairs)
    charts.register(volatility_benchmark, ChartKind.indicator_multi_pair)
    charts.register(signal_comparison, ChartKind.indicator_multi_pair)
    charts.register(price_vs_signal, ChartKind.indicator_multi_pair)
    charts.register(all_vaults_share_price_and_tvl, ChartKind.indicator_all_pairs)
    charts.register(equity_curve_with_benchmark, ChartKind.state_all_pairs)
    charts.register(equity_curve_with_drawdown, ChartKind.state_all_pairs)
    charts.register(performance_metrics, ChartKind.state_all_pairs)
    charts.register(volatile_weights_by_percent, ChartKind.state_all_pairs)
    charts.register(volatile_and_non_volatile_percent, ChartKind.state_all_pairs)
    charts.register(equity_curve_by_asset, ChartKind.state_all_pairs)
    charts.register(weight_allocation_statistics, ChartKind.state_all_pairs)
    charts.register(positions_at_end, ChartKind.state_all_pairs)
    charts.register(last_messages, ChartKind.state_all_pairs)
    charts.register(alpha_model_diagnostics, ChartKind.state_all_pairs)
    charts.register(trading_pair_breakdown, ChartKind.state_all_pairs)
    charts.register(trading_metrics, ChartKind.state_all_pairs)
    charts.register(lending_pool_interest_accrued, ChartKind.state_all_pairs)
    charts.register(vault_statistics, ChartKind.state_all_pairs)
    charts.register(vault_position_timeline, ChartKind.state_single_vault_pair)
    charts.register(all_vault_positions, ChartKind.state_single_vault_pair)
    charts.register(trading_pair_positions, ChartKind.state_single_vault_pair)
    charts.register(trading_pair_price_and_trades, ChartKind.state_single_vault_pair)
    charts.register(inclusion_criteria_check_with_chain, ChartKind.indicator_all_pairs)
    return charts


#
# Metadata
#

tags = {StrategyTag.beta, StrategyTag.deposits_disabled}

name = "Cross-chain master vault strategy"

short_description = "Multi-chain vault allocation strategy across Ethereum, Base, Arbitrum, Avalanche, Hypercore, HyperEVM, and Monad"

icon = ""

long_description = """
# Cross-chain vault of vaults strategy

A diversified yield strategy that allocates across multiple DeFi vaults on 7 chains.

## Strategy features

- **Cross-chain allocation**: Invests across Ethereum, Base, Arbitrum, Avalanche, Hypercore, HyperEVM, and Monad
- **Multi-vault universe**: 70+ vaults selected by 1-year CAGR from top vault databases
- **Weekly rebalancing**: Adjusts positions based on rolling 7-day returns
- **Risk management**: Caps individual positions at 20% of portfolio and 20% of pool TVL
- **TVL filtering**: Only considers vaults with at least $50,000 TVL
- **Denomination flexibility**: Supports USDC, USDT, USDC.e, crvUSD, USD₮0, USDt, and USDT0

## Vault universe

The strategy selects from vaults including:
- Morpho vaults (Ethereum)
- Gains Network (Arbitrum, Base)
- Hypercore trading vaults
- HyperEVM lending and yield vaults
- Monad DeFi vaults
- And many others

## Risk parameters

- Maximum 10 positions at any time
- 95% allocation target
- Minimum $500 per trade
- 20% maximum concentration per asset
"""
