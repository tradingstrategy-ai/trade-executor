"""Metadata describes strategy for website rendering.

Metadata is not stored as the part of the state, but configured
on the executor start up.
"""
import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, TypedDict, List, Set

from dataclasses_json import dataclass_json

from tradeexecutor.strategy.tag import StrategyTag
from tradingstrategy.chain import ChainId

from tradeexecutor.state.state import State
from tradeexecutor.state.types import ZeroExAddress
from tradeexecutor.strategy.execution_model import AssetManagementMode


class EnzymeSmartContracts(TypedDict):
    """Various smart contract addresses associated with Enzyme.

    - Vault specific contracts

    - Enzyme chain specific contracts

    See :py:class:`eth_defi.enzyme.deployment.EnzymeContracts` for the full list of protocol specific contracts.
    """

    #: Vault address
    vault: ZeroExAddress

    #: Comptroller proxy contract for vault
    comptroller: ZeroExAddress

    #: Generic adapter doing asset management transactions
    generic_adapter: ZeroExAddress

    #: Enzyme contract
    gas_relay_paymaster_lib: ZeroExAddress

    #: Enzyme contract
    gas_relay_paymaster_factory: ZeroExAddress

    #: Enzyme contract
    integration_manager: ZeroExAddress

    #: Calculate values for various positions
    #:
    #: See https://github.com/enzymefinance/protocol/blob/v4/contracts/persistent/off-chain/fund-value-calculator/IFundValueCalculator.sol
    fund_value_calculator: ZeroExAddress

    #: VaultUSDCPaymentForwarder.sol
    #:
    payment_forwarder: ZeroExAddress

    #: GuardV0
    #:
    guard: ZeroExAddress

    #: TermsOfService
    #:
    terms_of_service: ZeroExAddress


@dataclass_json
@dataclass
class OnChainData:
    """Smart contract information for a strategy.

    Needed for frontend deposit/redemptions/etc.
    """

    #: On which this strategy runs on
    chain_id: ChainId = field(default=ChainId.unknown)

    #: Is this s hot wallet strategy or vaulted strategy
    #:
    asset_management_mode: AssetManagementMode = field(default=AssetManagementMode.dummy)

    #: Smart contracts configured for this strategy.
    #:
    #: Depend on the vault backend.
    #:
    smart_contracts: EnzymeSmartContracts = field(default_factory=dict)

    #: Vault owner address
    #:
    #: Multisig or ProtoDAO.
    #:
    owner: ZeroExAddress = None

    #: Asset manager address.
    #:
    trade_executor_hot_wallet: ZeroExAddress = None


@dataclass_json
@dataclass
class Metadata:
    """Strategy metadata."""

    #: Strategy name
    name: str

    #: 1 sentence
    short_description: Optional[str]

    #: Multiple paragraphs.
    long_description: Optional[str]

    #: For <img src>
    icon_url: Optional[str]

    #: When the instance was started last time, UTC
    started_at: datetime.datetime

    #: Is the executor main loop running or crashed.
    #:
    #: Use /status endpoint to get the full exception info.
    #:
    #: Not really a part of metadata, but added here to make frontend
    #: queries faster. See also :py:class:`tradeexecutor.state.executor_state.ExecutorState`.
    executor_running: bool

    #: List of smart contracts and related web3 interaction information for this strategy.
    #:
    on_chain_data: OnChainData = field(default_factory=OnChainData)

    #: The previous backtest run results for this strategy.
    #:
    #: Used in the web frontend to display the backtested values.
    #:
    backtested_state: Optional[State] = None

    #: Backtest notebook .ipynb file
    #:
    #:
    backtest_notebook: Optional[Path] = None

    #: Backtest notebook .html file
    #:
    #:
    backtest_html: Optional[Path] = None

    #: How many days live data is collected until key metrics are switched from backtest to live trading based
    #:
    #: Two years: by default we do not show live trading metrics until the strategy has been running for long.
    #:
    key_metrics_backtest_cut_off: datetime.timedelta = datetime.timedelta(days=365*2)

    #: List of badges strategy tile can display.
    #:
    #: Used for the user to visualise context information about the strategy.
    #:
    #: E.g. "metamask", "polygon", "eth", "usdc"
    #:
    #: - For the available badges see the frontend repo.
    #: - Chain badge e.g. Polygon does not need to be declared as it is part of the strategh
    #: - Vault type. e.g. Enzyme badge is the same
    #: - Can contain badges like USDC
    #: - Some of badges are automatically derived, some are manually set.
    #:
    #: See also :py:attr:`tags`
    #:
    badges: List[str] = field(default_factory=list)

    #: Tags on this strategy
    #:
    #: See also :py:attr:`badges`
    #:
    tags: Set[StrategyTag] = field(default_factory=set)

    #: The display priority for this strategy.
    #:
    #: Higher = the strategy apppears in the frontend first.
    sort_priority: int = 0

    #: Fees for this strategy
    #: In the format
    #: {
    #:     management_fee,
    #:     trading_strategy_protocol_fee,
    #:     strategy_developer_fee,
    #:     enzyme_protocol_fee,
    #: }
    fees: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def create_dummy() -> "Metadata":
        return Metadata(
            name="Dummy",
            short_description="Dummy metadata",
            long_description=None,
            icon_url=None,
            started_at=datetime.datetime.utcnow(),
            executor_running=True,
        )

    def has_backtest_data(self) -> bool:
        """Does this strategy have backtest data available on the file system?"""
        return (self.backtest_notebook and self.backtest_notebook.exists()) and (self.backtest_html and self.backtest_html.exists())

    @staticmethod
    def parse_badges_configuration(config_line: str | None) -> List[str]:
        """Parse BADGES environment variable.

        Comma separated list, support whitespaces.
        """
        if not config_line:
            return []

        return [s.strip() for s in config_line.split(",")]
