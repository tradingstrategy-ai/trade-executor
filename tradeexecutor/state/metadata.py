"""Metadata describes strategy for website rendering.

Metadata is not stored as the part of the state, but configured
on the executor start up.
"""
import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, TypedDict

from dataclasses_json import dataclass_json

from tradeexecutor.state.types import ZeroExAddress
from tradeexecutor.strategy.execution_model import AssetManagementMode


class EnzymeSmartContracts(TypedDict):
    """Various smart contract addresses associated with Enzyme.

    - Vault specific contracts

    - Enzyme chain specific contracts

    See :py:class:`eth_defi.enzyme.deployment.EnzymeContracts` for more protocol specific contracts.
    """

    #: Vault address
    vault: ZeroExAddress

    #: Comptroller proxy contract for vault
    comptroller: ZeroExAddress

    #: Generic adapter doing asset management transactions
    generic_adapter: ZeroExAddress

    gas_relay_paymaster_lib: ZeroExAddress

    gas_relay_paymaster_factory: ZeroExAddress

    integration_manager: ZeroExAddress



@dataclass_json
@dataclass
class OnChainData:
    """Smart contract information for a strategy.

    Needed for frontend deposit/redemptions/etc.
    """

    #: Is this s hot wallet strategy or vaulted strategy
    #:
    asset_management_mode: AssetManagementMode = field(default=AssetManagementMode.dummy)

    #: Smart contracts configured for this strategy.
    #:
    #: Depend on the vault backend.
    #:
    smart_contracts: EnzymeSmartContracts = field(default_factory=dict)


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
