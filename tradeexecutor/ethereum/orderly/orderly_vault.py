"""Orderly vault integration with VaultBase pattern."""

from decimal import Decimal
from functools import cached_property

from eth_typing import HexAddress, BlockIdentifier
from web3 import Web3
from web3.contract import Contract

from eth_defi.abi import get_deployed_contract
from eth_defi.token import TokenDetails, fetch_erc20_details
from eth_defi.vault.base import (
    VaultBase,
    VaultSpec,
    VaultInfo,
    VaultFlowManager,
    VaultDepositManager,
    VaultHistoricalReader,
    TradingUniverse,
    VaultPortfolio,
    BlockRange,
)


class OrderlyFlowManager(VaultFlowManager):
    """Manage Orderly vault deposit/withdrawal flow events.

    Tracks pending deposits and withdrawals through vault contract events.
    """

    def __init__(self, vault: "OrderlyVault"):
        self.vault = vault

    def fetch_pending_redemption(self, block_identifier: BlockIdentifier) -> Decimal:
        """Get how much users want to redeem from the vault.

        :param block_identifier:
            Block number

        :return:
            Number of tokens users want to withdraw
        """
        # TODO: Implement by querying pending withdraw requests
        # This would read from vault contract storage or events
        return Decimal(0)

    def fetch_pending_deposit(self, block_identifier: BlockIdentifier) -> Decimal:
        """Get how much users want to deposit to the vault.

        :param block_identifier:
            Block number

        :return:
            Number of underlying tokens in pending deposits
        """
        # TODO: Implement by querying pending deposit requests
        # This would read from vault contract or Orderly API
        return Decimal(0)

    def fetch_pending_deposit_events(self, range: BlockRange) -> None:
        """Read incoming pending deposits.

        TODO: Implement by scanning for deposit events in block range
        """
        raise NotImplementedError("Orderly pending deposit event scanning not yet implemented")

    def fetch_pending_redemption_event(self, range: BlockRange) -> None:
        """Read outgoing pending withdraws.

        TODO: Implement by scanning for withdrawal request events
        """
        raise NotImplementedError("Orderly pending redemption event scanning not yet implemented")

    def fetch_processed_deposit_event(self, range: BlockRange) -> None:
        """Read processed deposits.

        TODO: Implement by scanning for processed deposit events
        """
        raise NotImplementedError("Orderly processed deposit event scanning not yet implemented")

    def fetch_processed_redemption_event(self, vault: VaultSpec, range: BlockRange) -> None:
        """Read processed withdraws.

        TODO: Implement by scanning for processed withdrawal events
        """
        raise NotImplementedError("Orderly processed redemption event scanning not yet implemented")


class OrderlyVaultInfo(VaultInfo):
    """Information about Orderly vault deployment.

    Stores Orderly-specific configuration like broker_id and supported tokens.
    """

    #: Vault contract address
    address: HexAddress

    #: Underlying token address (e.g., USDC)
    asset: HexAddress

    #: Orderly broker ID for this vault
    broker_id: str

    #: Supported token mappings: symbol -> Orderly token_id
    #: Example: {"USDC": "USDC", "WETH": "WETH"}
    supported_tokens: dict[str, str]


class OrderlyVault(VaultBase):
    """Orderly vault integration following VaultBase pattern.

    Orderly is a CEX-like orderbook exchange with an on-chain vault for deposits/withdrawals.
    This vault handles the bridge between on-chain assets and off-chain Orderly trading accounts.

    Key characteristics:
    - Deposits go to vault then credited to Orderly account (off-chain)
    - Trading happens on Orderly exchange (off-chain orderbook)
    - Withdrawals are requested from vault back to user wallet

    See: https://orderly.network/docs/build-on-omnichain/user-flows/withdrawal-deposit
    """

    def __init__(
        self,
        web3: Web3,
        spec: VaultSpec,
        broker_id: str,
        supported_tokens: dict[str, str] | None = None,
        token_cache: dict | None = None,
    ):
        """Initialize Orderly vault.

        :param web3:
            Web3 connection

        :param spec:
            Vault specification with chain_id and vault_address

        :param broker_id:
            Orderly broker ID for transactions

        :param supported_tokens:
            Token symbol to Orderly token_id mapping.
            Example: {"USDC": "USDC", "WETH": "WETH"}
            Defaults to empty dict.

        :param token_cache:
            Optional token details cache
        """
        super().__init__(token_cache=token_cache)
        self.spec = spec
        self.web3 = web3
        self.broker_id = broker_id
        self.supported_tokens = supported_tokens or {}
        self.vault_contract = get_deployed_contract(web3, "orderly/Vault.json", spec.vault_address)

    def __repr__(self):
        return f"<OrderlyVault address:{self.address} broker:{self.broker_id}>"

    @property
    def chain_id(self) -> int:
        """Chain this vault is on"""
        return self.spec.chain_id

    @property
    def address(self) -> HexAddress:
        """Vault contract address"""
        return self.spec.vault_address

    @property
    def contract(self) -> Contract:
        """Vault contract instance for backwards compatibility"""
        return self.vault_contract

    @cached_property
    def name(self) -> str:
        """Vault name - read from contract if available"""
        try:
            return self.vault_contract.functions.name().call()
        except Exception:
            return f"Orderly Vault {self.address[:8]}"

    @cached_property
    def symbol(self) -> str:
        """Vault share token symbol - read from contract if available"""
        try:
            return self.vault_contract.functions.symbol().call()
        except Exception:
            return "ORDERLY"

    def has_block_range_event_support(self) -> bool:
        """Orderly vault supports event-based queries"""
        return True

    def has_deposit_distribution_to_all_positions(self) -> bool:
        """Orderly deposits go to off-chain account, not distributed to positions"""
        return False

    def fetch_portfolio(
        self,
        universe: TradingUniverse,
        block_identifier: BlockIdentifier | None = None,
    ) -> VaultPortfolio:
        """Read current token balances in Orderly vault.

        Note: This only reads on-chain vault balances.
        Off-chain Orderly account balances require API integration.
        """
        raise NotImplementedError("Orderly vault portfolio fetching not yet implemented")

    def fetch_info(self) -> OrderlyVaultInfo:
        """Read vault parameters from the chain"""
        # Fetch denomination token (assumed to be first supported token or USDC)
        asset_address = self.fetch_denomination_token_address()

        return OrderlyVaultInfo(
            address=self.address,
            asset=asset_address,
            broker_id=self.broker_id,
            supported_tokens=self.supported_tokens,
        )

    def get_flow_manager(self) -> OrderlyFlowManager:
        """Get flow manager for reading deposit/redemption events"""
        return OrderlyFlowManager(self)

    def get_deposit_manager(self) -> VaultDepositManager:
        """Get deposit manager for vault operations"""
        raise NotImplementedError("Orderly deposit manager not yet implemented")

    def get_historical_reader(self, stateful: bool) -> VaultHistoricalReader:
        """Get share price reader for historical returns"""
        raise NotImplementedError("Orderly historical reader not yet implemented")
    
    def fetch_share_token(self) -> TokenDetails:
        raise NotImplementedError("Orderly fetch share token not yet implemented")

    def fetch_denomination_token_address(self) -> HexAddress:
        """Get the denomination token address.

        For Orderly, this is typically USDC.
        """
        # Try to read from contract if available
        try:
            return self.vault_contract.functions.getAsset().call()
        except Exception:
            # Fallback: return zero address, caller should handle
            return "0x0000000000000000000000000000000000000000"

    def fetch_denomination_token(self) -> TokenDetails:
        """Read denomination token details from chain"""
        asset_address = self.fetch_denomination_token_address()
        return fetch_erc20_details(
            self.web3,
            asset_address,
            chain_id=self.chain_id,
            token_cache=self.token_cache,
        )

    def fetch_nav(self, block_identifier: BlockIdentifier | None = None) -> Decimal:
        """Fetch current NAV from vault.

        For Orderly, NAV would need to be calculated from:
        - On-chain vault balance
        - Off-chain Orderly account balance (requires API)
        - Pending deposits/withdrawals
        """
        raise NotImplementedError("Orderly NAV fetching not yet implemented")
