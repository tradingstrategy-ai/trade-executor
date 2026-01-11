"""Freqtrade routing model for deposit/withdrawal management."""

import logging
import time
from dataclasses import dataclass
from decimal import Decimal

from hexbytes import HexBytes
from web3 import Web3

from eth_defi.abi import get_deployed_contract
from eth_defi.token import fetch_erc20_details

from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.routing import RoutingModel, RoutingState
from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse
from tradeexecutor.strategy.freqtrade.config import (
    FreqtradeConfig,
    FreqtradeDepositConfig,
    FreqtradeDepositMethod,
    AsterDepositConfig,
    HyperliquidDepositConfig,
    OrderlyDepositConfig,
    HYPERLIQUID_BRIDGE_MAINNET,
    HYPERLIQUID_BRIDGE_TESTNET,
    USDC_ARBITRUM_MAINNET,
    USDC_ARBITRUM_TESTNET,
)
from tradeexecutor.strategy.freqtrade.freqtrade_client import FreqtradeClient

logger = logging.getLogger(__name__)


@dataclass
class FreqtradeRoutingState(RoutingState):
    """Routing state for Freqtrade deposits.

    Holds tx_builder and FreqtradeClient instances for the execution cycle.
    """

    tx_builder: TransactionBuilder | None
    web3: Web3 | None
    freqtrade_clients: dict[str, FreqtradeClient]


class FreqtradeRoutingModel(RoutingModel):
    """Route capital deposits/withdrawals to Freqtrade instances.

    Supports multiple deposit methods:
    - Aster vault: ERC20 approve + AstherusVault.deposit() on BSC
    - Hyperliquid: Bridge transfer on Arbitrum + SDK vault deposit
    - Orderly vault: ERC20 approve + Vault.deposit() with hashed params
    """

    def __init__(self, freqtrade_configs: dict[str, FreqtradeConfig]):
        """Initialise routing model.

        Args:
            freqtrade_configs: Dict mapping freqtrade_id -> FreqtradeConfig
        """
        # Freqtrade doesn't use DEX routing - pass empty intermediary pairs
        super().__init__(
            allowed_intermediary_pairs={},
            reserve_token_address="0x0000000000000000000000000000000000000000",
        )
        self.freqtrade_configs = freqtrade_configs

    def create_routing_state(
        self,
        universe: StrategyExecutionUniverse,
        execution_details: object,
    ) -> FreqtradeRoutingState:
        """Create routing state for this cycle.

        Args:
            universe: Strategy execution universe
            execution_details: Dict containing tx_builder from ExecutionModel

        Returns:
            FreqtradeRoutingState with tx_builder and FreqtradeClients
        """
        tx_builder = None
        web3 = None

        if execution_details:
            tx_builder = execution_details.get("tx_builder")
            if tx_builder:
                web3 = tx_builder.web3

        # Create FreqtradeClient for each config
        clients = {}
        for freqtrade_id, config in self.freqtrade_configs.items():
            clients[freqtrade_id] = FreqtradeClient(
                config.api_url,
                config.api_username,
                config.api_password,
            )

        return FreqtradeRoutingState(
            tx_builder=tx_builder,
            web3=web3,
            freqtrade_clients=clients,
        )

    def setup_trades(
        self,
        state: State,
        routing_state: FreqtradeRoutingState,
        trades: list[TradeExecution],
        check_balances: bool = False,
        rebroadcast: bool = False,
        **kwargs,
    ):
        """Prepare deposit transactions.

        Dispatches to method-specific builders based on deposit config.

        Args:
            state: Current portfolio state
            routing_state: Routing state with tx_builder and clients
            trades: Trades to prepare
            check_balances: Whether to check balances (not used)
            rebroadcast: Whether this is a rebroadcast (not used)
            **kwargs: Additional arguments
        """
        for trade in trades:
            freqtrade_id = trade.pair.other_data["freqtrade_id"]
            config = self.freqtrade_configs[freqtrade_id]
            deposit_config = config.deposit
            assert deposit_config
            # amount = trade.planned_reserve if trade.planned_reserve else trade.planned_quantity

            if deposit_config.method == FreqtradeDepositMethod.aster_vault:
                assert isinstance(deposit_config, AsterDepositConfig)
                txs = self._build_aster_deposit_tx(trade, config, deposit_config, routing_state)
                trade.blockchain_transactions = txs

            elif deposit_config.method == FreqtradeDepositMethod.hyperliquid:
                assert isinstance(deposit_config, HyperliquidDepositConfig)
                txs = self._build_hyperliquid_deposit_tx(trade, config, deposit_config, routing_state)
                trade.blockchain_transactions = txs

            elif deposit_config.method == FreqtradeDepositMethod.orderly_vault:
                assert isinstance(deposit_config, OrderlyDepositConfig)
                txs = self._build_orderly_deposit_tx(trade, config, deposit_config, routing_state)
                trade.blockchain_transactions = txs

            else:
                raise NotImplementedError(
                    f"Deposit method {deposit_config.method} not yet implemented"
                )

    def _build_aster_deposit_tx(
        self,
        trade: TradeExecution,
        config: FreqtradeConfig,
        deposit_config: AsterDepositConfig,
        routing_state: FreqtradeRoutingState,
    ) -> list[BlockchainTransaction]:
        """Build Aster vault deposit transactions.

        Flow:
        1. ERC20.approve(vault_address, amount)
        2. AstherusVault.deposit(token_address, amount, broker_id)

        Args:
            trade: Trade to build transaction for
            config: Freqtrade configuration
            deposit_config: Aster deposit configuration
            routing_state: Routing state with tx_builder

        Returns:
            List of BlockchainTransactions (approve + deposit)
        """
        if routing_state.tx_builder is None:
            raise ValueError("tx_builder required for aster_vault deposits")

        if routing_state.web3 is None:
            raise ValueError("web3 required for aster_vault deposits")

        if deposit_config.vault_address is None:
            raise ValueError(f"deposit.vault_address required for {config.freqtrade_id}")

        web3 = routing_state.web3

        # Get Freqtrade balance before deposit
        client = routing_state.freqtrade_clients[config.freqtrade_id]
        balance_before = Decimal(str(client.get_balance().get("total", 0)))

        # Store balance_before in trade for later verification
        if trade.other_data is None:
            trade.other_data = {}
        trade.other_data["balance_before_deposit"] = str(balance_before)

        # Get token details
        amount = trade.planned_reserve if trade.planned_reserve else trade.planned_quantity
        token = fetch_erc20_details(web3, config.reserve_currency)
        amount_raw = token.convert_to_raw(amount)

        vault_address = Web3.to_checksum_address(deposit_config.vault_address)

        # 1. Build approve transaction
        approve_call = token.contract.functions.approve(vault_address, amount_raw)
        approve_tx = routing_state.tx_builder.sign_transaction(
            token.contract,
            approve_call,
            gas_limit=100_000,
            notes=f"Approve Aster vault for {config.freqtrade_id}",
        )

        # 2. Build vault.deposit transaction
        vault = get_deployed_contract(
            web3,
            "aster/AstherusVault.json",
            vault_address,
        )
        deposit_call = vault.functions.deposit(
            Web3.to_checksum_address(config.reserve_currency),
            amount_raw,
            deposit_config.broker_id,
        )
        deposit_tx = routing_state.tx_builder.sign_transaction(
            vault,
            deposit_call,
            gas_limit=200_000,
            notes=f"Aster vault deposit for {config.freqtrade_id}",
        )

        trade.notes = (
            f"Aster vault deposit: {amount} to {vault_address}"
        )
        logger.info(f"Trade {trade.trade_id}: {trade.notes}")

        return [approve_tx, deposit_tx]

    def _build_hyperliquid_deposit_tx(
        self,
        trade: TradeExecution,
        config: FreqtradeConfig,
        deposit_config: HyperliquidDepositConfig,
        routing_state: FreqtradeRoutingState,
    ) -> list[BlockchainTransaction]:
        """Build Hyperliquid bridge transfer transaction.

        On-chain part only: Transfer USDC to bridge on Arbitrum.
        The off-chain SDK vault deposit happens in settle_trade().

        Args:
            trade: Trade to build transaction for
            config: Freqtrade configuration
            deposit_config: Hyperliquid deposit configuration
            routing_state: Routing state with tx_builder

        Returns:
            List containing bridge transfer transaction
        """
        if routing_state.tx_builder is None:
            raise ValueError("tx_builder required for hyperliquid deposits")

        if routing_state.web3 is None:
            raise ValueError("web3 required for hyperliquid deposits")

        if deposit_config.vault_address is None:
            raise ValueError(f"deposit.vault_address required for {config.freqtrade_id}")

        web3 = routing_state.web3

        # Get Freqtrade balance before deposit
        client = routing_state.freqtrade_clients[config.freqtrade_id]
        balance_before = Decimal(str(client.get_balance().get("total", 0)))

        # Store balance_before and vault_address in trade for settle_trade
        if trade.other_data is None:
            trade.other_data = {}
        trade.other_data["balance_before_deposit"] = str(balance_before)
        trade.other_data["hyperliquid_vault_address"] = deposit_config.vault_address
        trade.other_data["hyperliquid_is_mainnet"] = deposit_config.is_mainnet

        # Get bridge and USDC addresses based on network
        if deposit_config.is_mainnet:
            bridge_address = HYPERLIQUID_BRIDGE_MAINNET
            usdc_address = USDC_ARBITRUM_MAINNET
        else:
            bridge_address = HYPERLIQUID_BRIDGE_TESTNET
            usdc_address = USDC_ARBITRUM_TESTNET

        # Build transfer to bridge (USDC on Arbitrum)
        amount = trade.planned_reserve if trade.planned_reserve else trade.planned_quantity
        token = fetch_erc20_details(web3, usdc_address)
        amount_raw = token.convert_to_raw(amount)

        transfer_call = token.contract.functions.transfer(
            Web3.to_checksum_address(bridge_address),
            amount_raw,
        )
        transfer_tx = routing_state.tx_builder.sign_transaction(
            token.contract,
            transfer_call,
            gas_limit=100_000,
            notes=f"Hyperliquid bridge transfer for {config.freqtrade_id}",
        )

        trade.notes = (
            f"Hyperliquid bridge transfer: {amount} USDC to {bridge_address}"
        )
        logger.info(f"Trade {trade.trade_id}: {trade.notes}")

        return [transfer_tx]

    def _build_orderly_deposit_tx(
        self,
        trade: TradeExecution,
        config: FreqtradeConfig,
        deposit_config: OrderlyDepositConfig,
        routing_state: FreqtradeRoutingState,
    ) -> list[BlockchainTransaction]:
        """Build Orderly vault deposit transactions.

        Flow:
        1. ERC20.approve(vault_address, amount)
        2. Vault.deposit((account_id, broker_hash, token_hash, amount))

        Args:
            trade: Trade to build transaction for
            config: Freqtrade configuration
            deposit_config: Orderly deposit configuration
            routing_state: Routing state with tx_builder

        Returns:
            List of BlockchainTransactions (approve + deposit)
        """
        if routing_state.tx_builder is None:
            raise ValueError("tx_builder required for orderly_vault deposits")

        if routing_state.web3 is None:
            raise ValueError("web3 required for orderly_vault deposits")

        if deposit_config.vault_address is None:
            raise ValueError(f"deposit.vault_address required for {config.freqtrade_id}")

        if deposit_config.orderly_account_id is None:
            raise ValueError(f"deposit.orderly_account_id required for {config.freqtrade_id}")

        if deposit_config.broker_id is None:
            raise ValueError(f"deposit.broker_id required for {config.freqtrade_id}")

        web3 = routing_state.web3

        # Get Freqtrade balance before deposit
        client = routing_state.freqtrade_clients[config.freqtrade_id]
        balance_before = Decimal(str(client.get_balance().get("total", 0)))

        # Store balance_before in trade for later verification
        if trade.other_data is None:
            trade.other_data = {}
        trade.other_data["balance_before_deposit"] = str(balance_before)

        # Get token details
        amount = trade.planned_reserve if trade.planned_reserve else trade.planned_quantity
        token = fetch_erc20_details(web3, config.reserve_currency)
        amount_raw = token.convert_to_raw(amount)

        vault_address = Web3.to_checksum_address(deposit_config.vault_address)

        # 1. Build approve transaction
        approve_call = token.contract.functions.approve(vault_address, amount_raw)
        approve_tx = routing_state.tx_builder.sign_transaction(
            token.contract,
            approve_call,
            gas_limit=100_000,
            notes=f"Approve Orderly vault for {config.freqtrade_id}",
        )

        # 2. Build vault.deposit transaction with hashed parameters
        broker_hash = web3.keccak(text=deposit_config.broker_id)
        token_hash = web3.keccak(text=deposit_config.token_id) if deposit_config.token_id else bytes(32)

        # Build deposit input tuple
        deposit_input = (
            bytes.fromhex(deposit_config.orderly_account_id[2:]),  # Remove 0x prefix
            broker_hash,
            token_hash,
            amount_raw,
        )

        vault = get_deployed_contract(
            web3,
            "orderly/Vault.json",
            vault_address,
        )
        deposit_call = vault.functions.deposit(deposit_input)
        deposit_tx = routing_state.tx_builder.sign_transaction(
            vault,
            deposit_call,
            gas_limit=200_000,
            notes=f"Orderly vault deposit for {config.freqtrade_id}",
        )

        trade.notes = (
            f"Orderly vault deposit: {amount} to {vault_address}"
        )
        logger.info(f"Trade {trade.trade_id}: {trade.notes}")

        return [approve_tx, deposit_tx]

    def settle_trade(
        self,
        web3: Web3,
        state: State,
        trade: TradeExecution,
        receipts: dict[HexBytes, dict],
        stop_on_execution_failure: bool = False,
        **kwargs,
    ):
        """Settle a trade after transaction broadcast.

        For vault methods: polls Freqtrade balance until deposit confirmed.
        For Hyperliquid: also performs SDK vault deposit after bridge transfer.

        Args:
            web3: Web3 instance
            state: Current portfolio state
            trade: Trade to settle
            receipts: Transaction receipts
            stop_on_execution_failure: Whether to stop on failure
            **kwargs: Additional arguments
        """
        freqtrade_id = trade.pair.other_data["freqtrade_id"]
        config = self.freqtrade_configs[freqtrade_id]
        deposit_config = config.deposit
        assert deposit_config, f"deposit config required for {freqtrade_id}"

        if deposit_config.method == FreqtradeDepositMethod.aster_vault:
            self._confirm_deposit(trade, config, deposit_config)

        elif deposit_config.method == FreqtradeDepositMethod.hyperliquid:
            assert isinstance(deposit_config, HyperliquidDepositConfig)
            self._confirm_hyperliquid_deposit(trade, config, deposit_config, **kwargs)

        elif deposit_config.method == FreqtradeDepositMethod.orderly_vault:
            self._confirm_deposit(trade, config, deposit_config)

        else:
            raise NotImplementedError(
                f"Deposit method {deposit_config.method} not yet implemented"
            )

    def _confirm_deposit(
        self,
        trade: TradeExecution,
        config: FreqtradeConfig,
        deposit_config: FreqtradeDepositConfig,
    ):
        """Poll Freqtrade balance until deposit is confirmed.

        Works for Aster and Orderly vault deposits.

        Args:
            trade: Trade containing balance_before_deposit
            config: Freqtrade configuration
            deposit_config: Deposit configuration with timeout settings

        Raises:
            Exception: If deposit not confirmed within timeout
        """
        client = FreqtradeClient(
            config.api_url,
            config.api_username,
            config.api_password,
        )

        balance_before = Decimal(trade.other_data.get("balance_before_deposit", "0"))
        amount = trade.planned_reserve if trade.planned_reserve else trade.planned_quantity
        expected_min = balance_before + amount - deposit_config.fee_tolerance

        deadline = time.time() + deposit_config.confirmation_timeout

        logger.info(
            f"Trade {trade.trade_id}: Waiting for deposit confirmation. "
            f"Expected balance >= {expected_min}"
        )

        balance_after = None
        while time.time() < deadline:
            try:
                balance_after = Decimal(str(client.get_balance().get("total", 0)))

                if balance_after >= expected_min:
                    trade.notes = f"Deposit confirmed: {balance_before} -> {balance_after}"
                    logger.info(f"Trade {trade.trade_id}: {trade.notes}")
                    return

                logger.debug(
                    f"Trade {trade.trade_id}: Balance {balance_after}, "
                    f"waiting for {expected_min}"
                )

            except Exception as e:
                logger.warning(f"Trade {trade.trade_id}: Balance check failed: {e}")

            time.sleep(deposit_config.poll_interval)

        raise Exception(
            f"Deposit not confirmed within {deposit_config.confirmation_timeout}s. "
            f"Expected balance >= {expected_min}, got {balance_after}"
        )

    def _confirm_hyperliquid_deposit(
        self,
        trade: TradeExecution,
        config: FreqtradeConfig,
        deposit_config: HyperliquidDepositConfig,
        **kwargs,
    ):
        """Perform Hyperliquid SDK vault deposit and confirm.

        After bridge transfer, this:
        1. Calls SDK vault_usd_transfer() to deposit into vault
        2. Polls Freqtrade balance until confirmed

        Args:
            trade: Trade containing vault info
            config: Freqtrade configuration
            deposit_config: Hyperliquid deposit configuration
            **kwargs: May contain 'private_key' for SDK signing

        Raises:
            Exception: If deposit not confirmed within timeout
        """
        try:
            from eth_account import Account
            from hyperliquid.exchange import Exchange
            from hyperliquid.utils.constants import MAINNET_API_URL, TESTNET_API_URL
        except ImportError:
            raise ImportError(
                "hyperliquid-py package required for Hyperliquid deposits. "
                "Install with: pip install hyperliquid-py"
            )

        vault_address = trade.other_data.get("hyperliquid_vault_address")
        is_mainnet = trade.other_data.get("hyperliquid_is_mainnet", True)

        # Get private key for SDK signing
        private_key = kwargs.get("private_key")
        if private_key is None:
            raise ValueError("private_key required for Hyperliquid SDK vault deposit")

        # Perform SDK vault deposit
        wallet = Account.from_key(private_key)
        base_url = MAINNET_API_URL if is_mainnet else TESTNET_API_URL
        exchange = Exchange(wallet=wallet, base_url=base_url)

        amount = trade.planned_reserve if trade.planned_reserve else trade.planned_quantity
        amount_int = int(amount * Decimal("1000000"))  # Convert to micro-USD

        logger.info(
            f"Trade {trade.trade_id}: Performing Hyperliquid SDK vault deposit. "
            f"Vault: {vault_address}, Amount: {amount} USD"
        )

        result = exchange.vault_usd_transfer(
            vault_address=vault_address,
            is_deposit=True,
            usd=amount_int,
        )

        if result.get("status") != "ok":
            raise Exception(f"Hyperliquid vault deposit failed: {result}")

        logger.info(f"Trade {trade.trade_id}: SDK vault deposit successful")

        # Now confirm via Freqtrade balance polling
        self._confirm_deposit(trade, config, deposit_config)
