"""GMX-specific exchange account support.

Provides the exchange account pair creation and account value function
for GMX perpetuals positions traded through a Lagoon vault.

GMX positions are on-chain (unlike Derive which uses an off-chain API),
so the account value function reads position data directly from the
GMX Reader contract via :class:`~eth_defi.gmx.core.open_positions.GetOpenPositions`.
"""

import logging
from decimal import Decimal
from typing import Callable

from eth_defi.gmx.config import GMXConfig
from eth_defi.gmx.contracts import get_contract_addresses
from eth_defi.gmx.core.open_positions import GetOpenPositions
from eth_defi.gmx.lagoon.approvals import (UNLIMITED,
                                           approve_gmx_collateral_via_vault)
from eth_defi.token import fetch_erc20_details
from hexbytes import HexBytes

from tradeexecutor.state.identifier import (AssetIdentifier,
                                            TradingPairIdentifier,
                                            TradingPairKind)

logger = logging.getLogger(__name__)


def create_gmx_exchange_account_pair(
    quote: AssetIdentifier,
    is_testnet: bool = False,
) -> TradingPairIdentifier:
    """Create a TradingPairIdentifier for a GMX exchange account.

    Builds the pair with correct ``kind``, ``exchange_name``, and ``other_data``
    fields needed by the sync, pricing, and valuation pipeline.

    The base asset is a synthetic ``GMX-ACCOUNT`` token whose address is set
    to the GMX ExchangeRouter contract. ``pool_address`` and ``exchange_address``
    also point to the ExchangeRouter — the Safe address (which holds GMX
    positions) is resolved at runtime from the execution model's transaction
    builder, not stored in the pair.

    Example::

        from tradeexecutor.exchange_account.gmx import create_gmx_exchange_account_pair

        pair = create_gmx_exchange_account_pair(quote=usdc)

    :param quote:
        Reserve / quote asset (e.g. ``USDC``).
    :param is_testnet:
        Whether this targets a testnet deployment.
    :return:
        Fully configured exchange account pair.
    """
    chain = "arbitrum_sepolia" if is_testnet else "arbitrum"
    addresses = get_contract_addresses(chain)

    base = AssetIdentifier(
        chain_id=quote.chain_id,
        address=addresses.exchangerouter,
        token_symbol="GMX-ACCOUNT",
        decimals=6,
    )

    return TradingPairIdentifier(
        base=base,
        quote=quote,
        pool_address=addresses.exchangerouter,
        exchange_address=addresses.exchangerouter,
        internal_id=1,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.exchange_account,
        exchange_name="GMX",
        other_data={
            "exchange_protocol": "gmx",
            "exchange_is_testnet": is_testnet,
        },
    )


def create_gmx_account_value_func(
    execution_model=None,
    *,
    web3=None,
    safe_address: str | None = None,
) -> Callable[[TradingPairIdentifier], Decimal]:
    """Create GMX-specific account value function.

    The returned function queries the GMX Reader contract for all open
    positions held by the Safe address, and returns the total equity
    (collateral + unrealised PnL) across all positions.

    Free USDC sitting in the Safe is tracked separately by Lagoon
    treasury sync, so this function only returns the value locked
    in GMX positions to avoid double counting.

    USDC flow and NAV implications
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    When GMX positions are opened, USDC is **transferred** from the Safe
    to the GMX OrderVault via ``sendTokens()`` in an ``ExchangeRouter.multicall()``.
    This reduces the Safe's USDC balance but happens outside the trade engine,
    so ``reserve_position.quantity`` in the portfolio becomes stale.

    The total vault NAV is: ``Safe USDC (reserves) + GMX position value (this function)``.

    To prevent double-counting (stale reserves + real position value),
    ``LagoonVaultSyncModel.sync_treasury()`` reconciles reserves from the
    actual on-chain Safe balance before calculating NAV.  See
    ``README-GMX-Lagoon.md`` for the full security analysis and token flow.

    Can be called with either an *execution_model* (used by the runner)
    or explicit *web3* + *safe_address* (used by CLI commands like
    ``correct-accounts``).

    :param execution_model:
        The execution model (e.g. ``LagoonExecution``) that provides
        Web3 and the transaction builder with the Safe address.
    :param web3:
        Explicit Web3 instance (used when no execution model is available).
    :param safe_address:
        Explicit Safe address (used when no execution model is available).
    :return:
        Function that takes a TradingPairIdentifier and returns
        the total GMX position equity in USD.
    """
    if execution_model is None:
        assert web3 is not None and safe_address is not None, \
            "Either execution_model or both web3 and safe_address must be provided"
        _web3 = web3
        _safe_address = safe_address
    else:
        _web3 = None
        _safe_address = None

    def get_gmx_account_value(pair: TradingPairIdentifier) -> Decimal:
        """Get GMX account value for the given pair.

        Reads all open positions from the GMX Reader contract for the
        Safe address, then sums the equity of each position:
        ``equity = collateral_usd * (1 + percent_profit / 100)``.

        :param pair:
            Exchange account trading pair with GMX metadata in other_data.
        :return:
            Total position equity in USD, or ``Decimal(0)`` if no
            open positions.
        """
        assert pair.is_exchange_account(), f"Not an exchange account pair: {pair}"
        assert pair.get_exchange_account_protocol() == "gmx", \
            f"Not a GMX pair: {pair.get_exchange_account_protocol()}"

        if _web3 is not None:
            web3 = _web3
            safe_address = _safe_address
        else:
            web3 = execution_model.web3
            safe_address = execution_model.tx_builder.get_token_delivery_address()

        try:
            config = GMXConfig(web3=web3)
            positions_manager = GetOpenPositions(config)
            positions = positions_manager.get_data(safe_address)

            if not positions:
                logger.debug("No open GMX positions for %s", safe_address)
                return Decimal(0)

            total_equity = Decimal(0)
            for key, pos in positions.items():
                collateral_usd = pos.get("initial_collateral_amount_usd", 0)
                percent_profit = pos.get("percent_profit", 0)
                # equity = collateral * (1 + pnl%)
                equity = Decimal(str(collateral_usd)) * (1 + Decimal(str(percent_profit)) / 100)
                total_equity += equity
                logger.debug(
                    "GMX position %s: collateral=$%.2f, pnl=%.2f%%, equity=$%.2f",
                    key, collateral_usd, percent_profit, equity,
                )

            logger.debug(
                "GMX account %s total equity: $%.2f (%d position(s))",
                safe_address, total_equity, len(positions),
            )
            return total_equity

        except Exception as e:
            logger.error(
                "Failed to get GMX account value for %s: %s",
                safe_address, e,
            )
            raise

    return get_gmx_account_value


def approve_gmx_trading(
    vault,
    hot_wallet,
    collateral_address: str | None = None,
) -> HexBytes:
    """Approve GMX collateral token to trade via the Lagoon vault.

    Convenience function intended to be called from the trade executor
    console (``trade-executor console``) to initialise GMX trading.
    Approves the collateral token (e.g. USDC) for the GMX
    SyntheticsRouter through the vault's Safe, because
    ``sendTokens()`` in the multicall does ``transferFrom`` via the
    SyntheticsRouter on every trade.

    The approval uses an unlimited amount (``2**256 - 1``).
    The Guard contract ensures that tokens can only flow to
    the pre-configured OrderVault and that order proceeds always
    return to the Safe, so an unlimited allowance does not increase
    the attack surface.

    For the full security analysis of the approval and trading flow,
    see `README-GMX-Lagoon.md <https://github.com/tradingstrategy-ai/web3-ethereum-defi/blob/master/eth_defi/gmx/README-GMX-Lagoon.md>`_
    (section *"Pre-requisite: USDC approval"*).

    Example usage from the console::

        from tradeexecutor.exchange_account.gmx import approve_gmx_trading
        approve_gmx_trading(vault, hot_wallet)

    :param vault:
        ``LagoonVault`` instance (available as ``vault`` in the console).
    :param hot_wallet:
        ``HotWallet`` of the asset manager (available as ``hot_wallet``
        in the console).
    :param collateral_address:
        ERC-20 address of the collateral token to approve.
        Defaults to the vault's underlying token (denomination asset).
    :return:
        Transaction hash of the collateral approval.
    """
    web3 = vault.web3

    # Resolve collateral token
    if collateral_address:
        collateral_token = fetch_erc20_details(web3, collateral_address)
    else:
        collateral_token = vault.underlying_token
    logger.info("Using collateral token: %s (%s)", collateral_token.symbol, collateral_token.address)

    # Approve collateral for GMX SyntheticsRouter
    logger.info("Approving collateral token for GMX SyntheticsRouter...")
    collateral_tx = approve_gmx_collateral_via_vault(
        vault=vault,
        asset_manager=hot_wallet,
        collateral_token=collateral_token,
        amount=UNLIMITED,
    )
    logger.info("Collateral approval tx: %s", web3.to_hex(collateral_tx))

    logger.info("GMX trading approval complete")
    return collateral_tx
