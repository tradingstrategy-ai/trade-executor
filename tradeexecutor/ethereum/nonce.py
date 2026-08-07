"""Hot wallet nonce maintenance for the live trading loop.

:py:class:`eth_defi.hotwallet.HotWallet` tracks the transaction nonce in an
in-memory counter that is normally read from the blockchain only once, when
the executor starts. Every transaction the executor signs afterwards increments
that counter locally.

The counter silently desynchronises whenever the same private key is used
outside the executor process, for example

- a manual Safe multisig transaction executed by the asset manager key,
- an operator running a CLI command such as ``correct-accounts`` or
  ``lagoon-redeem`` against a live strategy,
- a recovery script broadcasting from the same key.

Once the on-chain nonce has moved past the local counter, every transaction the
executor signs is rejected with ``nonce too low`` and can never be broadcast,
because the nonce is permanently spent. This module re-reads the on-chain nonce
at the start of each scheduled live task so that the desynchronisation is
corrected before anything is signed, and so that the operator gets a warning in
the logs telling them the key was used elsewhere.

See :py:func:`refresh_hot_wallet_nonces`.
"""

import logging

from eth_defi.hotwallet import HotWallet
from tradingstrategy.chain import ChainId

from tradeexecutor.ethereum.web3config import Web3Config

logger = logging.getLogger(__name__)


def refresh_hot_wallet_nonces(
    web3config: Web3Config,
    wallets: dict[ChainId, HotWallet],
    *,
    context: str = "",
) -> dict[ChainId, int]:
    """Re-read the on-chain nonce for every hot wallet we hold.

    Each wallet is refreshed against the JSON-RPC connection of its own chain,
    because a :py:class:`~eth_defi.hotwallet.HotWallet` instance carries a
    single nonce counter that is only meaningful for one chain. Passing a wallet
    under the wrong chain id would corrupt its counter, so the caller owns the
    chain to wallet mapping.

    Logging follows the on-chain movement:

    - The nonce advanced since we last looked, meaning the private key was used
      outside this process: log a ``WARNING`` and adopt the on-chain value.

    - The local counter is ahead of the chain: log at ``DEBUG``. This is the
      normal state while a broadcast transaction is still unmined. The counter
      is deliberately left alone, because
      :py:meth:`eth_defi.hotwallet.HotWallet.sync_nonce` never lowers it and
      rewinding into a pending transaction's nonce would cause a collision.

    - Nothing changed: log at ``DEBUG``.

    Example:

    .. code-block:: python

        refresh_hot_wallet_nonces(
            web3config,
            {ChainId.hyperliquid: hot_wallet},
            context="live_cycle",
        )

    .. rubric:: Where this is called

    The only production caller is
    :py:meth:`tradeexecutor.cli.loop.ExecutionLoop.refresh_nonces`, which
    resolves the persistent hot wallet from
    :py:meth:`tradeexecutor.strategy.sync_model.SyncModel.get_hot_wallet` and the
    chain from ``execution_model.tx_builder.chain_id``, then calls this function
    with a single entry mapping.

    ``refresh_nonces()`` runs at the start of all three scheduled live tasks in
    :py:meth:`tradeexecutor.cli.loop.ExecutionLoop.run_live`, each passing its
    own ``context`` so the log line names the task that triggered the refresh:

    - ``live_cycle`` — the strategy cycle, which signs trades.

    - ``live_positions`` — the statistics refresh, which also posts the on-chain
      NAV update and settles the vault for Lagoon-style strategies. This runs on
      ``stats_refresh_frequency``, independently of the strategy cycle, and is
      the task that failed in the incident below.

    - ``live_trigger_checks`` — stop loss and take profit checks, which may
      execute trades.

    All three run on the same single-threaded APScheduler executor
    (``ThreadPoolExecutor(1)``), so they are serialised and a refresh can never
    race a signing operation in another task.

    Cost is one ``eth_getTransactionCount`` per wallet per task invocation, plus
    a second one via ``sync_nonce()`` only when the nonce actually moved.

    .. rubric:: Incident this was written for

    On 2026-08-06 the ``hyper-ai`` Lagoon strategy on HyperEVM (chain 999) died
    and stayed down. Sequence of events:

    - The asset manager hot wallet ``0x005B8d2FF173C8bCc980F275884B1E717082F10C``
      is one of six owners of the vault Safe
      ``0xa8F8DEbb722c6174B814b432169BF569603F673F``, which has a signing
      threshold of four.

    - At 10:50:40 UTC a 4-of-6 Safe ``execTransaction`` calling
      ``settleDeposit(27169.177262)`` on the Lagoon vault
      ``0xC723aDd84EE4646044ff28e552808E0a3ac48b54`` was executed with that hot
      wallet as the submitting owner, in transaction ``0xad73384d…`` at block
      42,453,420. This was an operator action through the Safe UI, not the
      executor: the executor's own Lagoon transactions go through the
      TradingStrategyModule, never through ``execTransaction``. It consumed
      wallet nonce 2546.

    - The executor never noticed. ``LagoonVaultSyncModel`` inherits the empty
      :py:meth:`tradeexecutor.strategy.sync_model.SyncModel.resync_nonce`, so the
      "make sure our hot wallet nonce is up to date" call in
      :py:class:`tradeexecutor.strategy.runner.StrategyRunner` was a no-op and
      the counter had not been read from the chain since process startup.

    - At 12:00:14 UTC the ``live_positions`` task signed
      ``updateNewTotalAssets(32500.498565)`` with the stale nonce 2546 and the
      chain replied ``nonce too low: next nonce 2547, tx nonce 2546``. The nonce
      was permanently spent, so no amount of rebroadcasting could ever succeed.

    - ``eth_defi.confirmation.wait_and_broadcast_multiple_nodes_mev_blocker()``
      treated it as retryable and re-sent the identical signed bytes across three
      RPC providers for ten minutes before raising ``ConfirmationTimedOut``,
      which escaped as ``LiveSchedulingTaskFailed`` and terminated the executor.

    Refreshing at the start of each task means the counter would have been
    corrected to 2547 before the NAV update was signed.

    .. rubric:: Known gaps

    This function narrows the failure window, it does not close it. Do not treat
    it as a complete fix.

    - **The race window remains.** An external transaction can land between the
      refresh and the signing that follows it. The durable fix is to treat
      ``chain_nonce > tx.nonce`` as a non-retryable condition in
      ``eth_defi.confirmation.wait_and_broadcast_multiple_nodes_mev_blocker()``
      and have the caller re-sync, re-sign and retry once, rather than
      rebroadcasting bytes that can never be accepted.

    - **A counter that is ahead of the chain cannot be repaired here.**
      :py:meth:`eth_defi.hotwallet.HotWallet.sync_nonce` refuses to lower
      ``current_nonce``, which is what makes this function safe to call
      repeatedly, but it also means a gap is permanent for the life of the
      process. Gaps arise when a signed transaction is discarded without being
      broadcast, for example when
      :py:meth:`tradeexecutor.ethereum.vault.hypercore_routing.HypercoreVaultRouting.setup_trades`
      signs approve and deposit transactions per trade in a loop and a later
      trade raises. Repairing that needs a separate, explicit reset that is
      allowed to lower the counter, gated on having no unmined transactions of
      our own.

    - **Satellite chain wallets are not covered.** Transactions on non-primary
      chains use throwaway :py:class:`~eth_defi.hotwallet.HotWallet` objects
      created and synced per trade in
      :py:meth:`tradeexecutor.strategy.generic.generic_router.GenericRouting.setup_trades`,
      which are discarded when the transaction builder is restored. There is no
      registry of them to refresh. Related:
      ``HypercoreVaultRouting.deployer`` is rebound to whichever wallet the
      routing state last held and is never restored, so on a multichain
      deployment it can end up pointing at a satellite wallet that this function
      does not reach. The ``wallets`` argument is a mapping specifically so that
      persistent per-chain wallets can be passed here without changing the
      signature, should they be introduced.

    - **Pending transactions are invisible.** The nonce is read at the ``latest``
      block, not ``pending``, so our own broadcast but unmined transactions do
      not count. This is exactly why the counter must never be lowered.

    :param web3config:
        Web3 connections for all configured chains.

        Used to look up the JSON-RPC connection for each wallet's chain.

    :param wallets:
        Chain id to hot wallet mapping.

        Every wallet is refreshed against the connection of the chain it is
        keyed under. Chains that have no configured connection are skipped
        with a warning.

    :return:
        Chain id to the nonce the wallet will use for its next transaction,
        after the refresh.

        Chains that were skipped are absent from the result.
    """

    assert isinstance(web3config, Web3Config), f"Expected Web3Config, got {type(web3config)}"
    assert isinstance(wallets, dict), f"Expected dict, got {type(wallets)}"

    suffix = f" ({context})" if context else ""
    result: dict[ChainId, int] = {}

    for chain_id, hot_wallet in wallets.items():
        assert isinstance(chain_id, ChainId), f"Expected ChainId key, got {type(chain_id)}"
        assert isinstance(hot_wallet, HotWallet), f"Expected HotWallet value, got {type(hot_wallet)}"

        try:
            web3 = web3config.get_connection(chain_id)
        except KeyError:
            logger.warning(
                "Cannot refresh hot wallet %s nonce%s: no JSON-RPC connection configured for chain %s",
                hot_wallet.address,
                suffix,
                chain_id.name,
            )
            continue

        local_nonce = hot_wallet.current_nonce
        onchain_nonce = web3.eth.get_transaction_count(hot_wallet.address)

        if local_nonce is None:
            # First sync for this wallet, nothing to compare against
            hot_wallet.sync_nonce(web3)
            logger.debug(
                "Hot wallet %s nonce initialised to %d on chain %s%s",
                hot_wallet.address,
                hot_wallet.current_nonce,
                chain_id.name,
                suffix,
            )
        elif onchain_nonce > local_nonce:
            # The key was used outside the trade executor and burned nonces we
            # do not know about. Adopt the chain value before we sign anything,
            # otherwise every transaction we sign is rejected as "nonce too low".
            hot_wallet.sync_nonce(web3)
            logger.warning(
                "Hot wallet %s nonce advanced outside the trade executor on chain %s%s: was %d, now %d. "
                "The private key has been used by another process, a manual Safe transaction or a CLI command.",
                hot_wallet.address,
                chain_id.name,
                suffix,
                local_nonce,
                hot_wallet.current_nonce,
            )
        elif onchain_nonce < local_nonce:
            # Normal while our own transaction is broadcast but not yet mined.
            # Also happens when a signed transaction was discarded, which leaves
            # a permanent gap that this function cannot repair.
            logger.debug(
                "Hot wallet %s local nonce %d is ahead of chain nonce %d on chain %s%s, "
                "leaving the counter untouched",
                hot_wallet.address,
                local_nonce,
                onchain_nonce,
                chain_id.name,
                suffix,
            )
        else:
            logger.debug(
                "Hot wallet %s nonce unchanged at %d on chain %s%s",
                hot_wallet.address,
                local_nonce,
                chain_id.name,
                suffix,
            )

        result[chain_id] = hot_wallet.current_nonce

    return result
