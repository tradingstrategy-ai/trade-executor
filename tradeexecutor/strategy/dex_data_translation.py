"""Helper functions to translate Trading Strategy client data to trade executor data format."""
from math import isnan

from tradeexecutor.state.identifier import AssetIdentifier, AssetType, TradingPairIdentifier, TradingPairKind
from tradingstrategy.exchange import ExchangeType
from tradingstrategy.lending import LendingReserve
from tradingstrategy.pair import DEXPair
from tradingstrategy.token import Token
from tradingstrategy.token_metadata import TokenMetadata
from tradingstrategy.vault import VaultMetadata


def translate_token(
    token: Token,
    require_decimals=True,
    underlying: AssetIdentifier | None = None,
    type: AssetType | None = AssetType.token,
    liquidation_threshold: float | None = None,
) -> AssetIdentifier:
    """Translate Trading Strategy token data definition to trade executor.

    Trading Strategy client uses compressed columnar data for pairs and tokens.

    Creates `AssetIdentifier` based on data coming from
    Trading Strategy :py:class:`tradingstrategy.pair.PandasPairUniverse`.

    :param underlying:
        Underlying asset for dynamic lending tokens.

    :param require_decimals:
        Most tokens / trading pairs are non-functional without decimals information.
        Assume decimals is in place. If not then raise AssertionError.
        This check allows some early error catching on bad data.

    :param type:
        What kind of asset this is.

    :param liquidation_theshold:
        Aave liquidation threhold for this asset, only collateral type asset can have this.
    """

    if require_decimals:
        assert token.decimals, f"Bad token: {token}"
        assert token.decimals > 0, f"Bad token: {token}"

    if liquidation_threshold:
        assert type == AssetType.collateral, f"Only collateral tokens can have liquidation threshold, got {type}"

    return AssetIdentifier(
        token.chain_id.value,
        token.address,
        token.symbol,
        token.decimals,
        underlying=underlying,
        type=type,
        liquidation_threshold=liquidation_threshold,
    )


def translate_trading_pair(dex_pair: DEXPair, cache: dict | None = None) -> TradingPairIdentifier:
    """Translate trading pair from client download to the trade executor.

    Trading Strategy client uses compressed columnar data for pairs and tokens.

    Translates a trading pair presentation from Trading Strategy client Pandas format to the trade executor format.

    Trade executor work with multiple different strategies, not just Trading Strategy client based.
    For example, you could have a completely on-chain data based strategy.
    Thus, Trade Executor has its internal asset format.

    This module contains functions to translate asset presentations between Trading Strategy client
    and Trade Executor.


    This is called when a trade is made: this is the moment when trade executor data format must be made available.

    :param cache:
        Cache of constructed objects.

        Pair internal id -> TradingPairIdentifier

        See :py:class:`tradingstrategy.state.identifier.AssetIdentifier` for life cycle notes.
    """

    if cache is not None:
        cached = cache.get(dex_pair.pair_id)
        if cached is not None:
            return cached

    assert isinstance(dex_pair, DEXPair), f"Expected DEXPair, got {type(dex_pair)}"
    assert dex_pair.base_token_decimals is not None, f"Base token missing decimals: {dex_pair}"
    assert dex_pair.quote_token_decimals is not None, f"Quote token missing decimals: {dex_pair}"

    base = AssetIdentifier(
        chain_id=dex_pair.chain_id.value,
        address=dex_pair.base_token_address,
        token_symbol=dex_pair.base_token_symbol,
        decimals=dex_pair.base_token_decimals,
    )
    quote = AssetIdentifier(
        chain_id=dex_pair.chain_id.value,
        address=dex_pair.quote_token_address,
        token_symbol=dex_pair.quote_token_symbol,
        decimals=dex_pair.quote_token_decimals,
    )

    if dex_pair.fee and isnan(dex_pair.fee):
        # Repair some broken data
        fee = None
    else:
        # Convert DEXPair.fee BPS to %
        # So, after this, fee can either be multiplier or None
        if dex_pair.fee is not None:
            # If BPS fee is set it must be more than 1 BPS.
            # Allow explicit fee = 0 in testing.
            # if pair.fee != 0:
            #     assert pair.fee > 1, f"DEXPair fee must be in BPS, got {pair.fee}"

            # can receive fee in bps or multiplier, but not raw form
            if dex_pair.fee >= 1:
                fee = dex_pair.fee / 10_000
            else:
                fee = dex_pair.fee

            # highest fee tier is currently 1% and lowest in 0.01%
            if fee != 0:
                assert 0.0001 <= fee <= 0.01, f"bug in converting fee to multiplier, make sure bps, got fee {fee}"
        else:
            fee = None

    if dex_pair.dex_type == ExchangeType.erc_4626_vault:
        # For vaults, exchange_name is set as the vault protocol name e.g. "morpho" or "ipor"
        kind = TradingPairKind.vault
    else:
        kind = TradingPairKind.spot_market_hold

    pair = TradingPairIdentifier(
        base=base,
        quote=quote,
        pool_address=dex_pair.address,
        internal_id=int(dex_pair.pair_id),
        info_url=dex_pair.get_trading_pair_page_url(),
        exchange_address=dex_pair.exchange_address,
        fee=fee,
        reverse_token_order=dex_pair.token0_symbol != dex_pair.base_token_symbol,
        exchange_name=dex_pair.exchange_name,
        kind=kind,
        internal_exchange_id=dex_pair.exchange_id,
    )

    # Need to be loaded with load_extra_metadata()
    if dex_pair.buy_tax and dex_pair.buy_tax < 900:
        # 900+ are error codes for built-in internal token tax measurer
        # that should be no longer used - don't bring over these error codes from DEXPair
        pair.base.other_data["buy_tax"] = dex_pair.buy_tax
        pair.base.other_data["sell_tax"] = dex_pair.sell_tax

    # Need to be loaded with load_extra_metadata().
    # see _reduce_other_data() for caveats.
    if dex_pair.other_data:
        # Because other_data is very heavy, we should only copy fields we really care.
        # Below are the whitelisted fields.

        pair.other_data = {}

        # Pass and parse TokenMetadata instance
        token_sniffer_data = None
        metadata = dex_pair.other_data.get("token_metadata")

        match metadata:
            case TokenMetadata():
                pair.other_data["token_metadata"] = metadata
                token_sniffer_data = metadata.token_sniffer_data
            case VaultMetadata():
                pair.other_data["token_metadata"] = metadata
                pair.other_data["vault_features"] = metadata.features
                pair.other_data["vault_protocol"] = metadata.protocol_slug
                pair.other_data["vault_name"] = metadata.vault_name
                pair.other_data["vault_performance_fee"] = metadata.performance_fee
                pair.other_data["vault_management_fee"] = metadata.management_fee
            case None:
                pass
            case _:
                raise NotImplementedError(f"Unknown token metadata type {type(metadata)}")

        if token_sniffer_data is None:
            token_sniffer_data = dex_pair.other_data.get("token_sniffer_data")

        if token_sniffer_data:
            # TODO: Legacy, remove. Instead use TradingPairIdentifier.get_xxx() accessor functions.
            pair.other_data.update({
                "token_sniffer_data": {
                    "swap_simulation": token_sniffer_data.get("swap_simulation"),
                    "score": token_sniffer_data.get("score"),
                }
            })

    # if dex_pair.dex_type == ExchangeType.erc_4626_vault:
    #    import ipdb ; ipdb.set_trace()

    if cache is not None:
        cache[pair.internal_id] = pair

    return pair


def translate_credit_reserve(
    lending_reserve: LendingReserve,
    strategy_reserve: AssetIdentifier,
) -> TradingPairIdentifier:
    """Translate lending protocol reserve from client download to the trade executor.

    :param lending_reserve:
        Raw Token data from Trading Strategy Client

    :param reverse:
        The trading universe reserve asset
    """

    assert isinstance(lending_reserve, LendingReserve)
    internal_id = lending_reserve.reserve_id
    atoken = lending_reserve.get_atoken()

    assert isinstance(atoken, Token)
    assert isinstance(strategy_reserve, AssetIdentifier)

    lending_reserve_underlying = translate_token(lending_reserve.get_asset())

    # TODO: This is the hack fix when Polygon renamed
    # token symbol USDC -> USDC.e
    # In this case
    # Reserve asset: AssetIdentifier(chain_id=137, address='0x2791bca1f2de4661ed88a30c99a7a9449aa84174', token_symbol='USDC', decimals=6, internal_id=None, info_url=None, underlying=None, type=None, liquidation_threshold=None)
    # Underlying: AssetIdentifier(chain_id=137, address='0x2791bca1f2de4661ed88a30c99a7a9449aa84174', token_symbol='USDC.e', decimals=6, internal_id=None, info_url=None, underlying=None, type=<AssetType.token: 'token'>, liquidation_threshold=None)
    if lending_reserve.asset_symbol == "USDC.e":
        underlying = reserve_asset = lending_reserve_underlying
    else:
        underlying = strategy_reserve

    atoken = translate_token(atoken, underlying=underlying)

    return TradingPairIdentifier(
        atoken,
        underlying,
        pool_address=strategy_reserve.address,  # TODO: Now using reserve asset
        exchange_address=strategy_reserve.address,  # TODO: Now using reserve asset
        internal_id=internal_id,
        kind=TradingPairKind.credit_supply,
    )
