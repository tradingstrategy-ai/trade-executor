"""Vault fee metadata completeness checks.

- Decide whether a vault has enough fee metadata to model investor-net returns

- Mark vaults with missing fee data as ignored pairs:
  retained in the trading universe for their price data,
  but no new positions may be entered in them,
  see :py:meth:`tradeexecutor.state.identifier.TradingPairIdentifier.get_ignore_reason`

- Build a human-readable table of the ignored vaults,
  rendered by :py:func:`tradeexecutor.strategy.chart.standard.trading_universe.ignored_vault_data`
"""

from typing import TYPE_CHECKING

import pandas as pd
from pandas.io.formats.style import Styler

from tradingstrategy.chain import ChainId
from tradingstrategy.vault import VaultMetadata, get_vault_page

from tradeexecutor.state.identifier import IGNORE_REASON_LACKS_FEE_DATA, TradingPairIdentifier

if TYPE_CHECKING:
    from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


#: Key in :py:attr:`TradingPairIdentifier.other_data` recording which fee fields the vault lacks.
MISSING_VAULT_FEE_DATA_KEY = "missing_vault_fee_data"

#: Columns of the ignored vault table, see :py:func:`build_ignored_vault_dataframe`.
IGNORED_VAULT_COLUMNS = [
    "Name",
    "Chain",
    "Address",
    "Curator",
    "Protocol",
    "Missing fees",
    "Link",
]


def get_missing_vault_fee_data(metadata: VaultMetadata | None) -> list[str]:
    """Check which fee fields are missing to model investor-net vault returns.

    The fee data is considered complete when

    - The net management/performance fee treatment can be derived:
      fees are internalised in the share price, both fees are known zero,
      or fees are externalised and both values are known

    - Deposit and withdrawal fees are known (zero or not)

    :param metadata:
        Vault metadata, or ``None`` if not loaded.

    :return:
        Human-readable names of the missing fee fields.
        Empty list if the fee data is complete.
    """
    if metadata is None:
        return ["vault metadata missing"]

    missing = []

    management_fee = metadata.management_fee
    performance_fee = metadata.performance_fee

    if metadata.fee_internalised or (management_fee == 0 and performance_fee == 0):
        # Net returns derivable from the share price alone
        pass
    elif metadata.fee_internalised is False and management_fee is not None and performance_fee is not None:
        # Externalised fees with known values
        pass
    else:
        if metadata.fee_internalised is None:
            missing.append("fee mode")
        if management_fee is None:
            missing.append("management fee")
        if performance_fee is None:
            missing.append("performance fee")

    if metadata.deposit_fee is None:
        missing.append("deposit fee")
    if metadata.withdrawal_fee is None:
        missing.append("withdrawal fee")

    return missing


def mark_missing_fee_vaults_ignored(
    strategy_universe: "TradingStrategyUniverse",
) -> list[TradingPairIdentifier]:
    """Flag vaults with incomplete fee metadata as ignored pairs.

    - Call in ``create_trading_universe()`` after the universe has been constructed

    - Flagged vaults keep their price data, but no new positions may be entered in them:
      :py:meth:`tradeexecutor.strategy.alpha_model.AlphaModel.set_signal` zeroes their signals,
      see :py:meth:`TradingPairIdentifier.get_ignore_reason`

    - The missing fee field names are recorded in
      :py:attr:`TradingPairIdentifier.other_data` under :py:data:`MISSING_VAULT_FEE_DATA_KEY`

    :param strategy_universe:
        The trading universe to scan and mutate.

    :return:
        Vault pairs that were flagged.
    """
    flagged = []
    for pair in strategy_universe.iterate_pairs():
        if not pair.is_vault():
            continue
        missing = get_missing_vault_fee_data(pair.get_vault_metadata())
        if not missing:
            continue
        pair.set_ignore_reason(IGNORE_REASON_LACKS_FEE_DATA)
        pair.other_data[MISSING_VAULT_FEE_DATA_KEY] = missing
        flagged.append(pair)
    return flagged


def build_ignored_vault_dataframe(
    strategy_universe: "TradingStrategyUniverse",
) -> pd.DataFrame:
    """Build a table of vaults ignored because of missing fee data.

    Lists vault pairs flagged by :py:func:`mark_missing_fee_vaults_ignored`
    with the vault identity, curator and which fee fields are missing.

    :param strategy_universe:
        The trading universe to scan.

    :return:
        DataFrame with columns :py:data:`IGNORED_VAULT_COLUMNS`,
        one row per ignored vault. Empty if no vaults were flagged.
    """
    rows = []
    for pair in strategy_universe.iterate_pairs():
        if not pair.is_vault() or pair.get_ignore_reason() != IGNORE_REASON_LACKS_FEE_DATA:
            continue

        metadata = pair.get_vault_metadata()
        missing = pair.other_data.get(MISSING_VAULT_FEE_DATA_KEY) or get_missing_vault_fee_data(metadata)
        address = pair.pool_address

        link = getattr(metadata, "trading_strategy_link", None) or get_vault_page(address)
        rows.append({
            "Name": pair.get_vault_name() or pair.base.token_symbol,
            "Chain": ChainId(pair.chain_id).get_name(),
            "Address": address,
            "Curator": getattr(metadata, "curator_name", None) or "",
            "Protocol": pair.get_vault_protocol() or getattr(metadata, "protocol_slug", None) or "",
            "Missing fees": ", ".join(missing),
            "Link": link,
        })

    df = pd.DataFrame(rows, columns=IGNORED_VAULT_COLUMNS)
    return df.sort_values(["Chain", "Name"]).reset_index(drop=True)


def style_ignored_vault_table(df: pd.DataFrame) -> Styler:
    """Style the ignored vault table with a clickable Trading Strategy link.

    Renders the ``Link`` column as an HTML anchor labelled with the vault page slug,
    so the URL is clickable instead of truncated text.

    :param df:
        Output of :py:func:`build_ignored_vault_dataframe`.

    :return:
        Styler for notebook display and chart rendering.
    """

    def _linkify(url: str) -> str:
        if not url:
            return ""
        label = url.rstrip("/").rsplit("/", 1)[-1]
        return f'<a href="{url}" target="_blank">{label}</a>'

    return df.style.format({"Link": _linkify}).hide(axis="index")
