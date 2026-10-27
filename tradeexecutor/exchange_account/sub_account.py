"""Sub-accounts a strategy runs as positions on an external exchange.

A strategy declares its sub-accounts in ``Parameters`` as
:py:class:`SubAccount` values: what sub-strategies run and under which
stable id. It never names exchange account indices, which the exchange
assigns at deployment. The deployment records each id's account index in
the deployment artefact, ``init`` copies that resolution into the state,
and the state is the single source of truth for mapping ids to accounts and
positions from then on.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State

#: Id of the Safe-owned master account; never a sub-account id.
LIGHTER_MASTER_ACCOUNT_ID = "master"

#: Sub-account ids are short identifiers usable in file names and tickers.
SUB_ACCOUNT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


@dataclass(frozen=True, slots=True)
class SubAccount:
    """One sub-account declared by a strategy.

    Once a deployment has created the exchange account for an id, keep the
    id unchanged in the strategy and in the state: it is the key that maps
    the sub-strategy to its exchange account and position.
    """

    #: Stable identifier, for example ``trend``; lower-case letters, digits, ``_`` and ``-``.
    id: str

    #: Human-readable name of the sub-strategy run in this account.
    name: str

    #: Target share of the investable capital for the automatic cash policy; ``None`` leaves the account unmanaged.
    weight: float | None = None

    def __post_init__(self) -> None:
        """Reject ids that cannot serve as stable keys."""
        if not SUB_ACCOUNT_ID_PATTERN.match(self.id):
            raise ValueError(f"Sub-account id must match {SUB_ACCOUNT_ID_PATTERN.pattern}: {self.id!r}")
        if self.id == LIGHTER_MASTER_ACCOUNT_ID:
            raise ValueError(f"{LIGHTER_MASTER_ACCOUNT_ID!r} is the master account, not a sub-account id")


def get_sub_account_ids(sub_accounts: Iterable[SubAccount] | None) -> list[str]:
    """Return the declared sub-account ids in order, checking they are unique.

    :param sub_accounts:
        ``Parameters.lighter_sub_accounts``.
    :return:
        Ids in declaration order.
    """
    ids = [sub_account.id for sub_account in sub_accounts or ()]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Sub-account ids must be unique: {ids}")
    return ids


def get_lighter_account_resolution(state: State) -> dict[str, int]:
    """Return the account index of the master and of every sub-account id recorded in the state.

    :param state:
        Executor state initialised from a Lighter deployment.
    :return:
        ``{"master": index, sub_account_id: index, ...}``; empty before ``init`` resolved the accounts.
    """
    return dict(state.sync.deployment.lighter_accounts or {})


def store_lighter_account_resolution(state: State, resolution: dict[str, int]) -> None:
    """Record which Lighter account each id maps to, never changing an existing mapping.

    :param state:
        Executor state.
    :param resolution:
        ``{"master": index, sub_account_id: index, ...}``.
    :raise ValueError:
        If an id is already mapped to a different account.
    """
    current = get_lighter_account_resolution(state)
    for account_id, index in resolution.items():
        if account_id in current and current[account_id] != int(index):
            raise ValueError(f"Lighter account {account_id!r} is already mapped to {current[account_id]}, refusing to change it to {index}")
    current.update({account_id: int(index) for account_id, index in resolution.items()})
    state.sync.deployment.lighter_accounts = current


def read_lighter_account_resolution(payload: dict[str, Any], vault_address: str | None) -> dict[str, int] | None:
    """Read the master and sub-account indices of a vault from its public deployment artefact.

    :param payload:
        Parsed ``<executor id>.deployment.json``.
    :param vault_address:
        Vault the executor runs; the artefact may hold several chains.
    :return:
        ``{"master": index, sub_account_id: index, ...}``, or ``None`` when the
        artefact has no Lighter activation for the vault.
    """
    deployments = payload.get("deployments")
    if not isinstance(deployments, dict):
        return None
    for deployment in deployments.values():
        if not isinstance(deployment, dict):
            continue
        recorded_vault = deployment.get("vault_address")
        if vault_address and (not isinstance(recorded_vault, str) or recorded_vault.lower() != vault_address.lower()):
            continue
        setup = deployment.get("lighter_account_setup")
        if not isinstance(setup, dict) or setup.get("account_index") is None:
            continue
        resolution = {LIGHTER_MASTER_ACCOUNT_ID: int(setup["account_index"])}
        for sub in setup.get("sub_accounts") or []:
            label = sub.get("label")
            if not label:
                raise ValueError(f"Lighter sub-account {sub.get('account_index')} has no id in the deployment artefact; it was deployed without a strategy declaring sub-accounts")
            resolution[str(label)] = int(sub["account_index"])
        return resolution
    return None


def get_sub_account_id_for_position(state: State, position: TradingPosition) -> str | None:
    """Return the id of the account a position tracks, from the state's resolution.

    :param state:
        Executor state.
    :param position:
        An exchange-account position.
    :return:
        ``master``, a sub-account id, or ``None`` when the position's account is not resolved.
    """
    if not position.is_exchange_account():
        return None
    index = int(position.pair.get_exchange_account_id())
    for account_id, resolved in get_lighter_account_resolution(state).items():
        if resolved == index:
            return account_id
    return None


def get_position_for_sub_account(state: State, account_id: str) -> TradingPosition | None:
    """Return the open position of the account with the given id, if funds have moved into it.

    :param state:
        Executor state.
    :param account_id:
        ``master`` or a sub-account id.
    :return:
        The open exchange-account position, or ``None``.
    """
    index = get_lighter_account_resolution(state).get(account_id)
    if index is None:
        raise ValueError(f"Lighter account {account_id!r} is not resolved in the state; run init")
    for position in state.portfolio.open_positions.values():
        if position.is_exchange_account() and int(position.pair.get_exchange_account_id()) == index:
            return position
    return None
