"""In-memory Lighter Core for Anvil fork tests.

Lighter's sequencer cannot run on an Ethereum fork, so a Lighter vault test
must fake what the sequencer does while keeping every Ethereum effect real.
:class:`FakeLighterCore` is one stateful object behind every Lighter seam the
executor and the Lagoon deployment import, so all observations agree:

- account creation, sub-account creation and API-key registration;
- public equity reads, per account and as one by-owner snapshot;
- Safe deposits, credited when the executor waits for collateral;
- secure withdrawals of the master account with its API key;
- key-free L1 ``withdraw(accountIndex, ...)`` requests, observed from the
  real Safe module transaction on the fork;
- same-owner transfers between any two accounts, signed by the sender's key,
  and their transfer history;
- public pools: pool state, investor holdings, mint and burn operations with
  nonces and receipts, secure withdrawals with exact history ids, and real
  Safe ``deposit`` transactions observed on the fork. These answer
  :py:mod:`eth_defi.lighter.execution` through :class:`FakeLighterApiSession`,
  so upstream sizing, serialisation and reconciliation run unchanged.

A pool simulation spanning several command processes saves this state with
:py:meth:`FakeLighterCore.save` and reloads it with :py:meth:`FakeLighterCore.load`.

Withdrawal proceeds are made claimable with
:py:func:`eth_defi.lighter.testing.credit_lighter_pending_balance_on_anvil`,
so the real ``withdrawPendingBalance`` claim executes through the guard.

Install it with :py:meth:`FakeLighterCore.get_patches`, for example
``for target, replacement in core.get_patches(): mocker.patch(target, new=replacement)``.
"""

import datetime
import json
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from eth_defi.abi import get_deployed_contract
from eth_defi.lighter.constants import LIGHTER_API_URL, LIGHTER_L1_CONTRACT
from eth_defi.lighter.execution import API_SUCCESS, LIGHTER_USDC_ASSET_ID
from eth_defi.lighter.session import LighterSession
from eth_defi.lighter.api import LighterAccounts
from eth_defi.lighter.lagoon import request_usdc_withdrawal_from_lighter_account
from eth_defi.lighter.testing import credit_lighter_pending_balance_on_anvil, register_lighter_account_on_anvil
from eth_defi.lighter.transfer import LighterTransferError, LighterTransferRecord
from eth_defi.lighter.valuation import LighterEquity
from eth_defi.provider.receipt import wait_for_transaction_receipt_robust
from web3 import Web3

from tradeexecutor.exchange_account.lighter_operator import LighterFundingRequest

#: First sub-account index handed out; Lighter counts down from ``2**48 - 2``.
FAKE_LIGHTER_FIRST_SUB_ACCOUNT_INDEX = 2**48 - 1000

#: Raw units per USDC.
_USDC_SCALE = Decimal(10) ** 6


#: Lighter's withdrawal history end cursor, ``{"index":-1,"index2":-1}`` in base64.
FAKE_LIGHTER_END_CURSOR = "eyJpbmRleCI6LTEsImluZGV4MiI6LTF9"

#: Liquidity pool (LLP) index reported by the fake system configuration.
FAKE_LIGHTER_LLP_INDEX = 281474976710654

#: Fake transaction types returned by :class:`FakeLighterPoolSigner`.
FAKE_TX_TYPE_MINT = 1001
FAKE_TX_TYPE_BURN = 1002
FAKE_TX_TYPE_WITHDRAW = 1003


@dataclass(slots=True)
class FakeLighterPool:
    """Public pool state, in the API's units: rates are percentage points."""

    #: Pool account index.
    index: int
    #: Gross pool equity in USDC.
    equity: Decimal
    #: Raw share supply.
    total_shares: int
    #: Operator-owned raw shares.
    operator_shares: int
    #: Minimum operator share of supply, percentage points.
    min_operator_share_rate: Decimal = Decimal(5)
    #: Operator fee on positive investor profit, percentage points.
    operator_fee: Decimal = Decimal(20)
    #: Zero permits new investment.
    status: int = 0
    #: Fractional equity growth applied before every burn, so exits realise profit.
    growth_per_burn: Decimal = Decimal("0.001")
    #: Seconds after a mint during which its shares cannot be burned; Lighter uses five minutes.
    lockup_seconds: int = 5

    def get_share_value(self) -> Decimal:
        """Return USDC per raw share."""
        return self.equity / self.total_shares


@dataclass(slots=True)
class FakeLighterHolding:
    """An investor's raw shares in one pool and their cost."""

    #: Raw shares.
    shares: int = 0
    #: Cost of the shares still held, USDC.
    principal: Decimal = Decimal(0)
    #: Unix time of the last entry in milliseconds, as the real API reports it.
    entry_timestamp: int = 0


@dataclass(slots=True)
class FakeLighterAccount:
    """One account in the fake Lighter Core."""

    #: Lighter account index.
    index: int

    #: ``master`` or ``sub_account``.
    role: str

    #: USDC collateral; the fake has no open positions, so this is the equity.
    collateral: Decimal = Decimal(0)

    #: Collateral locked as margin by the external trader.
    margin_in_use: Decimal = Decimal(0)

    #: Registered public keys by API-key slot.
    api_keys: dict[int, str] = field(default_factory=dict)

    #: Pool holdings by pool index.
    pool_shares: dict[int, FakeLighterHolding] = field(default_factory=dict)

    def get_equity(self) -> LighterEquity:
        """Return the public equity record of the account."""
        return LighterEquity(
            account_index=self.index,
            collateral=self.collateral,
            unrealised_pnl=Decimal(0),
            total_asset_value=self.collateral,
            available_balance=max(Decimal(0), self.collateral - self.margin_in_use),
            initial_margin_requirement=self.margin_in_use,
            maintenance_margin_requirement=self.margin_in_use,
            position_count=0,
        )


@dataclass(slots=True)
class FakeLighterTransfer:
    """One submitted same-owner transfer."""

    #: Transfer hash.
    tx_hash: str

    #: Debited account.
    from_index: int

    #: Credited account.
    to_index: int

    #: Credited amount.
    amount: Decimal

    #: Fee charged on top of the amount.
    fee: Decimal

    #: Whether the sequencer has executed the transfer.
    executed: bool = False

    def as_record(self) -> LighterTransferRecord:
        """Return the transfer-history row."""
        return LighterTransferRecord(
            tx_hash=self.tx_hash,
            from_account_index=self.from_index,
            to_account_index=self.to_index,
            amount=self.amount,
            fee=self.fee,
            timestamp=int(time.time()),
        )


class FakeLighterCore:
    """Stateful stand-in for the Lighter sequencer behind one Safe.

    :param web3:
        Anvil fork connection.
    :param master_index:
        Account index assigned to the Safe's master account.
    :param transfer_fee:
        Fee charged on top of every transfer.
    """

    def __init__(self, web3: Web3, master_index: int, transfer_fee: Decimal = Decimal(0)):
        self.web3 = web3
        self.master_index = master_index
        self.transfer_fee = transfer_fee
        #: Execute transfers immediately; set ``False`` to leave them pending.
        self.execute_transfers = True
        self.owner: str | None = None
        self.accounts: dict[int, FakeLighterAccount] = {}
        self.transfers: dict[str, FakeLighterTransfer] = {}
        #: Executed withdrawals as ``(method, account_index, amount)``.
        self.withdrawals: list[tuple[str, int, Decimal]] = []
        self._next_sub_index = FAKE_LIGHTER_FIRST_SUB_ACCOUNT_INDEX
        self._counter = 0
        #: Public pools by index.
        self.pools: dict[int, FakeLighterPool] = {}
        #: Next API nonce by ``"account:key"``.
        self.nonces: dict[str, int] = {}
        #: Executed API transactions by hash, in the ``/api/v1/tx`` shape.
        self.api_transactions: dict[str, dict[str, Any]] = {}
        #: Secure-withdrawal history rows, newest last.
        self.secure_withdrawals: list[dict[str, Any]] = []
        #: Processed Ethereum deposit transactions, to credit each once.
        self.observed_deposits: set[str] = set()

    # ----- State helpers -----

    def get_account(self, account_index: int) -> FakeLighterAccount:
        """Return an existing account."""
        assert account_index in self.accounts, f"Unknown fake Lighter account {account_index}"
        return self.accounts[account_index]

    def get_sub_account_indices(self) -> list[int]:
        """Return sub-account indices in creation order."""
        return [index for index, account in self.accounts.items() if account.role == "sub_account"]

    def create_sub_account(self) -> int:
        """Create one sub-account under the master and return its index."""
        index = self._next_sub_index
        self._next_sub_index -= 1
        self.accounts[index] = FakeLighterAccount(index=index, role="sub_account")
        return index

    def _new_hash(self, prefix: str) -> str:
        self._counter += 1
        return "0x" + f"{prefix}{self._counter:06d}".encode().hex().ljust(64, "0")

    def _debit(self, account_index: int, amount: Decimal) -> None:
        account = self.get_account(account_index)
        free = account.get_equity().available_balance
        if amount > free:
            raise LighterTransferError(f"Fake Lighter account {account_index} has {free} free USDC, cannot move {amount}")
        account.collateral -= amount

    def _credit_pending_balance(self, amount: Decimal) -> None:
        assert self.owner is not None, "Fake Lighter master account is not registered"
        credit_lighter_pending_balance_on_anvil(self.web3, self.owner, int(amount * _USDC_SCALE))

    def execute_pending_transfers(self) -> None:
        """Execute every submitted but not executed transfer."""
        for transfer in self.transfers.values():
            if not transfer.executed:
                self._debit(transfer.from_index, transfer.amount + transfer.fee)
                self.get_account(transfer.to_index).collateral += transfer.amount
                transfer.executed = True

    # ----- Deployment seams (eth_defi.erc_4626.vault_protocol.lagoon.deployment) -----

    def create_lighter_session(self) -> "FakeLighterCore":
        """Return a session stand-in for the deployment ceremony."""
        return self

    def close(self) -> None:
        """Close the session stand-in."""

    def __enter__(self) -> "FakeLighterCore":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def wait_for_lighter_account(self, session: Any, l1_address: str, timeout: int = 900) -> int:
        """Register the Safe's master account on the fork, as the sequencer does after the first deposit."""
        register_lighter_account_on_anvil(self.web3, l1_address, self.master_index)
        self.owner = Web3.to_checksum_address(l1_address)
        self.accounts.setdefault(self.master_index, FakeLighterAccount(index=self.master_index, role="master"))
        return self.master_index

    def wait_for_lighter_collateral(self, session: Any, account_index: int, expected_usdc: Decimal, timeout: int = 900) -> Decimal:
        """Credit a real L1 deposit: raise the account collateral to the expected value."""
        account = self.get_account(account_index)
        account.collateral = max(account.collateral, Decimal(expected_usdc))
        return account.collateral

    def wait_for_lighter_api_key(self, session: Any, account_index: int, api_key_index: int, expected_public_key: str, timeout: int = 300) -> None:
        """Record a key registered by a real Safe ``changePubKey`` transaction."""
        self.get_account(account_index).api_keys[api_key_index] = expected_public_key

    def create_lighter_sub_accounts(self, *, session: Any, l1_address: str, master_account_index: int, api_key_index: int, api_private_key: str, count: int, timeout: int = 300, api_url: str | None = None) -> list[int]:
        """Create sub-accounts signed by the master key."""
        assert Web3.to_checksum_address(l1_address) == self.owner
        assert master_account_index == self.master_index
        assert api_key_index in self.get_account(self.master_index).api_keys, "Master API key is not registered"
        return [self.create_sub_account() for _ in range(count)]

    # ----- Public read seams -----

    def fetch_lighter_total_equity(self, session: Any, account_index: int, timeout: float = 30.0) -> LighterEquity:
        """Return one account's public equity."""
        return self.get_account(int(account_index)).get_equity()

    def fetch_lighter_accounts(self, session: Any, l1_address: str, timeout: float = 30.0) -> LighterAccounts | None:
        """Return the owner's master and sub-accounts."""
        if self.owner is None or Web3.to_checksum_address(l1_address) != self.owner:
            return None
        return LighterAccounts(master_index=self.master_index, sub_account_indices=tuple(self.get_sub_account_indices()))

    def fetch_lighter_accounts_equity(self, session: Any, l1_address: str, timeout: float = 30.0) -> dict[int, LighterEquity]:
        """Return every owned account's equity as one snapshot."""
        assert Web3.to_checksum_address(l1_address) == self.owner
        return {index: account.get_equity() for index, account in self.accounts.items()}

    # ----- Routing seams -----

    async def request_lighter_withdrawal(self, operator: Any, amount: Decimal) -> str:
        """Accept a master secure withdrawal signed with the master key."""
        assert operator.account_index == self.master_index
        self._debit(self.master_index, amount)
        self.withdrawals.append(("api_secure_withdrawal", self.master_index, amount))
        return self._new_hash("wd")

    async def wait_for_lighter_withdrawal_claimable(self, operator: Any, amount: Decimal, timeout: int, requested_at: int, request_id: str | None = None) -> Decimal:
        """Make a master secure withdrawal claimable onchain.

        ``request_id`` is passed by the routing layer and not by
        ``lighter-move-funds``; the fake does not need it.
        """
        self._credit_pending_balance(amount)
        return amount

    async def request_lighter_funding_transfer(self, operator: Any, to_account_index: int, amount: Decimal) -> LighterFundingRequest:
        """Accept a same-owner transfer signed by the sending account's key."""
        from_index = int(operator.account_index)
        assert from_index != to_account_index, "Transfer source and destination must differ"
        assert self.get_account(to_account_index) is not None
        free = self.get_account(from_index).get_equity().available_balance
        if amount + self.transfer_fee > free:
            raise LighterTransferError("Lighter transfer was rejected")
        transfer = FakeLighterTransfer(self._new_hash("tr"), from_index, to_account_index, amount, self.transfer_fee)
        self.transfers[transfer.tx_hash] = transfer
        if self.execute_transfers:
            self.execute_pending_transfers()
        return LighterFundingRequest(tx_hash=transfer.tx_hash, fee=self.transfer_fee)

    async def wait_for_lighter_funding_transfer(self, operator: Any, tx_hash: str, timeout: int) -> LighterTransferRecord:
        """Return an executed transfer, or time out at once for a pending one."""
        transfer = self.transfers[tx_hash]
        if not transfer.executed:
            raise TimeoutError(f"Lighter transfer {tx_hash} was not visible in transfer history within {timeout} seconds")
        return transfer.as_record()

    async def fetch_lighter_funding_transfer(self, operator: Any, tx_hash: str) -> LighterTransferRecord | None:
        """Return the history row of an executed transfer."""
        transfer = self.transfers.get(tx_hash)
        return transfer.as_record() if transfer and transfer.executed else None

    def observe_l1_withdraw(self, web3: Web3, tx_hash: Any) -> None:
        """Process a mined Safe L1 ``withdraw`` like the sequencer.

        The real contract only queues a priority request; the fake debits the
        account and makes the proceeds claimable, as Lighter does once the
        batch is proven.

        :param web3:
            Fork connection.
        :param tx_hash:
            Mined Safe module ``performCall`` transaction.
        """
        tx = web3.eth.get_transaction(tx_hash)
        module = get_deployed_contract(web3, "safe-integration/TradingStrategyModuleV0.json", tx["to"])
        function, arguments = module.decode_function_input(tx["input"])
        target = Web3.to_checksum_address(next(value for value in arguments.values() if isinstance(value, str)))
        if function.fn_name != "performCall" or target != Web3.to_checksum_address(LIGHTER_L1_CONTRACT):
            return
        call_data = next(value for value in arguments.values() if isinstance(value, (bytes, bytearray)))
        zk = get_deployed_contract(web3, "lighter/ZkLighter.json", target)
        zk_function, zk_arguments = zk.decode_function_input(call_data)
        if zk_function.fn_name == "withdraw":
            account_index = int(zk_arguments["_accountIndex"])
            amount = Decimal(int(zk_arguments["_baseAmount"])) / _USDC_SCALE
            self._debit(account_index, amount)
            self.withdrawals.append(("l1_withdraw", account_index, amount))
            self._credit_pending_balance(amount)

    def observe_l1_deposit(self, web3: Web3, tx_hash: Any) -> None:
        """Credit a mined Safe ZkLighter ``deposit`` like the sequencer, once.

        :param web3:
            Fork connection.
        :param tx_hash:
            Mined Safe module ``performCall`` transaction.
        """
        key = Web3.to_hex(tx_hash) if not isinstance(tx_hash, str) else tx_hash.lower()
        if key in self.observed_deposits:
            return
        tx = web3.eth.get_transaction(tx_hash)
        module = get_deployed_contract(web3, "safe-integration/TradingStrategyModuleV0.json", tx["to"])
        try:
            function, arguments = module.decode_function_input(tx["input"])
        except ValueError:
            return
        target = Web3.to_checksum_address(next(value for value in arguments.values() if isinstance(value, str)))
        if function.fn_name != "performCall" or target != Web3.to_checksum_address(LIGHTER_L1_CONTRACT):
            return
        call_data = next(value for value in arguments.values() if isinstance(value, (bytes, bytearray)))
        zk = get_deployed_contract(web3, "lighter/ZkLighter.json", target)
        zk_function, zk_arguments = zk.decode_function_input(call_data)
        if zk_function.fn_name != "deposit":
            return
        owner = Web3.to_checksum_address(next(value for value in zk_arguments.values() if isinstance(value, str)))
        amount_raw = next(value for name, value in zk_arguments.items() if "amount" in name.lower())
        assert owner == self.owner, f"Deposit to {owner}, but the fake master account belongs to {self.owner}"
        self.get_account(self.master_index).collateral += Decimal(int(amount_raw)) / _USDC_SCALE
        self.observed_deposits.add(key)

    def wait_for_transaction_receipt_robust(self, web3: Web3, tx_hash: Any, **kwargs: Any) -> Any:
        """Wait for a real receipt, then process a Safe L1 ``withdraw`` or pool ``deposit`` it carries."""
        receipt = wait_for_transaction_receipt_robust(web3, tx_hash, **kwargs)
        if receipt["status"] == 1:
            self.observe_l1_withdraw(web3, tx_hash)
            if self.owner is not None:
                self.observe_l1_deposit(web3, tx_hash)
        return receipt

    def request_usdc_withdrawal_from_lighter_account(self, web3: Web3, hot_wallet: Any, **kwargs: Any) -> str:
        """Send the real L1 ``withdraw`` request, then process it like the sequencer."""
        tx_hash = request_usdc_withdrawal_from_lighter_account(web3, hot_wallet, **kwargs)
        self.observe_l1_withdraw(web3, tx_hash)
        return tx_hash

    def fetch_lighter_account_by_index(self, session: Any, account_index: int, timeout: float = 30.0) -> dict[str, Any]:
        """Return a raw public account record."""
        equity = self.get_account(int(account_index)).get_equity()
        return {
            "account_index": equity.account_index,
            "collateral": str(equity.collateral),
            "total_asset_value": str(equity.total_asset_value),
            "available_balance": str(equity.available_balance),
            "positions": [],
        }

    def fetch_lighter_withdrawal_delay(self, session: Any, timeout: float = 30.0) -> int:
        """Return a fixed secure-withdrawal delay estimate."""
        return 60

    # ----- Public pools -----

    def add_pool(self, pool: FakeLighterPool) -> None:
        """Make a public pool available for investment."""
        self.pools[pool.index] = pool

    def _account_record(self, index: int) -> dict[str, Any]:
        """Return an ``/api/v1/account`` row for an investor or a pool."""
        if index in self.pools:
            pool = self.pools[index]
            return {
                "account_index": index,
                "collateral": str(pool.equity),
                "available_balance": str(pool.equity),
                "total_asset_value": str(pool.equity),
                "cross_initial_margin_requirement": "0",
                "cross_maintenance_margin_requirement": "0",
                "positions": [],
                "shares": [],
                "pool_info": {
                    "status": pool.status,
                    "total_shares": pool.total_shares,
                    "operator_shares": pool.operator_shares,
                    "min_operator_share_rate": str(pool.min_operator_share_rate),
                    "operator_fee": str(pool.operator_fee),
                },
            }
        account = self.get_account(index)
        equity = account.get_equity()
        # Simulation assumption: account equity excludes pool holdings. Executor
        # valuation never adds the two, so the choice cannot double count.
        return {
            "account_index": index,
            "l1_address": self.owner,
            "collateral": str(equity.collateral),
            "available_balance": str(equity.available_balance),
            "total_asset_value": str(equity.total_asset_value),
            "cross_initial_margin_requirement": str(equity.initial_margin_requirement),
            "cross_maintenance_margin_requirement": str(equity.maintenance_margin_requirement),
            "positions": [],
            "shares": [
                {
                    "public_pool_index": pool_index,
                    "shares_amount": holding.shares,
                    "principal_amount": str(holding.principal),
                    "entry_usdc": str(holding.principal),
                    "entry_timestamp": holding.entry_timestamp,
                }
                for pool_index, holding in account.pool_shares.items()
                if holding.shares > 0
            ],
        }

    def handle_get(self, path: str, params: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """Answer one replay-safe Lighter API read.

        :param path:
            API path such as ``/api/v1/account``.
        :param params:
            Query parameters.
        :return:
            HTTP status and JSON body.
        """
        if path == "/api/v1/account":
            index = int(params["value"])
            if index not in self.pools and index not in self.accounts:
                return 400, {"code": 21100, "message": "account not found"}
            return 200, {"code": API_SUCCESS, "accounts": [self._account_record(index)]}
        if path == "/api/v1/systemConfig":
            return 200, {"code": API_SUCCESS, "liquidity_pool_index": FAKE_LIGHTER_LLP_INDEX, "liquidity_pool_cooldown_period": 300_000}
        if path == "/api/v1/nextNonce":
            return 200, {"code": API_SUCCESS, "nonce": self.nonces.get(f"{params['account_index']}:{params['api_key_index']}", 0)}
        if path == "/api/v1/tx":
            tx = self.api_transactions.get(str(params["value"]).lower())
            if tx is None:
                return 400, {"code": 21500, "message": "transaction not found"}
            return 200, {"code": API_SUCCESS, **tx}
        if path == "/api/v1/withdraw/history":
            # Mainnet returns the same end cursor after the last row and an
            # empty page when that cursor is followed (observed 2026-11-03).
            rows = [] if params.get("cursor") == FAKE_LIGHTER_END_CURSOR else list(self.secure_withdrawals)
            return 200, {"code": API_SUCCESS, "withdraws": rows, "cursor": FAKE_LIGHTER_END_CURSOR}
        raise AssertionError(f"Fake Lighter API has no read {path}")

    def handle_send_tx(self, tx_type: int, tx_info: str) -> dict[str, Any]:
        """Execute one signed pool or withdrawal transaction immediately.

        :param tx_type:
            Fake transaction type from :class:`FakeLighterPoolSigner`.
        :param tx_info:
            Signed payload JSON.
        :return:
            ``sendTx`` response body.
        """
        payload = json.loads(tx_info)
        tx_hash = payload["Hash"]
        account_index = int(payload.get("AccountIndex", payload.get("FromAccountIndex")))
        api_key_index = int(payload["ApiKeyIndex"])
        nonce_key = f"{account_index}:{api_key_index}"
        if int(payload["Nonce"]) != self.nonces.get(nonce_key, 0):
            return {"code": 21104, "message": "invalid nonce"}
        account = self.get_account(account_index)
        now = int(time.time())
        if tx_type == FAKE_TX_TYPE_MINT:
            pool = self.pools[int(payload["PublicPoolIndex"])]
            shares = int(payload["ShareAmount"])
            cost = shares * pool.get_share_value()
            if cost > account.get_equity().available_balance:
                return {"code": 21700, "message": "insufficient collateral"}
            account.collateral -= cost
            pool.equity += cost
            pool.total_shares += shares
            holding = account.pool_shares.setdefault(pool.index, FakeLighterHolding())
            holding.shares += shares
            holding.principal += cost
            holding.entry_timestamp = now * 1000
        elif tx_type == FAKE_TX_TYPE_BURN:
            pool = self.pools[int(payload["PublicPoolIndex"])]
            shares = int(payload["ShareAmount"])
            holding = account.pool_shares.get(pool.index)
            if holding is None or holding.shares < shares:
                return {"code": 21701, "message": "insufficient shares"}
            if now * 1000 < holding.entry_timestamp + pool.lockup_seconds * 1000:
                return {"code": 21705, "message": "pool shares are still locked"}
            pool.equity *= 1 + pool.growth_per_burn
            gross = shares * pool.get_share_value()
            principal = holding.principal * shares / holding.shares
            fee = max(gross - principal, Decimal(0)) * pool.operator_fee / 100
            account.collateral += gross - fee
            # The fee stays in the pool for the operator.
            pool.equity -= gross - fee
            pool.total_shares -= shares
            holding.shares -= shares
            holding.principal -= principal
        elif tx_type == FAKE_TX_TYPE_WITHDRAW:
            assert int(payload["AssetIndex"]) == LIGHTER_USDC_ASSET_ID and int(payload["RouteType"]) == 0
            amount = Decimal(int(payload["Amount"])) / _USDC_SCALE
            if amount > account.get_equity().available_balance:
                return {"code": 21702, "message": "insufficient collateral"}
            account.collateral -= amount
            self._counter += 1
            # Proven at once: the real claim then runs through the guard.
            self.secure_withdrawals.append({"id": str(self._counter), "type": "secure", "asset_id": LIGHTER_USDC_ASSET_ID, "amount": str(amount), "status": "claimable", "timestamp": now})
            self._credit_pending_balance(amount)
            self.withdrawals.append(("api_secure_withdrawal", account_index, amount))
        else:
            raise AssertionError(f"Unknown fake Lighter transaction type {tx_type}")
        self.nonces[nonce_key] = self.nonces.get(nonce_key, 0) + 1
        self.api_transactions[tx_hash.lower()] = {
            "hash": tx_hash,
            "account_index": account_index,
            "api_key_index": api_key_index,
            "nonce": int(payload["Nonce"]),
            "type": tx_type,
            "status": 2,
            "executed_at": now * 1000,
        }
        return {"code": API_SUCCESS, "tx_hash": tx_hash}

    def create_api_session(self) -> "FakeLighterApiSession":
        """Return an Ethereum Lighter session answered by this fake."""
        return FakeLighterApiSession(self)

    def create_pool_signer(self, operator: Any) -> "FakeLighterPoolSigner":
        """Return a deterministic signer for the operator's account and key."""
        return FakeLighterPoolSigner(operator.account_index)

    def fetch_lighter_api_key(self, session: Any, account_index: int, api_key_index: int, timeout: float = 30.0) -> dict[str, Any] | None:
        """Report the API-key slot, empty until registered."""
        public_key = self.accounts.get(account_index, FakeLighterAccount(account_index, "master")).api_keys.get(api_key_index)
        return {"public_key": public_key} if public_key else None

    # ----- Persistence across command processes -----

    def save(self, path: Path) -> None:
        """Write the fake venue state for the next command process.

        :param path:
            Simulation state file outside the executor run directory.
        """
        data = {
            "master_index": self.master_index,
            "owner": self.owner,
            "counter": self._counter,
            "accounts": {
                str(index): {
                    "role": account.role,
                    "collateral": str(account.collateral),
                    "api_keys": {str(k): v for k, v in account.api_keys.items()},
                    "pool_shares": {str(pool): {"shares": h.shares, "principal": str(h.principal), "entry_timestamp": h.entry_timestamp} for pool, h in account.pool_shares.items()},
                }
                for index, account in self.accounts.items()
            },
            "pools": {
                str(index): {
                    "equity": str(pool.equity),
                    "total_shares": pool.total_shares,
                    "operator_shares": pool.operator_shares,
                    "min_operator_share_rate": str(pool.min_operator_share_rate),
                    "operator_fee": str(pool.operator_fee),
                    "status": pool.status,
                }
                for index, pool in self.pools.items()
            },
            "nonces": self.nonces,
            "api_transactions": self.api_transactions,
            "secure_withdrawals": self.secure_withdrawals,
            "observed_deposits": sorted(self.observed_deposits),
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, web3: Web3, path: Path) -> "FakeLighterCore":
        """Restore the fake venue state saved by an earlier command process.

        :param web3:
            Ethereum fork connection.
        :param path:
            File written by :py:meth:`save`.
        :return:
            Fake core with the same accounts, pools and history.
        """
        data = json.loads(path.read_text())
        core = cls(web3, data["master_index"])
        core.owner = data["owner"]
        core._counter = data["counter"]
        for index, raw in data["accounts"].items():
            account = FakeLighterAccount(index=int(index), role=raw["role"], collateral=Decimal(raw["collateral"]), api_keys={int(k): v for k, v in raw["api_keys"].items()})
            account.pool_shares = {int(pool): FakeLighterHolding(h["shares"], Decimal(h["principal"]), h["entry_timestamp"]) for pool, h in raw["pool_shares"].items()}
            core.accounts[account.index] = account
        for index, raw in data["pools"].items():
            core.pools[int(index)] = FakeLighterPool(
                index=int(index),
                equity=Decimal(raw["equity"]),
                total_shares=raw["total_shares"],
                operator_shares=raw["operator_shares"],
                min_operator_share_rate=Decimal(raw["min_operator_share_rate"]),
                operator_fee=Decimal(raw["operator_fee"]),
                status=raw["status"],
            )
        core.nonces = data["nonces"]
        core.api_transactions = data["api_transactions"]
        core.secure_withdrawals = data["secure_withdrawals"]
        core.observed_deposits = set(data["observed_deposits"])
        return core

    def get_pool_patches(self) -> list[tuple[str, Callable[..., Any]]]:
        """Return the seams of Lighter pool execution and coordinated activation.

        Use together with :py:meth:`get_patches`.

        :return:
            Patch targets for ``mock.patch(target, new=replacement)``.
        """
        activation = "eth_defi.lighter.activation"
        pool_routing = "tradeexecutor.ethereum.lighter.lighter_pool_routing"
        return [
            (f"{activation}.create_lighter_session", self.create_lighter_session),
            (f"{activation}.wait_for_lighter_account", self.wait_for_lighter_account),
            (f"{activation}.wait_for_lighter_collateral", self.wait_for_lighter_collateral),
            (f"{activation}.fetch_lighter_api_key", self.fetch_lighter_api_key),
            (f"{activation}.wait_for_lighter_api_key", self.wait_for_lighter_api_key),
            (f"{pool_routing}.create_lighter_pool_session", self.create_api_session),
            ("tradeexecutor.ethereum.lagoon.vault.create_executor_lighter_session", self.create_api_session),
            ("tradeexecutor.cli.commands.init.create_executor_lighter_session", self.create_api_session),
            ("tradeexecutor.cli.commands.lagoon_redeem.create_executor_lighter_session", self.create_api_session),
            ("tradeexecutor.ethereum.vault.settlement_retry.create_executor_lighter_session", self.create_api_session),
            ("tradeexecutor.ethereum.lighter.lighter_pool_valuation.create_lighter_pool_session", self.create_api_session),
            (f"{pool_routing}.create_lighter_pool_signer", self.create_pool_signer),
            (f"{pool_routing}.wait_for_transaction_receipt_robust", self.wait_for_transaction_receipt_robust),
            ("eth_defi.lighter.execution.requests", SimpleNamespace(Session=lambda: FakeLighterSubmission(self))),
            # The fake's lockup is seconds, not Lighter's five minutes; keep the
            # router's wait consistent with it so simulations stay fast.
            (f"{pool_routing}.LIGHTER_POOL_EXIT_LOCKUP", datetime.timedelta(seconds=7)),
        ]

    # ----- Installation -----

    def get_patches(self) -> list[tuple[str, Callable[..., Any]]]:
        """Return ``(import path, replacement)`` for every Lighter seam.

        :return:
            Patch targets for ``mocker.patch(target, new=replacement)``.
        """
        deployment = "eth_defi.erc_4626.vault_protocol.lagoon.deployment"
        routing = "tradeexecutor.ethereum.lighter.lighter_routing"
        verification = "tradeexecutor.ethereum.lighter.transfer_verification"
        return [
            (f"{deployment}.create_lighter_session", self.create_lighter_session),
            (f"{deployment}.wait_for_lighter_account", self.wait_for_lighter_account),
            (f"{deployment}.wait_for_lighter_collateral", self.wait_for_lighter_collateral),
            (f"{deployment}.wait_for_lighter_api_key", self.wait_for_lighter_api_key),
            (f"{deployment}.create_lighter_sub_accounts", self.create_lighter_sub_accounts),
            ("tradeexecutor.exchange_account.lighter.fetch_lighter_total_equity", self.fetch_lighter_total_equity),
            ("tradeexecutor.exchange_account.lighter.fetch_lighter_accounts_equity", self.fetch_lighter_accounts_equity),
            ("tradeexecutor.exchange_account.lighter.fetch_lighter_accounts", self.fetch_lighter_accounts),
            (f"{routing}.fetch_lighter_total_equity", self.fetch_lighter_total_equity),
            (f"{routing}.request_lighter_withdrawal", self.request_lighter_withdrawal),
            (f"{routing}.wait_for_lighter_withdrawal_claimable", self.wait_for_lighter_withdrawal_claimable),
            (f"{routing}.request_lighter_funding_transfer", self.request_lighter_funding_transfer),
            (f"{routing}.wait_for_lighter_funding_transfer", self.wait_for_lighter_funding_transfer),
            (f"{routing}.wait_for_transaction_receipt_robust", self.wait_for_transaction_receipt_robust),
            (f"{verification}.fetch_lighter_total_equity", self.fetch_lighter_total_equity),
            (f"{verification}.wait_for_lighter_collateral", self.wait_for_lighter_collateral),
            (f"{verification}.fetch_lighter_funding_transfer", self.fetch_lighter_funding_transfer),
            ("tradeexecutor.cli.commands.lagoon_redeem.fetch_lighter_total_equity", self.fetch_lighter_total_equity),
            ("tradeexecutor.cli.commands.lighter_move_funds.fetch_lighter_total_equity", self.fetch_lighter_total_equity),
            ("tradeexecutor.cli.commands.lighter_move_funds.fetch_lighter_account_by_index", self.fetch_lighter_account_by_index),
            ("tradeexecutor.cli.commands.lighter_move_funds.fetch_lighter_withdrawal_delay", self.fetch_lighter_withdrawal_delay),
            ("tradeexecutor.cli.commands.lighter_move_funds.request_usdc_withdrawal_from_lighter_account", self.request_usdc_withdrawal_from_lighter_account),
            ("tradeexecutor.cli.commands.lighter_move_funds.request_lighter_withdrawal", self.request_lighter_withdrawal),
            ("tradeexecutor.cli.commands.lighter_move_funds.wait_for_lighter_withdrawal_claimable", self.wait_for_lighter_withdrawal_claimable),
            ("tradeexecutor.testing.lighter_manual.request_lighter_funding_transfer", self.request_lighter_funding_transfer),
            ("tradeexecutor.testing.lighter_manual.wait_for_lighter_funding_transfer", self.wait_for_lighter_funding_transfer),
        ]


def get_master_only_account_patches(master_index: int, equity: Callable[[], LighterEquity]) -> list[tuple[str, Callable[..., Any]]]:
    """Return patches that make Lighter's public API report a Safe owning only its master account.

    Lighter's public API does not know a Safe deployed on an Anvil fork, so a
    test without :py:class:`FakeLighterCore` patches the account list and the
    equity snapshot that the account-set check and the NAV read.

    :param master_index:
        The Safe's master account.
    :param equity:
        Returns the master's current equity.
    :return:
        Patch targets for ``mocker.patch(target, new=replacement)``.
    """
    return [
        ("tradeexecutor.exchange_account.lighter.fetch_lighter_accounts", lambda *_args, **_kwargs: LighterAccounts(master_index=master_index, sub_account_indices=())),
        ("tradeexecutor.exchange_account.lighter.fetch_lighter_accounts_equity", lambda *_args, **_kwargs: {master_index: equity()}),
    ]


class FakeLighterResponse:
    """Minimal ``requests.Response`` stand-in for the fake Lighter API."""

    def __init__(self, status_code: int, body: dict[str, Any]):
        self.status_code = status_code
        self._body = body

    def json(self) -> dict[str, Any]:
        """Return the JSON body."""
        return self._body

    def raise_for_status(self) -> None:
        """Raise like ``requests`` for an HTTP error status."""
        if self.status_code >= 400:
            raise RuntimeError(f"Fake Lighter API HTTP {self.status_code}: {self._body}")


class FakeLighterApiSession(LighterSession):
    """Ethereum Lighter session whose reads are answered by a :class:`FakeLighterCore`.

    It keeps the canonical API URL and deployment, so upstream identity checks
    pass; nothing leaves the process.
    """

    def __init__(self, core: FakeLighterCore):
        super().__init__()
        self.core = core

    def get(self, url: str, params: dict[str, Any] | None = None, **kwargs: Any) -> FakeLighterResponse:
        """Answer a read from the fake state."""
        assert url.startswith(LIGHTER_API_URL), f"Unexpected Lighter URL {url}"
        status, body = self.core.handle_get(url.removeprefix(LIGHTER_API_URL), params or {})
        return FakeLighterResponse(status, body)


class FakeLighterSubmission:
    """Stand-in for the plain ``requests.Session`` upstream uses for ``sendTx``."""

    def __init__(self, core: FakeLighterCore):
        self.core = core

    def __enter__(self) -> "FakeLighterSubmission":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def post(self, url: str, data: dict[str, Any], **kwargs: Any) -> FakeLighterResponse:
        """Submit one signed transaction."""
        assert url == f"{LIGHTER_API_URL}/api/v1/sendTx", f"Unexpected Lighter submission {url}"
        body = self.core.handle_send_tx(int(data["tx_type"]), data["tx_info"])
        return FakeLighterResponse(200 if body["code"] == API_SUCCESS else 400, body)


class FakeLighterPoolSigner:
    """Deterministic stand-in for the official SDK signer of one account.

    Produces payloads with the fields upstream validates and a hash derived
    from them; the fake sequencer accepts nothing else.
    """

    def __init__(self, account_index: int):
        self.account_index = account_index

    def _sign(self, tx_type: int, payload: dict[str, Any]) -> tuple[int, str, str, None]:
        tx_hash = Web3.keccak(text=json.dumps(payload, sort_keys=True)).hex()
        tx_hash = tx_hash if tx_hash.startswith("0x") else "0x" + tx_hash
        return tx_type, json.dumps({**payload, "Hash": tx_hash}), tx_hash, None

    def sign_mint_shares(self, public_pool_index: int, share_amount: int, *, nonce: int, api_key_index: int):
        """Sign a mint."""
        return self._sign(FAKE_TX_TYPE_MINT, {"AccountIndex": self.account_index, "ApiKeyIndex": api_key_index, "Nonce": nonce, "PublicPoolIndex": public_pool_index, "ShareAmount": share_amount})

    def sign_burn_shares(self, public_pool_index: int, share_amount: int, *, nonce: int, api_key_index: int):
        """Sign a burn."""
        return self._sign(FAKE_TX_TYPE_BURN, {"AccountIndex": self.account_index, "ApiKeyIndex": api_key_index, "Nonce": nonce, "PublicPoolIndex": public_pool_index, "ShareAmount": share_amount})

    def sign_withdraw(self, asset_index: int, route_type: int, amount: int, *, nonce: int, api_key_index: int):
        """Sign a secure withdrawal."""
        return self._sign(FAKE_TX_TYPE_WITHDRAW, {"FromAccountIndex": self.account_index, "ApiKeyIndex": api_key_index, "Nonce": nonce, "AssetIndex": asset_index, "RouteType": route_type, "Amount": amount})

    def create_auth_token(self) -> str:
        """Return a placeholder read token."""
        return "fake-auth-token"

    def close(self) -> None:
        """Nothing to close."""
