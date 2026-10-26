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
  and their transfer history.

Withdrawal proceeds are made claimable with
:py:func:`eth_defi.lighter.testing.credit_lighter_pending_balance_on_anvil`,
so the real ``withdrawPendingBalance`` claim executes through the guard.

Install it with :py:meth:`FakeLighterCore.get_patches`, for example
``for target, replacement in core.get_patches(): mocker.patch(target, new=replacement)``.
"""

import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable

from eth_defi.abi import get_deployed_contract
from eth_defi.lighter.constants import LIGHTER_L1_CONTRACT
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

    async def wait_for_lighter_withdrawal_claimable(self, operator: Any, amount: Decimal, timeout: int, requested_at: int, request_id: str) -> Decimal:
        """Make a master secure withdrawal claimable onchain."""
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

    def wait_for_transaction_receipt_robust(self, web3: Web3, tx_hash: Any, **kwargs: Any) -> Any:
        """Wait for a real receipt, then process a Safe L1 ``withdraw`` it carries."""
        receipt = wait_for_transaction_receipt_robust(web3, tx_hash, **kwargs)
        if receipt["status"] == 1:
            self.observe_l1_withdraw(web3, tx_hash)
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
        ]
