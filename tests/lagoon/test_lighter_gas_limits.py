"""Test the gas limits of Safe-wrapped Lighter transactions.

A node refuses a transaction unless the hot wallet holds its gas limit times the
maximum fee. On 2026-11-05 a fixed 1,500,000 limit stopped a deposit that cost about
200,000 gas, so these limits follow real use.
"""

from types import SimpleNamespace

import pytest
from pytest_mock import MockerFixture
from web3.contract.contract import Contract, ContractFunction

import tradeexecutor.ethereum.lagoon.tx as tx_module
from tradeexecutor.ethereum.lagoon.tx import LagoonTransactionBuilder
from tradeexecutor.ethereum.lighter.lighter_routing import (
    LIGHTER_DEPOSIT_GAS_LIMIT,
    estimate_lighter_gas_limit,
)

#: Gas used by a LIX L1 withdrawal.
MEASURED_GAS = 156_655
FALLBACK_GAS_LIMIT = 500_000
EXTRA_GNOSIS_GAS = 500_000


def test_gas_limit_follows_the_estimate() -> None:
    """A successful estimate replaces the fixed limit, with its margin.

    1. Estimate a call that needs far less than the fixed limit.
    2. Estimate a call that needs more than the fixed limit.
    3. Verify the limits follow the estimates in both directions.
    """
    # 1. Estimate a call that needs far less than the fixed limit.
    builder = SimpleNamespace(estimate_gas=lambda function: MEASURED_GAS, extra_gnosis_gas=EXTRA_GNOSIS_GAS)
    limit = estimate_lighter_gas_limit(builder, SimpleNamespace(fn_name="withdraw"), FALLBACK_GAS_LIMIT)

    # 2. Estimate a call that needs more than the fixed limit.
    expensive = SimpleNamespace(estimate_gas=lambda function: 2_000_000, extra_gnosis_gas=EXTRA_GNOSIS_GAS)
    expensive_limit = estimate_lighter_gas_limit(expensive, SimpleNamespace(fn_name="withdraw"), FALLBACK_GAS_LIMIT)

    # 3. Verify the limits follow the estimates in both directions.
    assert limit == 234_983
    assert expensive_limit == 3_000_000


def test_failed_estimate_falls_back_to_the_fixed_limit() -> None:
    """A failed estimate cannot stop a custody movement.

    1. Make estimation fail, as an RPC error would.
    2. Verify the old total limit is used.
    """
    # 1. Make estimation fail, as an RPC error would.
    def fail(function: ContractFunction) -> int:
        raise ValueError("execution reverted")

    builder = SimpleNamespace(estimate_gas=fail, extra_gnosis_gas=EXTRA_GNOSIS_GAS)

    # 2. Verify the old total limit is used.
    limit = estimate_lighter_gas_limit(builder, SimpleNamespace(fn_name="withdraw"), FALLBACK_GAS_LIMIT)
    assert limit == FALLBACK_GAS_LIMIT + EXTRA_GNOSIS_GAS


@pytest.mark.parametrize(
    ("kwargs", "expected_gas"),
    [
        ({"total_gas_limit": LIGHTER_DEPOSIT_GAS_LIMIT}, LIGHTER_DEPOSIT_GAS_LIMIT),
        ({"gas_limit": 100_000}, 100_000 + EXTRA_GNOSIS_GAS),
    ],
)
def test_lagoon_builder_signs_the_requested_limit(
    mocker: MockerFixture,
    kwargs: dict[str, int],
    expected_gas: int,
) -> None:
    """A total limit replaces the Safe allowance, and other callers keep it.

    The chain objects are mocked, because this tests what the builder asks to sign.

    1. Create a builder around a fake vault and hot wallet.
    2. Sign a transaction.
    3. Verify the gas sent to the chain and saved in the state.
    """
    # 1. Create a builder around a fake vault and hot wallet.
    mocker.patch.object(tx_module, "encode_pickle_over_json", return_value="")
    bound_call = mocker.Mock()
    bound_call.build_transaction.side_effect = lambda params: dict(params)
    vault = mocker.Mock(trading_strategy_module_address="0x" + "11" * 20)
    vault.web3.eth.chain_id = 1
    vault.transact_via_trading_strategy_module.return_value = bound_call
    hot_wallet = mocker.Mock(address="0x" + "22" * 20)
    hot_wallet.sign_transaction_with_new_nonce.return_value = SimpleNamespace(rawTransaction=b"\x01", hash=b"\x02", nonce=1)
    builder = LagoonTransactionBuilder(vault, hot_wallet, extra_gnosis_gas=EXTRA_GNOSIS_GAS)
    gas_price = mocker.Mock()
    gas_price.get_tx_gas_params.return_value = {"maxFeePerGas": 1, "maxPriorityFeePerGas": 1}

    # 2. Sign a transaction.
    function = mocker.Mock(spec=ContractFunction, address="0x" + "33" * 20, fn_name="deposit", args=[])
    tx = builder.sign_transaction(
        mocker.Mock(spec=Contract, address=function.address),
        function,
        gas_price_suggestion=gas_price,
        asset_deltas=[],
        **kwargs,
    )

    # 3. Verify the gas sent to the chain and saved in the state.
    assert bound_call.build_transaction.call_args.args[0]["gas"] == expected_gas
    assert tx.details["gas"] == expected_gas
