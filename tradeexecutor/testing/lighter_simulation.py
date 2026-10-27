"""Simulated Lighter environment: an Ethereum fork with a fake sequencer.

A tutorial or test that needs a Lighter vault without real money forks
Ethereum mainnet at a fixed block with Anvil and replaces Lighter's
sequencer with :py:class:`~tradeexecutor.testing.lighter_core.FakeLighterCore`.
Every Safe, guard, Lagoon and ZkLighter transaction still runs on the fork;
only account creation, L2 transfers, equity reads and withdrawal proofs come
from the fake.
"""

import logging
from contextlib import ExitStack
from dataclasses import dataclass, field
from decimal import Decimal
from unittest import mock

from eth_defi.hotwallet import HotWallet
from eth_defi.lighter.constants import LIGHTER_USDC_ETHEREUM
from eth_defi.provider.anvil import AnvilLaunch, fork_network_anvil
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.testing.fork_blocks import ETHEREUM_MIDNIGHT_BLOCK
from eth_defi.token import USDC_WHALE, TokenDetails, fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation
from web3 import Web3

from tradeexecutor.testing.lighter_core import FakeLighterCore

logger = logging.getLogger(__name__)

#: Public deterministic Anvil account zero key used as the simulated deployer; never a production secret.
ANVIL_DEPLOYER_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

#: Master account index the fake sequencer assigns.
SIMULATED_MASTER_INDEX = 127

#: ETH given to the simulated deployer for gas.
SIMULATED_DEPLOYER_ETH = 10


@dataclass(slots=True)
class SimulatedLighter:
    """A running Ethereum fork with the fake Lighter sequencer installed."""

    #: Fork connection.
    web3: Web3 = field(repr=False)
    #: Fork RPC URL for the executor commands.
    rpc_url: str = field(repr=False)
    #: Native Ethereum USDC on the fork.
    usdc: TokenDetails
    #: Funded deployer wallet, excluded from repr because it holds a key.
    deployer: HotWallet = field(repr=False)
    #: Fake sequencer behind every Lighter seam.
    fake_core: FakeLighterCore
    #: Anvil process, when this object launched it.
    anvil: AnvilLaunch | None
    #: Installed fake sequencer patches.
    patches: ExitStack

    def close(self) -> None:
        """Remove the patches and stop a fork this object launched."""
        self.patches.close()
        if self.anvil is not None:
            self.anvil.close()
            self.anvil = None


def launch_simulated_lighter(
    upstream_rpc_url: str,
    *,
    deployer_usdc: Decimal,
    existing_rpc_url: str | None = None,
    transfer_fee_usdc: Decimal = Decimal(0),
) -> SimulatedLighter:
    """Fork Ethereum, fund the Anvil deployer and install the fake sequencer.

    :param upstream_rpc_url:
        Ethereum RPC to fork at :py:data:`ETHEREUM_MIDNIGHT_BLOCK`.
    :param deployer_usdc:
        USDC moved from a whale to the deployer.
    :param existing_rpc_url:
        An already running fork to reuse instead of launching one; it must
        have the USDC whale unlocked.
    :param transfer_fee_usdc:
        Fee the fake sequencer charges per transfer.
    :return:
        The running environment; call :py:meth:`SimulatedLighter.close` when done.
    """
    anvil = None
    if existing_rpc_url:
        rpc_url = existing_rpc_url
    else:
        logger.info("Forking Ethereum at block %d", ETHEREUM_MIDNIGHT_BLOCK)
        anvil = fork_network_anvil(upstream_rpc_url, fork_block_number=ETHEREUM_MIDNIGHT_BLOCK, unlocked_addresses=[USDC_WHALE[1]])
        rpc_url = anvil.json_rpc_url
    web3 = create_multi_provider_web3(rpc_url, default_http_timeout=(3.0, 250.0))
    deployer = HotWallet.from_private_key(ANVIL_DEPLOYER_PRIVATE_KEY)
    usdc = fetch_erc20_details(web3, LIGHTER_USDC_ETHEREUM)
    web3.provider.make_request("anvil_setBalance", [deployer.address, hex(SIMULATED_DEPLOYER_ETH * 10**18)])
    funding = usdc.contract.functions.transfer(deployer.address, usdc.convert_to_raw(deployer_usdc)).transact({"from": USDC_WHALE[1]})
    assert_transaction_success_with_explanation(web3, funding)

    fake_core = FakeLighterCore(web3, SIMULATED_MASTER_INDEX, transfer_fee=transfer_fee_usdc)
    patches = ExitStack()
    for target, replacement in fake_core.get_patches():
        patches.enter_context(mock.patch(target, new=replacement))
    return SimulatedLighter(web3=web3, rpc_url=rpc_url, usdc=usdc, deployer=deployer, fake_core=fake_core, anvil=anvil, patches=patches)
