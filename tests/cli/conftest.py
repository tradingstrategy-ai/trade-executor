import os

import pytest


@pytest.fixture(scope="session")
def unit_test_cache_path():
    """Where unit tests  cache files.

    We have special path for CLI tests to make sure CLI tests
    always do fresh downloads.
    """
    path = os.path.join(os.path.dirname(__file__), "/tmp/cli_tests")
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture()
def anvil() -> AnvilLaunch:
    anvil = launch_anvil()
    try:
        yield anvil
    finally:
        anvil.close()


@pytest.fixture()
def web3(anvil: AnvilLaunch) -> Web3:
    """Set up the Anvil Web3 connection.

    Also perform the Anvil state reset for each test.
    """
    web3 = Web3(HTTPProvider(anvil.json_rpc_url, request_kwargs={"timeout": 2}))

    # Get rid of attributeddict slow down
    web3.middleware_onion.clear()

    install_chain_middleware(web3)

    return web3


@pytest.fixture()
def deployer(web3) -> HexAddress:
    """Deployer account.

    - This account will deploy all smart contracts

    - Starts with 10,000 ETH
    """
    return web3.eth.accounts[0]


@pytest.fixture()
def uniswap_v2(web3: Web3, deployer: HexAddress) -> UniswapV2Deployment:
    """Deploy Uniswap, WETH token."""
    assert web3.eth.get_balance(deployer) > 0
    deployment = deploy_uniswap_v2_like(web3, deployer, give_weth=500)  # Will also deploy WETH9 and give the deployer this many WETH tokens
    logger.info("Uni v2 factory deployed at %s", deployment.factory.address)
    return deployment


@pytest.fixture()
def user_1(web3) -> HexAddress:
    """User account.

    Do some account allocation for tests.
    """
    return web3.eth.accounts[1]


# WETH
@pytest.fixture
def weth(uniswap_v2) -> Contract:
    return uniswap_v2.weth


@pytest.fixture()
def weth_asset(weth: Contract) -> AssetIdentifier:
    """WETH as a persistent id.
    """
    details = fetch_erc20_details(weth.w3, weth.address)
    return translate_token_details(details)

# USDC
@pytest.fixture()
def usdc(web3, deployer) -> Contract:
    """Mock USDC token.

    All initial $100M goes to `deployer`
    """
    token = create_token(web3, deployer, "USD Coin", "USDC", 100_000_000 * 10**6, decimals=6)
    return token


@pytest.fixture()
def usdc_asset(usdc: Contract) -> AssetIdentifier:
    """USDC as a persistent id.
    """
    details = fetch_erc20_details(usdc.w3, usdc.address)
    return translate_token_details(details)


@pytest.fixture()
def weth_usdc_uniswap_pair(web3, deployer, uniswap_v2, usdc, weth) -> HexAddress:
    """Create Uniswap v2 pool for WETH-USDC.

    - Add 200k initial liquidity at 1600 ETH/USDC
    """

    deposit = 200_000  # USDC
    price = 1600

    pair = deploy_trading_pair(
        web3,
        deployer,
        uniswap_v2,
        usdc,
        weth,
        deposit * 10**6,
        (deposit // price) * 10**18,
    )

    logger.info("%s-%s pair is at %s", weth.address, usdc.address, pair)

    return pair


@pytest.fixture
def weth_usdc_trading_pair(uniswap_v2, weth_usdc_uniswap_pair, usdc_asset, weth_asset) -> TradingPairIdentifier:
    return TradingPairIdentifier(weth_asset, usdc_asset, weth_usdc_uniswap_pair, uniswap_v2.factory.address, fee=0.0030)


@pytest.fixture
def hot_wallet(web3, deployer, user_1, usdc: Contract) -> HotWallet:
    """Create hot wallet for the signing tests.

    Top is up with some gas money and 500 USDC.
    """
    private_key = HexBytes(secrets.token_bytes(32))
    account = Account.from_key(private_key)
    wallet = HotWallet(account)
    wallet.sync_nonce(web3)
    tx_hash = web3.eth.send_transaction({"to": wallet.address, "from": user_1, "value": 15 * 10**18})
    assert_transaction_success_with_explanation(web3, tx_hash)
    tx_hash = usdc.functions.transfer(wallet.address, 500 * 10**6).transact({"from": deployer})
    assert_transaction_success_with_explanation(web3, tx_hash)
    return wallet


@pytest.fixture()
def pair_universe(web3, weth_usdc_trading_pair) -> PandasPairUniverse:
    return create_pair_universe(web3, None, [weth_usdc_trading_pair])

