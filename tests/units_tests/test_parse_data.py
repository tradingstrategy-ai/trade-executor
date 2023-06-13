from tradeexecutor.cli.commands.perform_test_trade import parse_pair_data


def test_parse_data_with_fee():
    """Test that the pair data with a fee is parsed correctly from string to tuple."""
    s = '(ChainId.ethereum, "uniswap-v2", "WETH", "USDC", 0.003)'
    data = parse_pair_data(s)
    assert data == (1, 'uniswap-v2', 'WETH', 'USDC', 0.003)


def test_parse_data_without_fee():
    """Test that the pair data without a fee is parsed correctly from string to tuple."""
    s = '(ChainId.ethereum, "uniswap-v2", "WETH", "USDC")'
    data = parse_pair_data(s)
    assert data == (1, 'uniswap-v2', 'WETH', 'USDC', None)