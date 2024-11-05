import pytest
from tradeexecutor.cli.commands.pair_mapping import parse_pair_data


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


def test_parse_data_raises_value_error():
    """Test that the pair data with a bad format is parsed correctly from string to tuple."""
    
    with pytest.raises(ValueError):
        s = '(ChainId.ethereum, "uniswap-v2", "WETH", "USDC", 0.003'
        data = parse_pair_data(s)

    with pytest.raises(ValueError):
        s = '(ChainId.ethereum, "uniswap-v2", "WETH")'
        data = parse_pair_data(s)
    
    with pytest.raises(ValueError):
        s = '(ChainId.ethereum, "uniswap-v2", "WETH", "USDC", 0.003, 0.004)'
        data = parse_pair_data(s)
    