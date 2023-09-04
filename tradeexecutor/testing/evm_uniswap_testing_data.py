import json
from tradingstrategy.chain import ChainId


class UniswapV2TestData:
    def __init__(self, version: str, factory: str, router: str, init_code_hash: str, exchange_slug: str, exchange_id: int, chain_id: ChainId):
        assert (
            type(factory)
            == type(router)
            == type(init_code_hash)
            == type(exchange_slug)
            == str
        ), "UniswapV2TestData: all arguments except exchange_id must be strings."
        assert type(exchange_id) == int, "UniswapV2TestData: exchange_id must be int."

        if type(chain_id) == int:
            chain_id = ChainId(chain_id)

        assert version == "V2", "UniswapV2TestData: version must be V2"
        self.version = "V2"
        self.factory = factory
        self.router = router
        self.init_code_hash = init_code_hash
        self.exchange_slug = exchange_slug
        self.exchange_id = exchange_id
        self.chain_id = chain_id


class UniswapV3TestData:
    def __init__(
        self, version: str, factory: str, router: str, position_manager: str, quoter: str, exchange_slug: str, exchange_id: int, chain_id: ChainId
    ):
        assert (
            type(factory)
            == type(router)
            == type(position_manager)
            == type(quoter)
            == type(exchange_slug)
            == str
        ), "UniswapV3TestData: All arguments except exchange_id must be strings."
        assert type(exchange_id) == int, "UniswapV3TestData: exchange_id must be int."
        
        if type(chain_id) == int:
            chain_id = ChainId(chain_id)

        assert version == "V3", "UniswapV3TestData: version must be V3"
        self.version = "V3"
        self.factory = factory
        self.router = router
        self.position_manager = position_manager
        self.quoter = quoter
        self.exchange_slug = exchange_slug
        self.exchange_id = exchange_id
        self.chain_id = chain_id


def serialize_uniswap_test_data(data: UniswapV2TestData | UniswapV3TestData) -> str:
    """Serializes UniswapV2TestData or UniswapV3TestData to JSON string.

    :param data: UniswapV2TestData or UniswapV3TestData

    :returns: JSON string
    """
    return json.dumps({"version": data.version, "data": data.__dict__})


def deserialize_uniswap_test_data(
    json_str: str,
) -> UniswapV2TestData | UniswapV3TestData:
    """Deserializes JSON string to UniswapV2TestData or UniswapV3TestData.

    :param json_str: JSON string

    :returns: UniswapV2TestData or UniswapV3TestData
    """
    json_data = json.loads(json_str)
    version = json_data.get("version")
    data = json_data.get("data")

    if version == "V2":
        return UniswapV2TestData(**data)
    elif version == "V3":
        return UniswapV3TestData(**data)
    else:
        raise ValueError("Unknown version")


def serialize_uniswap_test_data_list(
    data_list: list[UniswapV2TestData | UniswapV3TestData],
) -> str:
    """Serializes list of UniswapV2TestData or UniswapV3TestData to JSON string.

    :param data_list: List of UniswapV2TestData or UniswapV3TestData

    :returns: JSON string
    """
    assert (
        type(data_list) == list
    ), "serialize_uniswap_test_data_list: data_list must be a list"
    serialized_list = []
    for data in data_list:
        assert isinstance(data, UniswapV2TestData | UniswapV3TestData)
        serialized_list.append(serialize_uniswap_test_data(data))
    return json.dumps(serialized_list)


def deserialize_uniswap_test_data_list(
    json_str: str,
) -> list[UniswapV2TestData | UniswapV3TestData]:
    """Deserializes JSON string to list of UniswapV2TestData or UniswapV3TestData.

    :param json_str: JSON string

    :returns: List of UniswapV2TestData or UniswapV3TestData"""
    serialized_list = json.loads(json_str)
    unvalidated = [
        deserialize_uniswap_test_data(serialized_item)
        for serialized_item in serialized_list
    ]
    validated = validate_uniswap_test_data_list(unvalidated)
    return validated


def validate_uniswap_test_data_list(
        test_data: list[UniswapV2TestData | UniswapV3TestData]
):
    """Validates list of UniswapV2TestData or UniswapV3TestData.
    
    :param test_data:
        List of UniswapV2TestData or UniswapV3TestData

    :raises AssertionError: If test_data is not a list of UniswapV2TestData or UniswapV3TestData

    :returns: None
    """
    assert (
        type(test_data) == list
    ), "validate_uniswap_test_data_list: test_data must be a list"
    for data in test_data:
        assert isinstance(data, UniswapV2TestData | UniswapV3TestData)
        assert data.version == "V2" or data.version == "V3"
    return test_data