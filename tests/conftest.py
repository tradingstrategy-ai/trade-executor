import os

import pytest


@pytest.fixture()
def strategy_folder():
    """Where unit test strategies are located."""
    return os.path.join(os.path.dirname(__file__), "test_strategies")