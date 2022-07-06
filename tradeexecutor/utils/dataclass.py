"""Dataclass helpers.

Help

- Describing blockchain data in Python `dataclass`

- Serialise this data with :py:mod:`dataclasses_json`
"""

from dataclasses_json import DataClassJsonMixin


class UTCFriendlyDataClassJsonMixin(DataClassJsonMixin):
    """Encode datetimes as iso8601 format to preseve timezones"""



class EthereumAddress(str):
    """Internal type alias for Ethereum addresses."""