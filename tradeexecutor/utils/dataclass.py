"""Dataclass helpers."""
from dataclasses_json import DataClassJsonMixin


class UTCFriendlyDataClassJsonMixin(DataClassJsonMixin):
    """Encode datetimes as iso8601 format to preseve timezones"""



class EthereumAddress(str):
    """Internal type alias for Ethereum addresses."""