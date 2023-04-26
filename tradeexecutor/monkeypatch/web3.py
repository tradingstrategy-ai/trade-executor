#
# https://github.com/ethereum/web3.py/issues/2936
#

from functools import (
    singledispatch,
)
import operator
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Iterable,
    Tuple,
    TypeVar,
    Union,
)

from eth_account import (
    Account,
)
from eth_account.signers.local import (
    LocalAccount,
)
from eth_keys.datatypes import (
    PrivateKey,
)
from eth_typing import (
    ChecksumAddress,
    HexStr,
)
from eth_utils import (
    to_dict,
)
from eth_utils.curried import (
    apply_formatter_if,
)
from eth_utils.toolz import (
    compose,
)

from web3._utils.method_formatters import (
    STANDARD_NORMALIZERS,
)
from web3._utils.rpc_abi import (
    TRANSACTION_PARAMS_ABIS,
    apply_abi_formatters_to_dict,
)
from web3._utils.transactions import (
    fill_nonce,
    fill_transaction_defaults,
)
from web3.middleware.signing import gen_normalized_accounts, format_transaction
from web3.types import (
    Middleware,
    RPCEndpoint,
    RPCResponse,
    TxParams,
)


def construct_sign_and_send_raw_middleware(
    private_key_or_account,
) -> Middleware:
    """Capture transactions sign and send as raw transactions


    Keyword arguments:
    private_key_or_account -- A single private key or a tuple,
    list or set of private keys. Keys can be any of the following formats:
      - An eth_account.LocalAccount object
      - An eth_keys.PrivateKey object
      - A raw private key as a hex string or byte string
    """

    accounts = gen_normalized_accounts(private_key_or_account)

    def sign_and_send_raw_middleware(
        make_request: Callable[[RPCEndpoint, Any], Any], w3: "Web3"
    ) -> Callable[[RPCEndpoint, Any], RPCResponse]:
        format_and_fill_tx = compose(
            format_transaction, fill_transaction_defaults(w3), fill_nonce(w3)
        )

        def middleware(method: RPCEndpoint, params: Any) -> RPCResponse:
            if method != "eth_sendTransaction":
                return make_request(method, params)
            else:
                transaction = format_and_fill_tx(params[0])

            if "from" not in transaction:
                return make_request(method, params)
            elif transaction.get("from") not in accounts:
                return make_request(method, params)

            account = accounts[transaction["from"]]
            raw_tx = account.sign_transaction(transaction).rawTransaction
            return make_request(RPCEndpoint("eth_sendRawTransaction"), [raw_tx.hex()])

        return middleware

    return sign_and_send_raw_middleware
