#
# Re-export construct_sign_and_send_raw_middleware from eth_defi.compat
# which handles web3.py 7.x SignAndSendRawMiddlewareBuilder.
#
# Previously this module contained a custom implementation that worked
# around https://github.com/ethereum/web3.py/issues/2936 but that
# approach is incompatible with web3.py 7.x which removed the
# function-based Middleware type from web3.types.
#

from eth_defi.compat import construct_sign_and_send_raw_middleware

__all__ = ["construct_sign_and_send_raw_middleware"]
