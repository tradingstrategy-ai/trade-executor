"""Revert reason helpers."""


def clean_revert_reason_message(msg: str | None) -> str:
    """Clean up Enzyme's mangling of the revert reason.

    - Clean everything with NULs at start and end based on
      what we have seen Enzyme tx spit out.

    - Non-enzyme revert messages are untouched.

    :param msg:
        Raw revert reason message from JSON-RPC API

    :return:
        Revert reason cleaned up from whatever binary Enzyme inserts there.

        If revert reason is `None` return empty string.
    """

    if not msg:
        return ""

    if "\x13" in msg:
        # '\x13Too little received\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        fmsg = msg[msg.find("\x13") + 1:]
        fmsg = fmsg[:fmsg.rfind("\x00")]
        return fmsg

    return msg