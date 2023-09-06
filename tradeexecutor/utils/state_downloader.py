import requests

from tradeexecutor.state.state import State


def download_state(url: str) -> State:
    """Get a copy of trade-executor state.

    Fetch full state from a remote trade-executor.

    To be used in notebooks.
    """

    resp = requests.get(f"{url}/state")

    if len(resp.content) == 0:
        raise RuntimeError(f"Could not download: {url}")

    try:
        state = State.read_json_blob(resp.text)
    except Exception as e:
        raise RuntimeError(f"Could not decode: {url}") from e

    print(f"Downloaded state for {url}, total {len(resp.content):,} chars")

    return state