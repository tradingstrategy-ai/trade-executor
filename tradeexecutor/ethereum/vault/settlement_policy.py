"""Small protocol exceptions to the shared vault execution contract.

Keep these policies here rather than teaching routing and retry code about
protocol ticket schemas. This module has no RPC reads or independent state.
"""

from eth_defi.erc_4626.vault_protocol.atoma.deposit_redeem import AtomaRedemptionRequest
from eth_defi.vault.deposit_redeem import RedemptionRequest
from tradingstrategy.chain import ChainId


#: Minimum buffer shared by fast-chain Lagoon snapshots and vault claim receipts.
LAGOON_FAST_CHAIN_CONFIRMATIONS = 16


def get_vault_settlement_confirmations(chain_id: int, provider_delay: int) -> int:
    """Keep vault claim receipts inside the treasury's accounting snapshot.

    Fast chains need Lagoon's existing minimum buffer. Ethereum keeps the
    provider's delay, and all chains retain any stronger provider requirement.
    The existing receipt waiter supplies its normal time-based minimum as well.

    :param chain_id: Chain containing the vault and custody account.
    :param provider_delay: Existing RPC block-tip delay.
    :return: Minimum confirmations; zero means no additional treasury buffer.
    """
    minimum = 0 if chain_id == ChainId.ethereum.value else LAGOON_FAST_CHAIN_CONFIRMATIONS
    return max(minimum, provider_delay)


def validate_single_redemption_ticket(data: dict) -> None:
    """Reject known aggregate or ambiguous tickets before any claim is sent.

    Atoma records request contributions and upgrade ambiguity in its ticket.
    Check the persisted evidence without RPC so an entire batch can be refused
    before an otherwise valid earlier trade moves cash. Other adapters enforce
    attribution through their own request and receipt methods.

    :param data: Serialised manager ticket, including existing state metadata.
    :return: None when the stored ticket satisfies the supported policy.
    :raises AssertionError: Atoma needs reconciliation or has several requests.
    """
    if "atoma_schema" in data:
        assert not data["atoma_reconciliation_required"] and len(data["atoma_parts"]) == 1, (
            "Atoma ticket requires reconciliation or contains an aggregate. "
            "Expected one request per trade; halt and inspect custody/state. "
            f"Vault {data.get('vault_address')}, owner {data.get('vault_owner')}"
        )


def capture_redemption_preflight(request: RedemptionRequest, data: dict) -> None:
    """Retain Atoma's checked implementation across request reconstruction.

    The request can mine after a proxy upgrade. Its parser needs the original
    implementation to distinguish that change from an ordinary mined request.
    Other request types need no extra metadata.

    :param request: Manager request before signing.
    :param data: Existing redemption preflight dictionary to update.
    :return: None; update the existing dictionary in place.
    """
    if isinstance(request, AtomaRedemptionRequest):
        data["implementation"] = request.implementation


def restore_redemption_preflight(request: RedemptionRequest, data: dict) -> None:
    """Restore the implementation checked before the original request broadcast.

    Keep the existing field name so saved Atoma requests retain upgrade checks.
    Legacy records without this evidence keep the manager's existing default.

    :param request: Rebuilt manager request used only for receipt parsing.
    :param data: Previously saved redemption preflight metadata.
    :return: None; restore evidence on the rebuilt request where needed.
    """
    if isinstance(request, AtomaRedemptionRequest) and data.get("implementation") is not None:
        request.implementation = data["implementation"]
