"""Strategy-file deployment information persisted in the executor state."""

import datetime
from dataclasses import dataclass, field

from dataclasses_json import dataclass_json

from eth_defi.compat import native_datetime_utc_now


@dataclass_json
@dataclass(slots=True)
class DeploymentInfoChange:
    """Audit entry for deployment information changes."""

    #: When deployment information was changed in the state.
    modified_at: datetime.datetime = field(default_factory=native_datetime_utc_now)

    #: Human-readable change summary.
    change_summary_message: str = ""


@dataclass_json
@dataclass(slots=True)
class DeploymentInfo:
    """Strategy-file based deployment information.

    This is separate from legacy ``sync.deployment`` data, which tracks on-chain
    vault sync state.
    """

    #: Deployment source. Currently only strategy-file multichain artifacts use this.
    source: str | None = None

    #: Deployment artifact path read during startup.
    deployment_file: str | None = None

    #: Machine-readable deployment data from the deployment artifact.
    data: dict = field(default_factory=dict)

    #: Deployment information change history.
    modified: list[DeploymentInfoChange] = field(default_factory=list)

    def update_strategy_file_deployment_data(
        self,
        deployment_file: str,
        data: dict,
    ) -> bool:
        """Update strategy-file deployment data if it changed.

        :return:
            ``True`` if data changed and an audit entry was added.
        """

        self.source = "strategy_file"
        self.deployment_file = deployment_file

        if self.data == data:
            return False

        if self.data:
            change_summary_message = f"Updated strategy-file deployment information from {deployment_file}"
        else:
            change_summary_message = f"Initialised strategy-file deployment information from {deployment_file}"

        self.data = data
        self.modified.append(DeploymentInfoChange(change_summary_message=change_summary_message))
        return True
