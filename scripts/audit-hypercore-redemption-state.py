"""Backwards-compatible wrapper for the generic redemption audit CLI."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


def main() -> int:
    """Forward the legacy script entry point to the generic audit CLI."""
    target = Path(__file__).with_name("audit-redemption-state.py")
    spec = spec_from_file_location("audit_redemption_state_cli", target)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main()


if __name__ == "__main__":
    sys.exit(main())
