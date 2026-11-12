"""No-trade band rules for alpha model rebalancing.

A no-trade band suppresses small adjustments to positions that stay open, so a
portfolio does not trade every cycle as weights drift. The band can be measured
against whole-portfolio equity or against each position's own target, and a
strategy can set different rules for the whole portfolio, for a category of
vaults and for a single vault.

The rules here only describe the band. :py:meth:`tradeexecutor.strategy.alpha_model.AlphaModel.apply_no_trade_band`
applies them to a cycle's signals. Example, built inside ``decide_trades()``:

.. code-block:: python

    policy = NoTradeBandPolicy(
        # Portfolio level
        default=NoTradeBandRule(pct=parameters.no_trade_band_pct),
        # Category level: a narrower band for HFT vaults
        categories={"hft": NoTradeBandRule(pct=parameters.hft_no_trade_band_pct)},
        # Vault level: one vault without a band
        pairs={(9999, "0x…"): NoTradeBandRule.disabled()},
        category_function=lambda pair: "hft" if is_hft(pair) else None,
    )
    suppressed = alpha_model.apply_no_trade_band(policy)
"""
import enum
import math
from dataclasses import dataclass, field
from typing import Callable

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.types import Percent, USDollarAmount


class NoTradeBandBasis(enum.Enum):
    """What a no-trade band percentage is measured against."""

    #: Band is a share of whole-portfolio equity: the same dollar width for
    #: every position.
    portfolio_equity = "portfolio_equity"

    #: Band is a share of the signal's own target value, so small positions get
    #: a proportionally small band.
    position_target = "position_target"


class NoTradeBandLevel(enum.Enum):
    """Which part of a :py:class:`NoTradeBandPolicy` supplied the rule."""

    #: The policy's rule for one vault.
    pair = "pair"

    #: The policy's rule for the pair's category.
    category = "category"

    #: The policy's portfolio-wide default rule.
    default = "default"


#: Vault key of a :py:class:`NoTradeBandPolicy` rule: ``(chain_id, lower-case pool address)``.
#:
#: Internal pair ids change between universes, so rules are keyed by the
#: on-chain identity instead, as manual vault overlays are.
NoTradeBandPairKey = tuple[int, str]


def get_no_trade_band_pair_key(pair: TradingPairIdentifier) -> NoTradeBandPairKey:
    """Get the key under which a vault-level rule for this pair is looked up."""
    return int(pair.chain_id), pair.pool_address.lower()


@dataclass(frozen=True, slots=True)
class NoTradeBandRule:
    """How wide the no-trade band is, and which directions it applies to.

    The band width in dollars is ``max(pct * basis, min_usd)``. An adjustment
    whose absolute size is strictly below the band is suppressed.
    """

    #: Band width as a fraction of the basis, e.g. ``0.15`` for 15%.
    #:
    #: Zero disables the rule entirely, whatever ``min_usd`` says.
    pct: Percent

    #: What ``pct`` is measured against.
    basis: NoTradeBandBasis = NoTradeBandBasis.position_target

    #: Never let the band be narrower than this many dollars.
    #:
    #: A per-rule floor. The engine's own rebalance thresholds are one value for
    #: the whole rebalance and cannot differ by category or vault.
    min_usd: USDollarAmount = 0.0

    #: Suppress small top-ups of an existing position.
    apply_to_increases: bool = True

    #: Suppress small reductions of an existing position.
    apply_to_reductions: bool = True

    def __post_init__(self):
        assert isinstance(self.basis, NoTradeBandBasis), f"basis must be NoTradeBandBasis, got {self.basis!r}"
        for name in ("pct", "min_usd"):
            value = getattr(self, name)
            assert isinstance(value, (int, float)) and not isinstance(value, bool), f"{name} must be a number, got {value!r}"
            assert math.isfinite(value) and value >= 0, f"{name} must be finite and not negative, got {value!r}"

    @staticmethod
    def disabled() -> "NoTradeBandRule":
        """A rule that never suppresses anything (``pct=0``)."""
        return NoTradeBandRule(pct=0.0)

    def is_disabled(self) -> bool:
        """Does this rule never suppress anything."""
        return self.pct == 0 or not (self.apply_to_increases or self.apply_to_reductions)

    def applies_to(self, adjust_usd: USDollarAmount) -> bool:
        """Does the rule apply to an adjustment in this direction."""
        if adjust_usd > 0:
            return self.apply_to_increases
        if adjust_usd < 0:
            return self.apply_to_reductions
        return False

    def calculate_band_usd(
        self,
        position_target: USDollarAmount,
        portfolio_equity: USDollarAmount | None,
    ) -> USDollarAmount:
        """Calculate the band width in dollars.

        :param position_target:
            The signal's target value in USD.

        :param portfolio_equity:
            Whole-portfolio equity in USD. Required for the
            ``portfolio_equity`` basis only.
        """
        if self.pct == 0:
            return 0.0
        match self.basis:
            case NoTradeBandBasis.portfolio_equity:
                assert portfolio_equity is not None, "A portfolio_equity basis rule needs portfolio_equity"
                basis_usd = portfolio_equity
            case NoTradeBandBasis.position_target:
                basis_usd = position_target
            case _:
                raise NotImplementedError(f"Unsupported basis {self.basis}")
        return max(self.pct * basis_usd, self.min_usd)


@dataclass(slots=True)
class NoTradeBandPolicy:
    """No-trade band rules at portfolio, category and vault level.

    Built by the strategy in ``decide_trades()``. The most specific rule wins:
    vault, then category, then the portfolio default. A level without a rule
    for the pair is skipped, so a category name without a rule falls through
    to the default.
    """

    #: Portfolio level: the rule for every signal without a more specific one.
    default: NoTradeBandRule

    #: Category level: rule by category name, as returned by ``category_function``.
    categories: dict[str, NoTradeBandRule] = field(default_factory=dict)

    #: Vault level: rule by ``(chain_id, pool address)``.
    #:
    #: Addresses are lower-cased when the policy is created, so the case of a
    #: key cannot cause a silent miss.
    pairs: dict[NoTradeBandPairKey, NoTradeBandRule] = field(default_factory=dict)

    #: Maps a pair to its category name, or ``None`` for no category.
    #:
    #: Supplied by the strategy, because the engine has no single notion of a
    #: vault category.
    category_function: Callable[[TradingPairIdentifier], str | None] | None = None

    def __post_init__(self):
        assert isinstance(self.default, NoTradeBandRule), f"default must be NoTradeBandRule, got {self.default!r}"
        for name, rule in self.categories.items():
            assert isinstance(rule, NoTradeBandRule), f"Category {name!r} rule must be NoTradeBandRule, got {rule!r}"
        if self.categories:
            assert self.category_function is not None, "Category rules need a category_function"

        normalised = {}
        for key, rule in self.pairs.items():
            assert isinstance(key, tuple) and len(key) == 2, f"Pair key must be (chain_id, address), got {key!r}"
            chain_id, address = key
            assert isinstance(address, str) and address.startswith("0x"), f"Pair key address must be a 0x string, got {address!r}"
            assert isinstance(rule, NoTradeBandRule), f"Pair {key!r} rule must be NoTradeBandRule, got {rule!r}"
            normalised_key = (int(chain_id), address.lower())
            assert normalised_key not in normalised, f"Duplicate pair key {normalised_key!r}"
            normalised[normalised_key] = rule
        self.pairs = normalised

    def resolve(self, pair: TradingPairIdentifier) -> tuple[NoTradeBandRule, NoTradeBandLevel]:
        """Return the rule for a pair and the level it came from."""
        rule = self.pairs.get(get_no_trade_band_pair_key(pair))
        if rule is not None:
            return rule, NoTradeBandLevel.pair

        if self.category_function is not None:
            category = self.category_function(pair)
            if category is not None and category in self.categories:
                return self.categories[category], NoTradeBandLevel.category

        return self.default, NoTradeBandLevel.default
