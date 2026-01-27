"""Learning layer for RFSN Controller.

This package contains SAFE learning components that operate
in PROPOSAL SPACE ONLY. Learning can influence:
- Which model to call for which failure class
- Which plan template to start from
- Which files to inspect first
- Which patch strategy to try first

Learning CANNOT influence:
- Gates/allowlists
- Budgets/timeouts
- Sandbox constraints
- Verifier behavior
"""

from .fingerprint import (
    FailureFingerprint,
    fingerprint_failure,
    compute_fingerprint_hash,
)
from .strategy_bandit import (
    StrategyBandit,
    StrategyStats,
)
from .quarantine import (
    QuarantineLane,
    QuarantineConfig,
    is_quarantined,
)
from .learned_strategy_selector import (
    LearnedStrategySelector,
    StrategyRecommendation,
)

__all__ = [
    # Fingerprinting
    "FailureFingerprint",
    "fingerprint_failure",
    "compute_fingerprint_hash",
    # Bandit
    "StrategyBandit",
    "StrategyStats",
    # Quarantine
    "QuarantineLane",
    "QuarantineConfig",
    "is_quarantined",
    # Selector
    "LearnedStrategySelector",
    "StrategyRecommendation",
]
