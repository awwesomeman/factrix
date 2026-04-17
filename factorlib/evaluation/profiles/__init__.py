"""Per-factor-type FactorProfile dataclasses (new architecture, Phase A).

Importing this package triggers ``@register_profile`` decorators on the
concrete profile classes, populating ``_PROFILE_REGISTRY``.
"""

from factorlib.evaluation.profiles._base import (
    FactorProfile,
    _PROFILE_REGISTRY,
    get_profile_class,
    register_profile,
)

# Concrete Profile classes import themselves side-effect-style; adding new
# factor types means adding one line here. (Populated in Commit 2.)
# from factorlib.evaluation.profiles.cross_sectional import CrossSectionalProfile  # noqa: F401
# from factorlib.evaluation.profiles.event import EventProfile  # noqa: F401
# from factorlib.evaluation.profiles.macro_panel import MacroPanelProfile  # noqa: F401
# from factorlib.evaluation.profiles.macro_common import MacroCommonProfile  # noqa: F401

__all__ = [
    "FactorProfile",
    "_PROFILE_REGISTRY",
    "get_profile_class",
    "register_profile",
]
