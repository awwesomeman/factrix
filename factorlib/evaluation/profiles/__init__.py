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

# Importing the concrete classes is what fires @register_profile. Adding
# a new factor type here is the single place to touch for dispatch wiring.
from factorlib.evaluation.profiles.cross_sectional import CrossSectionalProfile
from factorlib.evaluation.profiles.event import EventProfile
from factorlib.evaluation.profiles.macro_panel import MacroPanelProfile
from factorlib.evaluation.profiles.macro_common import MacroCommonProfile

__all__ = [
    "FactorProfile",
    "_PROFILE_REGISTRY",
    "get_profile_class",
    "register_profile",
    "CrossSectionalProfile",
    "EventProfile",
    "MacroPanelProfile",
    "MacroCommonProfile",
]
