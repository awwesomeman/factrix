"""Statistical tooling shared across the library.

``multiple_testing`` — BHY procedure for FDR control across many factors.
"""

from factrix.stats.multiple_testing import bhy_adjust, bhy_adjusted_p

__all__ = ["bhy_adjust", "bhy_adjusted_p"]
