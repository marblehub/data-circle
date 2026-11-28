"""Compatibility entry point for team_wtm.analysis.numeric."""

import sys

from team_wtm.analysis import numeric as _implementation
from team_wtm.analysis.numeric import *

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
