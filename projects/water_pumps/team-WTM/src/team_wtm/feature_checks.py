"""Compatibility entry point for team_wtm.analysis.feature_checks."""

import sys

from team_wtm.analysis import feature_checks as _implementation
from team_wtm.analysis.feature_checks import *

if __name__ == "__main__":
    _implementation.main(sys.argv[1:])
else:
    sys.modules[__name__] = _implementation
