"""Compatibility entry point for team_wtm.evaluation.comparison."""

import sys

from team_wtm.evaluation import comparison as _implementation
from team_wtm.evaluation.comparison import *

if __name__ == "__main__":
    _implementation.main(sys.argv[1:])
else:
    sys.modules[__name__] = _implementation
