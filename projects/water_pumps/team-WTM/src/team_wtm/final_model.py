"""Compatibility entry point for team_wtm.modeling.training."""

import sys

from team_wtm.modeling import training as _implementation
from team_wtm.modeling.training import *

if __name__ == "__main__":
    _implementation.main(sys.argv[1:])
else:
    sys.modules[__name__] = _implementation
