"""Compatibility entry point for team_wtm.data.cleaning."""

import sys

from team_wtm.data import cleaning as _implementation
from team_wtm.data.cleaning import *

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
