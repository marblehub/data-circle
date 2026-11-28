"""Compatibility entry point for team_wtm.data.export."""

import sys

from team_wtm.data import export as _implementation
from team_wtm.data.export import *

if __name__ == "__main__":
    _implementation.main(sys.argv[1:])
else:
    sys.modules[__name__] = _implementation
