"""Compatibility entry point for team_wtm.analysis.categorical."""

import sys

from team_wtm.analysis import categorical as _implementation
from team_wtm.analysis.categorical import *

sys.modules[__name__] = _implementation
