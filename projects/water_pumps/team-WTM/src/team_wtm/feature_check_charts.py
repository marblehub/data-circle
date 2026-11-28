"""Compatibility entry point for team_wtm.analysis.charts."""

import sys

from team_wtm.analysis import charts as _implementation
from team_wtm.analysis.charts import *

sys.modules[__name__] = _implementation
