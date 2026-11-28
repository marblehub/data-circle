"""Compatibility entry point for team_wtm.modeling.features."""

import sys

from team_wtm.modeling import features as _implementation
from team_wtm.modeling.features import *

sys.modules[__name__] = _implementation
