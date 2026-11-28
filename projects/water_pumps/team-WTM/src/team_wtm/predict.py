"""Compatibility entry point for team_wtm.modeling.predict."""

import sys

from team_wtm.modeling import predict as _implementation
from team_wtm.modeling.predict import *

if __name__ == "__main__":
    _implementation.main(sys.argv[1:])
else:
    sys.modules[__name__] = _implementation
