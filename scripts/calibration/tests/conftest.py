"""Put `scripts/` on the import path so the tests import `calibration` as a package.

The same three lines the benchmark harness's tests use, for the same reason: this tooling
runs as a script from the repository root and needs no packaging of its own.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
