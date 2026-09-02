"""Put `scripts/` on the import path so the tests import `benchmark` as a package.

The benchmark harness lives outside `apps/api`, has no packaging of its own and needs
none — it is run as a script from the repository root. Three lines here are the whole
cost of that, and they keep the harness free of a `pyproject.toml` that would exist
only to satisfy an import.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
