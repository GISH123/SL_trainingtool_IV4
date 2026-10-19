from __future__ import annotations

import compileall
import sys
import unittest
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    ok = compileall.compile_dir(str(root), quiet=1)
    suite = unittest.defaultTestLoader.discover(str(root / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if ok and result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
