"""Entry point for ``python -m gui`` (PYTHONPATH=src) from the repository root."""

import sys

from .app import main

if __name__ == "__main__":
    sys.exit(main())
