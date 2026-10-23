#!/usr/bin/env python3
"""Entry point for VLM-VAD Studio."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from webapp.app import main


if __name__ == "__main__":
    main()
