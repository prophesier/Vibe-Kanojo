"""Thin wrapper around ``python -m src.open_llm_vtuber.discord_bot``.

Provided for parity with ``scripts/run_bilibili_live.py``. Either invocation
works — pick whichever fits your launcher.
"""

import os
import sys

# Make `src.*` importable when run directly.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.open_llm_vtuber.discord_bot.__main__ import main  # noqa: E402


if __name__ == "__main__":
    main()
