"""Puget — a minimal coding agent powered by Ollama.

Central configuration helpers live here so every module resolves
paths and settings from one place.

Environment variables:
  PUGET_HOME          Base directory for puget data (default: ~/.puget).
  PUGET_OLLAMA_MODEL  Ollama model name (default: minimax-m2.7:cloud).
  PUGET_OLLAMA_HOST   Ollama server URL (default: whatever ollama-python
                      uses, typically http://localhost:11434). Overrides
                      the standard OLLAMA_HOST if both are set.
  PUGET_DB            Override the database path directly. Takes
                      precedence over PUGET_HOME for the DB location.
  PUGET_SHOW_THINKING If "true", display model thinking/reasoning blocks
                      instead of silently stripping them.
  PUGET_OLLAMA_THINK  Thinking policy for Ollama requests: off, low, on,
                      or auto (default: auto).
"""

import os
from pathlib import Path


def puget_home() -> Path:
    """Return the puget home directory.

    Uses $PUGET_HOME if set, otherwise ~/.puget.
    """
    env = os.environ.get("PUGET_HOME")
    if env:
        return Path(env)
    return Path.home() / ".puget"
