"""Test configuration and shared fixtures."""

import os
import sys

# Make the vectordb package importable when running pytest from the
# project root (the package lives in ./py/vectordb/).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
