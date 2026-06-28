"""Shared test configuration for the P-Ork service test suite.

Stubs out optional/heavy native dependencies that aren't installed in the
standard dev environment so executor modules can be imported freely in tests.
The real modules are only needed when actually connecting to OpenClaw — all
OpenClaw-related tests should be integration tests in a fully provisioned env.
"""
import sys
from unittest.mock import MagicMock

for _mod in [
    "cryptography",
    "cryptography.hazmat",
    "cryptography.hazmat.primitives",
    "cryptography.hazmat.primitives.serialization",
    "websockets",
    "websockets.legacy",
    "websockets.legacy.client",
]:
    sys.modules.setdefault(_mod, MagicMock())
