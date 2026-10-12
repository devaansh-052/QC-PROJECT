"""Fraud detection research infrastructure. No quantum advantage is assumed."""

import sys

# sklearn 1.4.2's Windows wheel preloads its bundled C++ runtime. If that occurs
# before torch 2.10.0 initializes, c10.dll fails with WinError 1114 on the verified
# host. Initialize Torch first for every project entry point. Custom Windows
# clients must likewise import this package (or torch) before sklearn/XGBoost.
if sys.platform == "win32":
    import torch  # noqa: F401

__version__ = "0.1.0"
