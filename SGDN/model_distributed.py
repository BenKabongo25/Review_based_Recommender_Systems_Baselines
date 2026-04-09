"""Distributed-compatible SGDN model entrypoint.

This module intentionally reuses the mono-GPU SGDN implementation to preserve
model behavior. Training distribution is handled in `main_distributed.py` via
PyTorch DistributedDataParallel (DDP).
"""

from model import SGDN, cal_c_loss  # noqa: F401

__all__ = ["SGDN", "cal_c_loss"]
