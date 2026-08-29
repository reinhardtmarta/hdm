"""
HDM (High Dimensions Map)
Um motor leve de projeção geométrica, indexação e rastreamento cinemático em altas dimensões.
"""

from .scanner import HDMScanner
from .kinematics import HDMKinematics

__version__ = "0.1.0"
__all__ = ["HDMScanner", "HDMKinematics"]

