from .scanner import HDMScanner
from .kinematics import HDMKinematics
from .tracker import RobustMotionTracker, RobustMotionTracker as MotionNoiseTracker
from .core import hdm
from .api import app
from .tensor import TrajectoryMotionTensor
from .compressor import HDMCompressor

__version__ = "0.2.2"  # Incremento de versão minor por adição de funcionalidade
__all__ = [
    "HDMScanner", 
    "HDMKinematics", 
    "MotionNoiseTracker", 
    "RobustMotionTracker", 
    "hdm", 
    "app", 
    "TrajectoryMotionTensor",
    "HDMCompressor"
]
