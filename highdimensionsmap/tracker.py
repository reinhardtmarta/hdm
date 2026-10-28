import numpy as np
from scipy.signal import savgol_filter

# Dentro da classe RobustMotionTracker no arquivo tracker.py:
def track(self, trajectory_sequence):
    trajectory_sequence = np.asarray(trajectory_sequence, dtype=np.float64)
    n_passos = trajectory_sequence.shape[0]
    signatures = self.scanner.transform(trajectory_sequence)
    
    if signatures.ndim == 1:
        signatures = signatures.reshape(1, -1)

    # Suavização para derivada limpa
    window_length = min(7, n_passos if n_passos % 2 != 0 else n_passos - 1)
    if window_length >= 3:
        smooth_signatures = savgol_filter(signatures, window_length=window_length, polyorder=2, axis=0)
    else:
        smooth_signatures = signatures

    velocities = np.linalg.norm(np.diff(smooth_signatures, axis=0), axis=1) if n_passos > 1 else np.array([0.0])
    
    if n_passos > 2:
        accelerations = np.linalg.norm(np.diff(smooth_signatures, n=2, axis=0), axis=1)
        curvature = np.pad(accelerations, (1, 1), mode="edge")
    else:
        curvature = np.zeros(n_passos)

    total_instability = curvature
    structural_coherence = 1.0 / (1.0 + total_instability)

    return {
        "signatures": signatures,
        "velocity": velocities,
        "curvature": curvature,
        "structural_coherence": structural_coherence,
        "is_coherent": bool(np.median(structural_coherence) > 0.4),
    }
    
