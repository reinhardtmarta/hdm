import numpy as np
from scipy.signal import savgol_filter


class RobustMotionTracker:
    """Calcula métricas descritivas de movimento em assinaturas do scanner.

    As métricas são indicadores numéricos; não provam, por si só, que uma
    trajetória seja semanticamente coerente ou que o desvio seja ruído.
    """

    def __init__(self, scanner):
        self.scanner = scanner

    def track(self, trajectory_sequence):
        trajectory_sequence = np.asarray(trajectory_sequence, dtype=np.float64)
        if trajectory_sequence.ndim != 2:
            raise ValueError("trajectory_sequence deve ser uma matriz 2D.")
        if trajectory_sequence.shape[0] == 0:
            raise ValueError("trajectory_sequence não pode estar vazia.")
        if not np.isfinite(trajectory_sequence).all():
            raise ValueError("trajectory_sequence deve conter apenas valores finitos.")

        n_steps = trajectory_sequence.shape[0]
        signatures = np.asarray(self.scanner.transform(trajectory_sequence), dtype=np.float64)
        if signatures.ndim == 1:
            signatures = signatures.reshape(1, -1)

        window_length = min(7, n_steps if n_steps % 2 else n_steps - 1)
        if window_length >= 3:
            smooth = savgol_filter(
                signatures, window_length=window_length,
                polyorder=min(2, window_length - 1), axis=0
            )
        else:
            smooth = signatures.copy()

        velocities = (
            np.linalg.norm(np.diff(smooth, axis=0), axis=1)
            if n_steps > 1 else np.array([], dtype=np.float64)
        )
        if n_steps > 2:
            accelerations = np.linalg.norm(np.diff(smooth, n=2, axis=0), axis=1)
            curvature = np.pad(accelerations, (1, 1), mode="edge")
        else:
            curvature = np.zeros(n_steps, dtype=np.float64)

        noise = np.linalg.norm(signatures - smooth, axis=1)
        structural_coherence = 1.0 / (1.0 + curvature)

        return {
            "signatures": signatures,
            "velocity": velocities,
            "curvature": curvature,
            "noise": noise,
            "structural_coherence": structural_coherence,
            "is_coherent": bool(np.median(structural_coherence) > 0.4),
        }


MotionNoiseTracker = RobustMotionTracker
