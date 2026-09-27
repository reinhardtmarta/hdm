import numpy as np
from numpy.polynomial import chebyshev
from scipy.special import expit


class RobustMotionTracker:
    """
    Rastreador dimensional cinemático determinístico.
    Elimina geração de ruído pseudoaleatório (Monte Carlo) e avalia a coerência
    estutural por meio de curvatura analítica e derivadas de ordem superior.
    """

    def __init__(self, scanner, **kwargs):
        self.scanner = scanner

    def track(self, trajectory_sequence):
        trajectory_sequence = np.asarray(trajectory_sequence, dtype=np.float64)
        if trajectory_sequence.ndim != 2:
            raise ValueError("trajectory_sequence precisa ser uma matriz 2D (passos, dimensão).")

        n_passos, input_dim = trajectory_sequence.shape
        signatures = self.scanner.transform(trajectory_sequence)
        
        if signatures.ndim == 1:
            signatures = signatures.reshape(1, -1)

        # 1. Velocidade esférica (taxa de variação angular no espaço projetado)
        if n_passos > 1:
            velocities = np.linalg.norm(np.diff(signatures, axis=0), axis=1)
        else:
            velocities = np.array([0.0], dtype=np.float64)

        # 2. Aceleração / Curvatura discreta (segunda derivada)
        if n_passos > 2:
            accelerations = np.linalg.norm(np.diff(signatures, n=2, axis=0), axis=1)
            # Alinha o tamanho preenchendo as bordas com continuidade
            curvature = np.pad(accelerations, (1, 1), mode="edge")
        else:
            curvature = np.zeros(n_passos, dtype=np.float64)

        # 3. Tendência analítica via polinômios ortogonais de Chebyshev
        # Evita a oscilação de Runge do polyfit convencional em séries curtas
        deg = min(2, max(1, n_passos - 2)) if n_passos >= 3 else 1
        t = np.linspace(-1.0, 1.0, n_passos)
        
        # Ajuste ortogonal coluna a coluna
        trend = np.zeros_like(signatures)
        for col in range(signatures.shape[1]):
            c = chebyshev.chebfit(t, signatures[:, col], deg=deg)
            trend[:, col] = chebyshev.chebval(t, c)

        # 4. Resíduo de dispersão local
        raw_noise = np.linalg.norm(signatures - trend, axis=1)

        # 5. Escala analítica invariante (dispensa simulação de Monte Carlo com rng)
        # A instabilidade estrutural combina dispersão do polinômio e curvatura abrupta
        total_instability = raw_noise + 0.5 * curvature

        # Coerência estrutural contínua no intervalo (0, 1]
        structural_coherence = 1.0 / (1.0 + total_instability)

        return {
            "signatures": signatures,
            "velocity": velocities,
            "raw_noise": raw_noise,
            "curvature": curvature,
            "structural_coherence": structural_coherence,
            "is_coherent": bool(np.mean(structural_coherence) > 0.5),
        }
        
