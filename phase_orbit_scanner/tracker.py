import numpy as np

class MotionNoiseTracker:
    """
    Rastreia trajetórias temporais em alta dimensão separando dinâmica de ruído estocástico.
    """
    def __init__(self, scanner):
        self.scanner = scanner

    def track(self, trajectory_sequence, filter_window=3):
        signatures = self.scanner.transform(trajectory_sequence)
        n_passos, n_features = signatures.shape
        
        # Velocidade instantânea espectral
        velocities = np.linalg.norm(np.diff(signatures, axis=0), axis=1)
        
        # Filtro temporal 1D independente por canal
        smoothed = np.zeros_like(signatures)
        pad = filter_window // 2
        for col in range(n_features):
            padded = np.pad(signatures[:, col], pad, mode='edge')
            smoothed[:, col] = np.convolve(padded, np.ones(filter_window)/filter_window, mode='valid')
            
        noise_levels = np.linalg.norm(signatures - smoothed, axis=1)
        
        return {
            "signatures": signatures,
            "velocity": velocities,
            "noise": noise_levels
        }
      
