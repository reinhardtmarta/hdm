import numpy as np

class TrajectoryMotionTensor:
    """
    Tensor de Trajetória no Espaço de Estados para Alta Dimensão.
    Mede a coerência angular contínua e detecta anomalias dinâmicas.
    """
    def __init__(self, scanner=None):
        self.scanner = scanner

    def compute_coherence(self, trajectory_matrix, step_lag=1):
        matriz = np.asarray(trajectory_matrix, dtype=np.float64)
        if matriz.ndim != 2:
            raise ValueError("A matriz de trajetória precisa ter 2 dimensões (N_passos, Dimensão).")
            
        if matriz.shape[0] <= step_lag:
            return {
                "global_coherence": 0.0,
                "instantaneous_profile": np.array([]),
                "drift_instability": np.array([]),
                "is_structured": False
            }

        x0 = matriz[:-step_lag]
        x1 = matriz[step_lag:]
        
        # Produto escalar normalizado
        dot_products = np.sum(x0 * x1, axis=1)
        normas = np.linalg.norm(x0, axis=1) * np.linalg.norm(x1, axis=1)
        
        # Evita divisão por zero/valores insignificantes
        valid_mask = normas > 1e-8
        instantaneous_coherence = np.ones(len(dot_products))
        instantaneous_coherence[valid_mask] = dot_products[valid_mask] / normas[valid_mask]
        
        # Garante intervalo [-1.0, 1.0]
        instantaneous_coherence = np.clip(instantaneous_coherence, -1.0, 1.0)
        drift_instability = 1.0 - instantaneous_coherence
        
        global_coherence = float(np.median(instantaneous_coherence)) # Uso de mediana para robustez a outliers
        
        return {
            "global_coherence": global_coherence,
            "instantaneous_profile": instantaneous_coherence,
            "drift_instability": drift_instability,
            "is_structured": bool(global_coherence > 0.3) # Limiar adaptativo
        }
        
