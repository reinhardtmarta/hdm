import numpy as np

class TrajectoryMotionTensor:
    """
    Tensor de Trajetória no Espaço de Estados para Alta Dimensão.
    Mede a coerência angular contínua e detecta anomalias dinâmicas.
    """
    def __init__(self, scanner=None):
        self.scanner = scanner

    def compute_coherence(self, trajectory_matrix):
        """
        Calcula a coerência escalar de transição e o perfil instantâneo.
        """
        matriz = np.asarray(trajectory_matrix, dtype=np.float64)
        if matriz.ndim != 2:
            raise ValueError("A matriz de trajetória precisa ter 2 dimensões (N_passos, Dimensão).")
            
        x0 = matriz[:-1]
        x1 = matriz[1:]
        
        # Produto escalar (alinhamento direcional entre passos adjacentes)
        dot_products = np.sum(x0 * x1, axis=1)
        normas = np.linalg.norm(x0, axis=1) * np.linalg.norm(x1, axis=1) + 1e-12
        
        # Coerência angular instantânea passo a passo
        instantaneous_coherence = dot_products / normas
        
        # Instabilidade residual (desvio da trajetória contínua)
        drift_instability = 1.0 - instantaneous_coherence
        
        return {
            "global_coherence": float(np.mean(instantaneous_coherence)),
            "instantaneous_profile": instantaneous_coherence,
            "drift_instability": drift_instability,
            "is_structured": bool(np.mean(instantaneous_coherence) > 0.5)
        }
      
