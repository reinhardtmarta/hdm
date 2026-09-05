import numpy as np
from scipy.special import expit

class RobustMotionTracker:
    """
    Rastreador dimensional com calibração nula e correção assintótica de bordas.
    Pronto para séries temporais curtas (N >= 5) e longas.
    """
    def __init__(self, scanner, n_null_samples=50):
        self.scanner = scanner
        self.n_null_samples = n_null_samples

    def _estimate_null_baseline(self, shape):
        """Gera a linha de base estatística para dados aleatórios equivalentes."""
        # Usa um gerador com semente fixa (derivada da semente do scanner) para que
        # a linha de base nula -- e portanto o resultado de track() -- seja
        # reprodutível entre chamadas com a mesma entrada.
        rng = np.random.default_rng(self.scanner.seed)
        null_noises = []
        for _ in range(self.n_null_samples):
            # Matriz nula aleatória com a mesma dimensão e energia
            null_data = rng.normal(0, 1, size=shape)
            sigs = self.scanner.transform(null_data)
            
            # Tendência polinomial de grau 2 (elimina viés de borda)
            t = np.linspace(-1, 1, shape[0])
            poly = np.polyfit(t, sigs, deg=min(2, shape[0] - 2))
            trend = np.polyval(poly, t.reshape(-1, 1))
            
            res = np.linalg.norm(sigs - trend, axis=1)
            null_noises.append(res)
            
        return np.mean(null_noises, axis=0), np.std(null_noises, axis=0)

    def track(self, trajectory_sequence):
        trajectory_sequence = np.asarray(trajectory_sequence)
        n_passos, input_dim = trajectory_sequence.shape
        signatures = self.scanner.transform(trajectory_sequence)
        
        # 1. Ajuste de trajetória contínua via polinômio ortogonal
        t = np.linspace(-1, 1, n_passos)
        grau = 2 if n_passos >= 5 else 1
        poly_coeffs = np.polyfit(t, signatures, deg=grau)
        trend = np.polyval(poly_coeffs, t.reshape(-1, 1))
        
        # 2. Resíduo bruto de dispersão
        raw_noise = np.linalg.norm(signatures - trend, axis=1)
        
        # 3. Velocidade esférica
        velocities = np.linalg.norm(np.diff(signatures, axis=0), axis=1)
        
        # 4. Calibração contra a Linha de Base Nula (Z-Score Dimensional)
        null_mean, null_std = self._estimate_null_baseline((n_passos, input_dim))
        null_std = np.where(null_std == 0, 1e-6, null_std)
        
        # Métrica Formal: Excesso de Estrutura vs Ruído Branco
        # Z negativo = resíduo MENOR que o esperado ao acaso = mais estrutura (bom).
        # Z positivo = resíduo MAIOR que o esperado ao acaso = mais ruidoso que o acaso (ruim).
        excess_structure = (raw_noise - null_mean) / null_std

        return {
            "signatures": signatures,
            "velocity": velocities,
            "raw_noise": raw_noise,
            "excess_structure_zscore": excess_structure,
            # Antes: np.exp(-np.abs(excess_structure)) tratava z muito negativo (mais
            # coerente que o acaso) da mesma forma que z muito positivo (mais ruidoso
            # que o acaso), cancelando o sinal útil. Agora preserva o sinal do z-score:
            # expit(-z) -> perto de 1.0 quando z é bem negativo (coerente),
            #              perto de 0.0 quando z é bem positivo (ruidoso).
            "structural_coherence": expit(-excess_structure)
        }

