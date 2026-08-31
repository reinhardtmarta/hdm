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
        
        # 1. Velocidade instantânea espectral
        velocities = np.linalg.norm(np.diff(signatures, axis=0), axis=1)
        
        # 2. Suavização temporal sem distorção artificial de borda (mode='reflect')
        smoothed = np.zeros_like(signatures)
        pad = filter_window // 2
        for col in range(n_features):
            # 'reflect' evita a falsa convergência nas pontas gerada por 'edge'
            padded = np.pad(signatures[:, col], pad, mode='reflect')
            smoothed[:, col] = np.convolve(padded, np.ones(filter_window)/filter_window, mode='valid')
            
        # 3. Resíduo pontual de ruído
        raw_noise = np.linalg.norm(signatures - smoothed, axis=1)
        
        # 4. Correção de escala por grau de liberdade das bordas
        # Ajusta o fator de variância esperada na convolução
        correcao_borda = np.ones(n_passos)
        correcao_borda[0] = np.sqrt(3.0 / 2.0)
        correcao_borda[-1] = np.sqrt(3.0 / 2.0)
        calibrated_noise = raw_noise * correcao_borda
        
        return {
            "signatures": signatures,
            "velocity": velocities,
            "noise": calibrated_noise
        }
        
