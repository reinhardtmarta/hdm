import numpy as np
from scipy.fftpack import dct, idct


class HDMCompressor:
    """Compressor e redutor de dimensionalidade adaptativo baseado na DCT-II.
    
    Seleciona componentes por ordem decrescente de magnitude até atingir
    o limiar de energia acumulada especificado.
    """

    def __init__(self, target_energy: float = 0.999):
        if not 0.0 < target_energy <= 1.0:
            raise ValueError("target_energy deve estar entre 0.0 e 1.0.")
        self.target_energy = target_energy

    def fit_transform(self, X: np.ndarray):
        """Aplica a DCT-II e projeta os vetores com base na energia acumulada.

        Parâmetros:
            X (np.ndarray): Matriz 2D (n_amostras, n_dimensoes) ou vetor 1D.

        Retorna:
            tuple: (X_reconstruido, k_por_linha, tempo_execucao_ms)
        """
        # Garante suporte a vetores 1D convertendo temporariamente para 2D
        is_1d = X.ndim == 1
        if is_1d:
            X = X.reshape(1, -1)

        # 1. DCT-II bidimensional nas linhas (axis=1)
        coefs = dct(X, type=2, norm="ortho", axis=1)

        # 2. Ordena magnitudes por linha (decrescente)
        abs_coefs = np.abs(coefs)
        sorted_indices = np.argsort(-abs_coefs, axis=1)

        # 3. Reorganiza os coeficientes ao quadrado na ordem de magnitude
        rows = np.arange(X.shape[0])[:, None]
        sorted_sq_coefs = coefs[rows, sorted_indices] ** 2

        # 4. Energia acumulada por linha
        total_energy = np.sum(coefs**2, axis=1, keepdims=True)
        total_energy[total_energy == 0] = 1.0  # Evita divisão por zero
        cum_energy = np.cumsum(sorted_sq_coefs, axis=1) / total_energy

        # 5. Determina K (quantidade de componentes) para cada linha
        mask_energy = cum_energy >= self.target_energy
        k_per_row = np.argmax(mask_energy, axis=1) + 1

        # Caso não atinja o limiar exato, retém todas as dimensões
        no_reach = ~np.any(mask_energy, axis=1)
        k_per_row[no_reach] = X.shape[1]

        # 6. Máscara para filtrar componentes fora do top-K
        col_indices = np.arange(X.shape[1])[None, :]
        mask_top_k = col_indices < k_per_row[:, None]

        # 7. Zera coeficientes fora do top-K
        filtered_coefs_sorted = np.zeros_like(coefs)
        filtered_coefs_sorted[mask_top_k] = coefs[rows, sorted_indices][
            mask_top_k
        ]

        # 8. Restaura a ordem original das colunas
        filtered_coefs = np.zeros_like(coefs)
        filtered_coefs[rows, sorted_indices] = filtered_coefs_sorted

        # 9. Reconstrução via IDCT-II
        X_reconstructed = idct(filtered_coefs, type=2, norm="ortho", axis=1)

        if is_1d:
            return X_reconstructed.squeeze(0), k_per_row[0]

        return X_reconstructed, k_per_row
      
