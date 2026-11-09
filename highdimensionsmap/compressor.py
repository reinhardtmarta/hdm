import time
from typing import Tuple, Union
import numpy as np
from scipy.fftpack import dct, idct

# Importações condicionais para frameworks de terceiros (não força dependência rígida)
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class HDMCompressor:
    """Compressor e redutor de dimensionalidade adaptativo baseado na DCT-II.
    
    Seleciona coeficientes por ordem decrescente de magnitude até atingir
    o limiar de energia acumulada especificado (target_energy).
    """

    def __init__(self, target_energy: float = 0.999):
        if not (isinstance(target_energy, (float, int)) and 0.0 < target_energy <= 1.0):
            raise ValueError("target_energy deve ser um numero entre 0.0 (exclusivo) e 1.0 (inclusivo).")
        self.target_energy = float(target_energy)

    def _convert_input(self, X: Union[np.ndarray, "pd.DataFrame", "pd.Series", "torch.Tensor"]) -> Tuple[np.ndarray, str]:
        """Converte entradas diversas (Pandas, PyTorch) para numpy.ndarray com seguranca."""
        original_type = "numpy"

        if HAS_TORCH and isinstance(X, torch.Tensor):
            original_type = "torch"
            X = X.detach().cpu().numpy()
        elif HAS_PANDAS and isinstance(X, (pd.DataFrame, pd.Series)):
            original_type = "pandas"
            X = X.to_numpy()

        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        return X, original_type

    def fit_transform(
        self, X: Union[np.ndarray, "pd.DataFrame", "pd.Series", "torch.Tensor"]
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Aplica a DCT-II e projeta os vetores com base na energia acumulada.

        Trata casos de borda como NaN, Inf e vetores nulos.

        Parâmetros:
            X: Matriz 2D (n_amostras, n_dimensoes) ou vetor 1D. Aceita NumPy, Pandas e PyTorch.

        Retorna:
            tuple: (X_reconstruido, k_por_linha, tempo_execucao_ms)
        """
        t0 = time.perf_counter()

        # 1. Normalização de tipo de entrada
        X_np, original_type = self._convert_input(X)

        # 2. Validação de formato (dimensionalidade)
        if X_np.ndim not in (1, 2):
            raise ValueError(f"Entrada invalida. Esperado array 1D ou 2D, mas recebido ndim={X_np.ndim}.")

        is_1d = X_np.ndim == 1
        if is_1d:
            X_np = X_np.reshape(1, -1)

        # 3. Tratamento de NaN e Inf
        if not np.isfinite(X_np).all():
            raise ValueError("A matriz de entrada contem valores invalidos (NaN ou Inf). Limpe os dados antes de processar.")

        # 4. Processamento da DCT-II nas linhas (axis=1)
        coefs = dct(X_np, type=2, norm="ortho", axis=1)

        # 5. Ordenação por magnitude decrescente por linha
        abs_coefs = np.abs(coefs)
        sorted_indices = np.argsort(-abs_coefs, axis=1)

        rows = np.arange(X_np.shape[0])[:, None]
        sorted_sq_coefs = coefs[rows, sorted_indices] ** 2

        # 6. Cálculo da Energia Total por linha (Tratamento para Vetores Nulos)
        total_energy = np.sum(coefs**2, axis=1, keepdims=True)
        null_vectors_mask = total_energy.squeeze(axis=1) == 0.0

        # Evita divisão por zero em vetores nulos
        safe_total_energy = np.where(total_energy == 0.0, 1.0, total_energy)
        cum_energy = np.cumsum(sorted_sq_coefs, axis=1) / safe_total_energy

        # 7. Determinação adaptativa do parâmetro K por linha
        mask_energy = cum_energy >= self.target_energy
        k_per_row = np.argmax(mask_energy, axis=1) + 1

        # Trata linhas que não atingem o limiar exacto ou são nulas
        no_reach = ~np.any(mask_energy, axis=1)
        k_per_row[no_reach] = X_np.shape[1]
        k_per_row[null_vectors_mask] = 1  # Vetores nulos exigem apenas 1 componente (zerado)

        # 8. Filtro dos coeficientes fora do Top-K
        col_indices = np.arange(X_np.shape[1])[None, :]
        mask_top_k = col_indices < k_per_row[:, None]

        filtered_coefs_sorted = np.zeros_like(coefs)
        filtered_coefs_sorted[mask_top_k] = coefs[rows, sorted_indices][mask_top_k]

        # 9. Reordenação para o espaço de frequência original
        filtered_coefs = np.zeros_like(coefs)
        filtered_coefs[rows, sorted_indices] = filtered_coefs_sorted

        # 10. Reconstrução vetorial via IDCT-II
        X_reconstructed = idct(filtered_coefs, type=2, norm="ortho", axis=1)

        t1 = time.perf_counter()
        exec_time_ms = (t1 - t0) * 1000.0

        if is_1d:
            return X_reconstructed.squeeze(0), k_per_row[0], exec_time_ms

        return X_reconstructed, k_per_row, exec_time_ms
        
