import os
import json
import time
from datetime import datetime
from typing import Tuple, Union, Optional, Dict, Any
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
    """Compressor, redutor dimensional e mapeador de resíduo baseado na DCT-II.
    
    Seleciona coeficientes por ordem decrescente de magnitude até atingir
    o limiar de energia acumulada especificado (target_energy), preservando
    e diagnosticando o resíduo (ruído) descartado.
    """

    def __init__(self, target_energy: float = 0.999, max_dim: int = 65536):
        if not (isinstance(target_energy, (float, int)) and 0.0 < target_energy <= 1.0):
            raise ValueError("target_energy deve ser um numero entre 0.0 (exclusivo) e 1.0 (inclusivo).")
        if not (isinstance(max_dim, int) and max_dim >= 1):
            raise ValueError("max_dim deve ser um inteiro >= 1.")
        self.target_energy = float(target_energy)
        self.max_dim = int(max_dim)
        self.last_residual_report: Optional[Dict[str, Any]] = None

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

    def _analyze_and_export_residual(
        self,
        X_np: np.ndarray,
        X_reconstructed: np.ndarray,
        discarded_coefs: np.ndarray,
        k_per_row: np.ndarray,
        output_dir: Optional[str] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Organiza o resíduo descartado, imprime diagnóstico e salva arquivo para pesquisa."""
        residual_matrix = X_np - X_reconstructed
        n_samples, n_dims = X_np.shape

        # 1. Energia do sinal vs Energia do ruído (resíduo)
        total_energy = np.sum(X_np ** 2, axis=1)
        residual_energy = np.sum(residual_matrix ** 2, axis=1)
        safe_total = np.where(total_energy == 0.0, 1.0, total_energy)
        noise_ratio = residual_energy / safe_total

        # 2. Entropia espectral do resíduo (1.0 = ruído branco aleatório | < 0.5 = padrão oculto)
        sq_discarded = discarded_coefs ** 2
        sum_sq = np.sum(sq_discarded, axis=1, keepdims=True)
        safe_sum_sq = np.where(sum_sq == 0.0, 1.0, sum_sq)
        probs = sq_discarded / safe_sum_sq
        probs_safe = np.where(probs > 1e-12, probs, 1.0)
        entropy = -np.sum(np.where(probs > 1e-12, probs * np.log2(probs_safe), 0.0), axis=1)
        max_entropy = np.log2(max(n_dims, 2))
        normalized_entropy = entropy / max_entropy

        # 3. Dimensões mais afetadas pelo ruído (variância por coluna do resíduo)
        dim_noise_variance = np.var(residual_matrix, axis=0)
        top_noisy_dims = np.argsort(-dim_noise_variance)[: min(10, n_dims)]

        mean_entropy = float(np.mean(normalized_entropy))
        classification = (
            "Ruido aleatorio/termico (alta entropia)"
            if mean_entropy > 0.7
            else "Residuo com estrutura/padrao oculto (baixa entropia)"
        )

        report: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "dimensions_mapped": int(n_dims),
            "samples_processed": int(n_samples),
            "target_energy": self.target_energy,
            "mean_k_retained": float(np.mean(k_per_row)),
            "mean_noise_energy_ratio": float(np.mean(noise_ratio)),
            "mean_residual_spectral_entropy": mean_entropy,
            "noise_classification": classification,
            "top_noisy_dimensions_indices": top_noisy_dims.tolist(),
            "top_noisy_dimensions_variance": dim_noise_variance[top_noisy_dims].tolist(),
        }

        if verbose:
            print("\n--- [HDM] Relatorio de Residuo e Ruido ---")
            print(f"Dimensoes mapeadas : {n_dims} | Amostras: {n_samples}")
            print(f"Componentes retidos: {report['mean_k_retained']:.1f} (media por amostra)")
            print(f"Energia no residuo : {report['mean_noise_energy_ratio']*100:.4f}%")
            print(f"Entropia espectral : {mean_entropy:.4f} -> {classification}")
            print(f"Top dimensoes ruidosas (indices): {report['top_noisy_dimensions_indices'][:5]}")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            json_path = os.path.join(output_dir, f"hdm_residual_report_{tag}.json")
            npz_path = os.path.join(output_dir, f"hdm_residual_tensors_{tag}.npz")

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            np.savez_compressed(
                npz_path,
                residual_matrix=residual_matrix,
                discarded_dct_coefs=discarded_coefs,
                dim_noise_variance=dim_noise_variance,
                sample_spectral_entropy=normalized_entropy,
            )
            report["saved_files"] = {"report_json": json_path, "tensors_npz": npz_path}
            if verbose:
                print(f"Arquivos gerados   : {json_path} | {npz_path}")

        if verbose:
            print("------------------------------------------\n")

        self.last_residual_report = report
        return report

    def fit_transform(
        self,
        X: Union[np.ndarray, "pd.DataFrame", "pd.Series", "torch.Tensor"],
        output_dir: Optional[str] = None,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Aplica a DCT-II, projeta os vetores e organiza o resíduo sem descartar dados.

        Parâmetros:
            X: Matriz 2D (n_amostras, n_dimensoes) ou vetor 1D.
            output_dir: Se informado, salva o relatório (.json) e os tensores de ruído (.npz).
            verbose: Se True, imprime o diagnóstico do resíduo no terminal.

        Retorna:
            tuple: (X_reconstruido, k_por_linha, tempo_execucao_ms)
        """
        t0 = time.perf_counter()

        # 1. Normalização de tipo de entrada
        X_np, _ = self._convert_input(X)

        # 2. Validação de formato e limite dimensional
        if X_np.ndim not in (1, 2):
            raise ValueError(f"Entrada invalida. Esperado array 1D ou 2D, mas recebido ndim={X_np.ndim}.")

        is_1d = X_np.ndim == 1
        if is_1d:
            X_np = X_np.reshape(1, -1)

        if X_np.shape[1] > self.max_dim:
            raise ValueError(
                f"Dimensao da entrada ({X_np.shape[1]}) excede max_dim={self.max_dim}. "
                "Aumente max_dim ao instanciar HDMCompressor se tiver recursos suficientes."
            )

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
        total_energy = np.sum(coefs ** 2, axis=1, keepdims=True)
        null_vectors_mask = total_energy.squeeze(axis=1) == 0.0

        safe_total_energy = np.where(total_energy == 0.0, 1.0, total_energy)
        cum_energy = np.cumsum(sorted_sq_coefs, axis=1) / safe_total_energy

        # 7. Determinação adaptativa do parâmetro K por linha
        mask_energy = cum_energy >= self.target_energy
        k_per_row = np.argmax(mask_energy, axis=1) + 1

        no_reach = ~np.any(mask_energy, axis=1)
        k_per_row[no_reach] = X_np.shape[1]
        k_per_row[null_vectors_mask] = 1

        # 8. Separação entre coeficientes retidos (Top-K) e coeficientes residuais
        col_indices = np.arange(X_np.shape[1])[None, :]
        mask_top_k = col_indices < k_per_row[:, None]

        filtered_coefs_sorted = np.zeros_like(coefs)
        filtered_coefs_sorted[mask_top_k] = coefs[rows, sorted_indices][mask_top_k]

        discarded_coefs_sorted = np.zeros_like(coefs)
        discarded_coefs_sorted[~mask_top_k] = coefs[rows, sorted_indices][~mask_top_k]

        # 9. Reordenação para o espaço de frequência original
        filtered_coefs = np.zeros_like(coefs)
        filtered_coefs[rows, sorted_indices] = filtered_coefs_sorted

        discarded_coefs = np.zeros_like(coefs)
        discarded_coefs[rows, sorted_indices] = discarded_coefs_sorted

        # 10. Reconstrução vetorial via IDCT-II
        X_reconstructed = idct(filtered_coefs, type=2, norm="ortho", axis=1)

        # 11. Mapeamento e exportação do resíduo (nunca joga informação fora)
        self._analyze_and_export_residual(
            X_np=X_np,
            X_reconstructed=X_reconstructed,
            discarded_coefs=discarded_coefs,
            k_per_row=k_per_row,
            output_dir=output_dir,
            verbose=verbose,
        )

        t1 = time.perf_counter()
        exec_time_ms = (t1 - t0) * 1000.0

        if is_1d:
            return X_reconstructed.squeeze(0), k_per_row[0], exec_time_ms

        return X_reconstructed, k_per_row, exec_time_ms

    def compress(
        self, X: Union[np.ndarray, "pd.DataFrame", "pd.Series", "torch.Tensor"]
    ) -> dict:
        """Comprime a matriz/vetor retornando os K coeficientes relevantes e o resíduo."""
        X_np, _ = self._convert_input(X)
        is_1d = X_np.ndim == 1
        if is_1d:
            X_np = X_np.reshape(1, -1)

        if X_np.shape[1] > self.max_dim:
            raise ValueError(
                f"Dimensao da entrada ({X_np.shape[1]}) excede max_dim={self.max_dim}."
            )

        if not np.isfinite(X_np).all():
            raise ValueError("A matriz de entrada contem valores invalidos (NaN ou Inf).")

        coefs = dct(X_np, type=2, norm="ortho", axis=1)
        abs_coefs = np.abs(coefs)
        sorted_indices = np.argsort(-abs_coefs, axis=1)

        rows = np.arange(X_np.shape[0])[:, None]
        sorted_sq_coefs = coefs[rows, sorted_indices] ** 2

        total_energy = np.sum(coefs ** 2, axis=1, keepdims=True)
        null_mask = total_energy.squeeze(axis=1) == 0.0
        safe_total = np.where(total_energy == 0.0, 1.0, total_energy)
        cum_energy = np.cumsum(sorted_sq_coefs, axis=1) / safe_total

        mask_energy = cum_energy >= self.target_energy
        k_per_row = np.argmax(mask_energy, axis=1) + 1
        k_per_row[~np.any(mask_energy, axis=1)] = X_np.shape[1]
        k_per_row[null_mask] = 1

        compressed_values = []
        compressed_indices = []
        residual_values = []
        residual_indices = []

        for i in range(X_np.shape[0]):
            k = k_per_row[i]
            idx_keep = sorted_indices[i, :k]
            idx_drop = sorted_indices[i, k:]

            compressed_indices.append(idx_keep.astype(np.int32))
            compressed_values.append(coefs[i, idx_keep].astype(np.float32))

            residual_indices.append(idx_drop.astype(np.int32))
            residual_values.append(coefs[i, idx_drop].astype(np.float32))

        return {
            "values": compressed_values,
            "indices": compressed_indices,
            "residual_values": residual_values,
            "residual_indices": residual_indices,
            "shape": X_np.shape,
            "is_1d": is_1d,
        }

    def decompress(self, compressed_data: dict, include_residual: bool = False) -> np.ndarray:
        """Reconstrói a matriz/vetor original (com opção de restaurar o resíduo completo)."""
        values = compressed_data["values"]
        indices = compressed_data["indices"]
        shape = compressed_data["shape"]
        is_1d = compressed_data["is_1d"]

        coefs = np.zeros(shape, dtype=np.float32)

        for i in range(shape[0]):
            coefs[i, indices[i]] = values[i]
            if include_residual and "residual_indices" in compressed_data:
                res_idx = compressed_data["residual_indices"][i]
                res_val = compressed_data["residual_values"][i]
                coefs[i, res_idx] = res_val

        X_reconstructed = idct(coefs, type=2, norm="ortho", axis=1)

        if is_1d:
            return X_reconstructed.squeeze(0)
        return X_reconstructed
        
