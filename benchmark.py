import time
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from highdimensionsmap import HDMCompressor


def cosine_similarity_batch(A: np.ndarray, B: np.ndarray) -> float:
    """Calcula a similaridade de cosseno média entre duas matrizes linha a linha."""
    dot_products = np.sum(A * B, axis=1)
    norms_A = np.linalg.norm(A, axis=1)
    norms_B = np.linalg.norm(B, axis=1)
    
    valid_mask = (norms_A > 0) & (norms_B > 0)
    if not np.any(valid_mask):
        return 0.0
    
    sims = dot_products[valid_mask] / (norms_A[valid_mask] * norms_B[valid_mask])
    return float(np.mean(sims))


def run_benchmark():
    np.random.seed(42)
    
    # Dataset simulado de embeddings: 5.000 vetores de 768 dimensões (padrão BERT/OpenAI)
    n_samples = 5000
    n_dims = 768
    print(f"=== BENCHMARK DE DESEMPENHO (Dataset: {n_samples} vetores x {n_dims} dims) ===\n")
    
    # Gerando dados com componentes de frequência
    t = np.linspace(0, 1, n_dims)
    X = np.zeros((n_samples, n_dims))
    for i in range(n_samples):
        freq = np.random.uniform(1, 20)
        X[i] = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(n_dims)

    target_energy = 0.99  # Preservação de 99% da energia acumulada

    # -----------------------------------------------------------------
    # 1. HDM (DCT-II Adaptativo)
    # -----------------------------------------------------------------
    hdm = HDMCompressor(target_energy=target_energy)
    
    t0 = time.perf_counter()
    X_hdm, k_hdm_list, _ = hdm.fit_transform(X)
    t_hdm = (time.perf_counter() - t0) * 1000.0
    
    k_hdm_avg = float(np.mean(k_hdm_list))
    sim_hdm = cosine_similarity_batch(X, X_hdm)

    # -----------------------------------------------------------------
    # 2. PCA (Scikit-Learn)
    # -----------------------------------------------------------------
    n_components = max(1, int(k_hdm_avg))
    pca = PCA(n_components=n_components)
    
    t0 = time.perf_counter()
    X_pca_trans = pca.fit_transform(X)
    X_pca = pca.inverse_transform(X_pca_trans)
    t_pca = (time.perf_counter() - t0) * 1000.0
    
    sim_pca = cosine_similarity_batch(X, X_pca)

    # -----------------------------------------------------------------
    # 3. TruncatedSVD (Scikit-Learn)
    # -----------------------------------------------------------------
    svd = TruncatedSVD(n_components=n_components)
    
    t0 = time.perf_counter()
    X_svd_trans = svd.fit_transform(X)
    X_svd = svd.inverse_transform(X_svd_trans)
    t_svd = (time.perf_counter() - t0) * 1000.0
    
    sim_svd = cosine_similarity_batch(X, X_svd)

    # -----------------------------------------------------------------
    # EXIBIÇÃO DOS RESULTADOS
    # -----------------------------------------------------------------
    print(f"{'Algoritmo':<18} | {'Tempo (ms)':<12} | {'Dims Retidas':<15} | {'Similaridade Cosseno':<20}")
    print("-" * 72)
    print(f"{'HDM (DCT-II)':<18} | {t_hdm:<12.2f} | {k_hdm_avg:<15.1f} | {sim_hdm:<20.5f}")
    print(f"{'PCA':<18} | {t_pca:<12.2f} | {n_components:<15.1f} | {sim_pca:<20.5f}")
    print(f"{'TruncatedSVD':<18} | {t_svd:<12.2f} | {n_components:<15.1f} | {sim_svd:<20.5f}")
    print("-" * 72)
    
    if t_hdm > 0:
        speedup = t_pca / t_hdm
        print(f"\n⚡ A HDM foi ~{speedup:.1f}x mais rápida que o PCA!")


if __name__ == "__main__":
    run_benchmark()
  
