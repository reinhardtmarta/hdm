import numpy as np
import pytest
from highdimensionsmap.compressor import HDMCompressor

# Importações opcionais tratadas sem quebrar a suíte
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


# =====================================================================
# 1. TESTES NATIVOS (NUMPY & LÓGICA CORE) - SEMPRE EXECUTAM
# =====================================================================

def test_compressor_structured_vector():
    """Garante alta taxa de compressão e baixo erro em sinais estruturados 1D."""
    t = np.linspace(0, 1, 100)
    vetor = np.sin(2 * np.pi * 3 * t)

    compressor = HDMCompressor(target_energy=0.999)
    vetor_rec, k, tempo_ms = compressor.fit_transform(vetor)

    assert k < 20
    assert tempo_ms > 0
    mse = np.mean((vetor - vetor_rec) ** 2)
    assert mse < 1e-3


def test_compressor_batch_2d():
    """Valida o funcionamento correto em lotes 2D (n_amostras, n_dimensoes)."""
    np.random.seed(42)
    X = np.random.randn(20, 50)

    compressor = HDMCompressor(target_energy=0.99)
    X_rec, k_per_row, tempo_ms = compressor.fit_transform(X)

    assert X_rec.shape == (20, 50)
    assert len(k_per_row) == 20
    assert tempo_ms > 0.0


@pytest.mark.parametrize("invalid_energy", [0.0, -0.5, 1.5])
def test_invalid_target_energy(invalid_energy):
    """Garante que limiares de energia fora do intervalo (0, 1] lancem ValueError."""
    with pytest.raises(ValueError):
        HDMCompressor(target_energy=invalid_energy)


def test_invalid_array_dimensions():
    """Garante que arrays 3D ou superiores lancem ValueError."""
    X_3d = np.ones((5, 10, 10))
    compressor = HDMCompressor()
    with pytest.raises(ValueError, match="Entrada invalida"):
        compressor.fit_transform(X_3d)


def test_edge_case_null_vectors():
    """Garante que vetores com energia zero sejam tratados sem divisão por zero."""
    X_null = np.zeros((5, 50))
    compressor = HDMCompressor(target_energy=0.999)
    X_rec, k_per_row, _ = compressor.fit_transform(X_null)

    np.testing.assert_array_equal(k_per_row, np.ones(5))
    np.testing.assert_allclose(X_rec, 0.0, atol=1e-12)


def test_edge_case_nan_input():
    """Garante que entradas contendo NaN lancem exceção tratada."""
    X_nan = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]])
    compressor = HDMCompressor()
    with pytest.raises(ValueError, match="NaN ou Inf"):
        compressor.fit_transform(X_nan)


def test_edge_case_inf_input():
    """Garante que entradas contendo valores Inf lancem exceção tratada."""
    X_inf = np.array([[1.0, np.inf, 3.0], [4.0, 5.0, 6.0]])
    compressor = HDMCompressor()
    with pytest.raises(ValueError, match="NaN ou Inf"):
        compressor.fit_transform(X_inf)


# =====================================================================
# 2. TESTES OPCIONAIS (PANDAS & PYTORCH)
# =====================================================================

def test_pandas_dataframe_input():
    if not HAS_PANDAS:
        pytest.skip("Pandas nao instalado")
    df = pd.DataFrame(np.random.randn(10, 20))
    compressor = HDMCompressor(target_energy=0.99)
    X_rec, k_per_row, _ = compressor.fit_transform(df)
    assert X_rec.shape == (10, 20)


def test_pytorch_tensor_input():
    if not HAS_TORCH:
        pytest.skip("PyTorch nao instalado")
    tensor = torch.randn(15, 30)
    compressor = HDMCompressor(target_energy=0.99)
    X_rec, k_per_row, _ = compressor.fit_transform(tensor)
    assert X_rec.shape == (15, 30)
    
