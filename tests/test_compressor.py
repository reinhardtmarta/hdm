import numpy as np
import pytest
from hdm.compressor import HDMCompressor


def test_compressor_structured_data():
    t = np.linspace(0, 1, 100)
    vetor = np.sin(2 * np.pi * 3 * t)

    compressor = HDMCompressor(target_energy=0.999)
    vetor_rec, k = compressor.fit_transform(vetor)

    # Garante alta compressão em dados estruturados (K pequeno)
    assert k < 20
    # Garante baixo erro quadrático médio
    mse = np.mean((vetor - vetor_rec) ** 2)
    assert mse < 1e-3


def test_compressor_invalid_energy():
    with pytest.raises(ValueError):
        HDMCompressor(target_energy=1.5)
      
