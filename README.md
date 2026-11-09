
# High Dimensions Map (HDM) 📐

<p align="center">
  <b>PT:</b> Biblioteca Python para projeção ortogonal analítica e compressão adaptativa de vetores via DCT-II.<br>
  <b>EN:</b> Python library for analytical orthogonal projection and adaptive vector compression using DCT-II.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Version-0.2.2-blue?style=flat-square" alt="Version">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/Status-Active%20Development-orange?style=flat-square" alt="Status">
  <img src="https://img.shields.io/badge/License-Apache-green?style=flat-square" alt="License">
</p>

---

## 🇧🇷 Português

### Visão Geral
O **HDM (High Dimensions Map)** é uma biblioteca focada na otimização, redução dimensional e compressão adaptativa de vetores em espaços de alta dimensão. 

Ao contrário de métodos baseados em aprendizado iterativo ou matrizes de covariância (como PCA), o HDM utiliza a **Transformada Discreta de Cosseno Tipo II (DCT-II)** de forma analítica e determinística (*training-free*), permitindo processamento em lote em tempo real com preservação garantida de energia.

### Principais Recursos
- **`HDMCompressor` (Compressão Adaptativa):** Reduz a dimensão de vetores ordenando os coeficientes por magnitude decrescente e cortando no limiar exato de energia acumulada (`target_energy`).
- **Sem Necessidade de Treino (*Training-Free*):** Funciona vetor por vetor ou em lotes sem requerer dados de ajuste prévio.
- **Processamento Vetorizado:** Algoritmo implementado em C/NumPy nativo, capaz de processar 10.000 vetores de 1.000 dimensões em ~1.7 segundos.

### Exemplo de Uso Prático

```python
import numpy as np
from highdimensionmap import HDMCompressor

# Matriz de 1.000 vetores de 100 dimensões
X = np.random.randn(1000, 100)

# Instancia o compressor para reter 99.9% da energia do sinal
compressor = HDMCompressor(target_energy=0.999)

# Executa a transformação adaptativa
X_rec, k_componentes, tempo_ms = compressor.fit_transform(X)

print(f"Tempo de execução: {tempo_ms:.2f} ms")
print(f"Média de dimensões retidas: {np.mean(k_componentes):.1f} / {X.shape[1]}")
```
# 🇬🇧 English
## Overview
**HDM (High Dimensions Map) is a Python library focused on optimization**, dimensional reduction, and adaptive compression of high-dimensional vectors.
Unlike methods relying on iterative learning or covariance matrices (such as PCA), HDM uses the Discrete Cosine Transform Type II (DCT-II) analytically and deterministically (training-free), enabling real-time batch processing with guaranteed energy preservation.
Key Features
***HDMCompressor (Adaptive Compression): Reduces vector dimensionality by sorting coefficients by descending magnitude and truncating at the exact threshold of cumulative energy (target_energy)."**
**Training-Free: Operates vector-by-vector or in batches without requiring prior calibration data.**
**Vectorized Performance: Core algorithm implemented natively using NumPy/SciPy, capable of processing 10,000 1,000-dimensional vectors in ~1.7 seconds.**
Quickstart Example

```
import numpy as np
from highdimensionmap import HDMCompressor

# Matrix with 1,000 vectors of 100 dimensions
X = np.random.randn(1000, 100)

# Initialize compressor with 99.9% target energy threshold
compressor = HDMCompressor(target_energy=0.999)

# Execute adaptive transformation
X_rec, k_components, time_ms = compressor.fit_transform(X)

print(f"Execution time: {time_ms:.2f} ms")
print(f"Average dimensions retained: {np.mean(k_components):.1f} / {X.shape[1]}")

```

⚙️ Instalação / Installation
# Clone o repositório / Clone the repository
git clone [https://github.com/reinhardtmarta/hdm.git](https://github.com/reinhardtmarta/hdm.git)
cd hdm

# Instalação local em modo editável / Local editable installation
pip install -e .


