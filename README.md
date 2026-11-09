
# High Dimensions Map (HDM) 📐

<p align="center">
  <b>PT:</b> Biblioteca Python para projeção ortogonal analítica, redução dimensional e compressão adaptativa de vetores via DCT-II.<br>
  <b>EN:</b> Python library for analytical orthogonal projection, dimensional reduction, and adaptive vector compression using DCT-II.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Version-0.2.2-blue?style=flat-square" alt="Version">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/Build-Passing-brightgreen?style=flat-square" alt="Build Status">
  <img src="https://img.shields.io/badge/License-Apache-green?style=flat-square" alt="License">
</p>

---

## 🇧🇷 Português

### Visão Geral
O **HDM (High Dimensions Map)** é uma biblioteca focada na otimização, redução dimensional e compressão adaptativa de vetores em espaços de alta dimensão (embeddings). 

Ao contrário de métodos baseados em aprendizado iterativo ou matrizes de covariância (como PCA), o HDM utiliza a **Transformada Discreta de Cosseno Tipo II (DCT-II)** de forma analítica e determinística (*training-free*), permitindo processamento em lote em tempo real com preservação garantida de energia e fidelidade semântica.

### Principais Recursos
- **`HDMCompressor` (Compressão Adaptativa):** Reduz a dimensão de vetores ordenando os coeficientes por magnitude decrescente e cortando no limiar exato de energia acumulada (`target_energy`).
- **Exportação Esparsa Compacta (`.compress()` / `.decompress()`):** Salva e transmite apenas os $K$ coeficientes relevantes e seus respetivos índices, garantindo redução real de bytes em disco e memória.
- **Robustez em Produção:** Proteção nativa contra vetores nulos (energia zero), entradas inválidas (`NaN`, `Inf`) e validação de dimensões.
- **Suporte Multiquadro:** Aceita matrizes `np.ndarray`, `torch.Tensor` (PyTorch) e `pd.DataFrame` / `pd.Series` (Pandas) nativamente.
- **Sem Necessidade de Treino (*Training-Free*):** Funciona sem requerer dados de ajuste prévio ou etapas pesadas de treino.

### Exemplos de Uso

#### 1. Redução e Reconstrução Direta (`fit_transform`)

```python
import numpy as np
from highdimensionmap import HDMCompressor

# Matriz de 1.000 vetores de 768 dimensões (padrão de embeddings)
X = np.random.randn(1000, 768)

# Instancia o compressor para reter 99.9% da energia do sinal
compressor = HDMCompressor(target_energy=0.999)

# Executa a transformação adaptativa
X_rec, k_componentes, tempo_ms = compressor.fit_transform(X)

print(f"Tempo de execução: {tempo_ms:.2f} ms")
print(f"Média de dimensões retidas: {np.mean(k_componentes):.1f} / {X.shape[1]}")

2. Exportação Esparsa Compacta (compress / decompress)
import numpy as np
from highdimensionmap import HDMCompressor

X = np.random.randn(100, 500)
compressor = HDMCompressor(target_energy=0.99)

# Exporta apenas os K coeficientes relevantes e seus índices
dados_compactos = compressor.compress(X)

print(f"Formato original: {dados_compactos['shape']}")
print(f"Coeficientes armazenados na linha 0: {len(dados_compactos['values'][0])} de 500")

# Reconstituição exata da matriz
X_rec = compressor.decompress(dados_compactos)
```

# 🇬🇧 English
## Overview
### HDM (High Dimensions Map) is a Python library focused on optimization, dimensional reduction, and adaptive compression of high-dimensional vectors (embeddings).
### Unlike methods relying on iterative learning or covariance matrices (such as PCA), HDM uses the Discrete Cosine Transform Type II (DCT-II) analytically and deterministically (training-free), enabling real-time batch processing with guaranteed energy preservation and semantic fidelity.
## Key Features
 ### * HDMCompressor (Adaptive Compression): Truncates vectors dynamically by sorting coefficients in descending magnitude order up to an exact energy threshold (target_energy).
### * Compact Sparse Export (.compress() / .decompress()): Stores and transmits only the top K relevant coefficients along with their original indices, delivering true memory and storage savings.
###  * Production Readiness: Native guardrails handling null vectors (zero energy), invalid inputs (NaN, Inf), and dimensionality validation.
### * Framework Support: Accepts np.ndarray, torch.Tensor (PyTorch), and pd.DataFrame / pd.Series (Pandas) seamlessly.
### * Training-Free: Operates vector-by-vector or in batches without prior calibration or training data.
## Usage Examples
1. Adaptive Transform & Reconstruction (fit_transform)

```
import numpy as np
from highdimensionmap import HDMCompressor

# Matrix with 1,000 vectors of 768 dimensions (embedding standard)
X = np.random.randn(1000, 768)

# Initialize compressor with 99.9% target energy threshold
compressor = HDMCompressor(target_energy=0.999)

# Execute adaptive transformation
X_rec, k_components, time_ms = compressor.fit_transform(X)

print(f"Execution time: {time_ms:.2f} ms")
print(f"Average dimensions retained: {np.mean(k_components):.1f} / {X.shape[1]}")

2. Compact Sparse Serialization (compress / decompress)
import numpy as np
from highdimensionmap import HDMCompressor

X = np.random.randn(100, 500)
compressor = HDMCompressor(target_energy=0.99)

# Compress into compact dictionary storing only K coefficients and indices
compressed_data = compressor.compress(X)

print(f"Original shape: {compressed_data['shape']}")
print(f"Stored coefficients for row 0: {len(compressed_data['values'][0])} of 500")

# Exact reconstruction
X_rec = compressor.decompress(compressed_data)

```

⚙️ Instalação / Installation
# Clone o repositório / Clone the repository
git clone [https://github.com/reinhardtmarta/hdm.git](https://github.com/reinhardtmarta/hdm.git)
cd hdm

# Instalação local em modo editável / Local editable installation
pip install -e .

🧪 Testes Automatizados / Running Tests
# Executar a suíte completa de testes via pytest
pytest tests/test_compressor.py -v


