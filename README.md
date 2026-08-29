# HighDimensionsMap

High Dimensions Map is a lightweight Python library for generating and analyzing high-dimensional data with spectral signatures, kinematic tracking, and similarity queries.

It is designed to work like a regular scientific Python package: import it, create a scanner, generate data, transform vectors, and run nearest-neighbor queries directly from Python, notebooks, or Google Colab.

## Installation

Install directly from PyPI:

```bash
pip install highdimensionsmap
```

Or install the latest development version from GitHub:
pip install git+[https://github.com/reinhardtmarta/highdimensionsmap.git](https://github.com/reinhardtmarta/highdimensionsmap.git)

Or from a local checkout:
cd highdimensionsmap
pip install -e .

Quick start
from highdimensionsmap import HDMScanner, MotionNoiseTracker

# 1. Initialize scanner
scanner = HDMScanner(input_dim=128, latent_modes=32, steps=48, seed=42)

# 2. Generate and transform dataset
dataset = scanner.generate_dataset(20)
signatures = scanner.transform(dataset)
print("Signatures shape:", signatures.shape)

# 3. Query nearest neighbors
query_vec = dataset[0]
nearest_indices, distances = scanner.query(signatures, query_vec, k=3)
print("Nearest neighbor indices:", nearest_indices)

# 4. Track motion and noise
tracker = MotionNoiseTracker(scanner)
metrics = tracker.track(dataset)
print("Velocity (first 5 steps):", metrics["velocity"][:5])
print("Estimated noise (first 5 steps):", metrics["noise"][:5])

API overview
HDMScanner
scanner = HDMScanner(input_dim=128, latent_modes=32, steps=48, seed=42)

Methods:
 * generate_dataset(n_samples=1, noise=0.05, drift=0.1)
 * transform(X)
 * query(dataset_signatures, query_vector, k=5)
MotionNoiseTracker
tracker = MotionNoiseTracker(scanner)
result = tracker.track(trajectory, filter_window=3)

Returns a dictionary containing:
 * signatures: Spectral projection signatures.
 * velocity: Instantaneous spectral displacement between consecutive steps.
 * noise: Residual stochastic dispersion per step.
Optional API Server
This package is designed as a Python library first. If you want to use the optional FastAPI interface:
uvicorn highdimensionsmap.api:app --reload

# HighDimensionsMap (Português)
High Dimensions Map é uma biblioteca Python leve para geração e análise de dados de alta dimensão com assinaturas espectrais, rastreamento cinemático e consultas de similaridade.
Foi projetada para funcionar como um pacote científico padrão do Python: basta importar, criar um scanner, gerar dados, transformar vetores e executar consultas de vizinhos mais próximos diretamente do Python, notebooks ou Google Colab.
Instalação
Instale diretamente do PyPI:
pip install highdimensionsmap

Ou instale a versão de desenvolvimento mais recente do GitHub:
pip install git+[https://github.com/reinhardtmarta/highdimensionsmap.git](https://github.com/reinhardtmarta/highdimensionsmap.git)

Ou a partir de um diretório local:
cd highdimensionsmap
pip install -e .

Início rápido
from highdimensionsmap import HDMScanner, MotionNoiseTracker

## 1. Inicializar o scanner
scanner = HDMScanner(input_dim=128, latent_modes=32, steps=48, seed=42)

## 2. Gerar e transformar o dataset
dataset = scanner.generate_dataset(20)
assinaturas = scanner.transform(dataset)
print("Formato das assinaturas:", assinaturas.shape)

## 3. Consultar vizinhos mais próximos
vetor_busca = dataset[0]
indices_proximos, distancias = scanner.query(assinaturas, vetor_busca, k=3)
print("Índices dos vizinhos mais próximos:", indices_proximos)

## 4. Rastrear movimento e ruído
tracker = MotionNoiseTracker(scanner)
metricas = tracker.track(dataset)
print("Velocidade (primeiros 5 passos):", metricas["velocity"][:5])
print("Ruído estimado (primeiros 5 passos):", metricas["noise"][:5])

Visão geral da API
HDMScanner
scanner = HDMScanner(input_dim=128, latent_modes=32, steps=48, seed=42)

Métodos:
 * generate_dataset(n_samples=1, noise=0.05, drift=0.1)
 * transform(X)
 * query(dataset_signatures, query_vector, k=5)
MotionNoiseTracker
tracker = MotionNoiseTracker(scanner)
resultado = tracker.track(trajectory, filter_window=3)

Retorna um dicionário contendo:
 * signatures: Assinaturas de projeção espectral.
 * velocity: Deslocamento espectral instantâneo entre passos consecutivos.
 * noise: Dispersão estocástica residual por passo.
Servidor de API Opcional
Este pacote foi projetado primariamente como uma biblioteca Python. Se desejar utilizar a interface opcional em FastAPI:
uvicorn highdimensionsmap.api:app --reload


