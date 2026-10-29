# HighDimensionsMap (HDM)

Biblioteca Python experimental para gerar dados sintéticos, projetar vetores em um espaço de menor dimensão, consultar vizinhos próximos e calcular métricas descritivas de trajetórias.

> **Nota científica:** assinaturas, coerência estrutural e métricas de ruído são medidas numéricas exploratórias. Elas não demonstram, por si só, que uma estrutura real foi descoberta nem que uma anomalia tem significado causal.

## Instalação

A partir do repositório:

```bash
pip install -e ".[test]"
```

Instalação a partir do GitHub:

```bash
pip install git+https://github.com/reinhardtmarta/hdm.git
```

## Exemplo rápido

```python
from highdimensionsmap import HDMScanner, MotionNoiseTracker

scanner = HDMScanner(input_dim=128, latent_modes=32, seed=42)
dataset = scanner.generate_dataset(n_samples=20, noise=0.05, drift=0.1)
signatures = scanner.transform(dataset)

indices, distances = scanner.query(signatures, dataset[0], k=3)
print("Vizinhos:", indices)
print("Distâncias:", distances)

tracker = MotionNoiseTracker(scanner)
metrics = tracker.track(dataset)
print("Formato das assinaturas:", metrics["signatures"].shape)
print("Velocidade:", metrics["velocity"][:5])
print("Resíduo da suavização:", metrics["noise"][:5])
```

## Testes

```bash
python -m pytest -q
```

Os testes cobrem a API básica, geração de dados reproduzíveis, consulta de vizinhos, rastreamento e alguns casos-limite. Ainda são necessários testes científicos mais amplos com dados sintéticos controlados e dados reais apropriados.

## API opcional

```bash
uvicorn highdimensionsmap.api:app --reload
```

Por padrão, o Uvicorn fica acessível localmente. Não exponha a API à rede pública sem autenticação, limites de entrada e configuração de segurança adequados.
