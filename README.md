# High Dimensions Map (HDM) 📐

<p align="center">
  <b>PT:</b> Pesquisa e prototipagem de projeção ortogonal analítica usando DCT-II.<br>
  <b>EN:</b> Research and prototyping of analytical orthogonal projection using DCT-II.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Experimental-orange?style=flat-square" alt="Status">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/License-Apache-green?style=flat-square" alt="License">
</p>

---

## 🇧🇷 Português

### Visão Geral
O **HDM (High Dimensions Map)** é um projeto experimental focado na redução e mapeamento de vetores em espaços de alta dimensão. Em vez de utilizar abordagens estatísticas ou treinamentos iterativos, o HDM explora a **Transformada Discreta de Cosseno Tipo II (DCT-II)** para realizar projeções ortogonais determinísticas.

> **Status Atual:** Protótipo em fase de testes conceituais e benchmarking de reconstrução. O pacote ainda **não** está publicado no PyPI.

### Como Funciona (Conceito)
1. **Entrada:** Vetores densos de alta dimensão (ex: $N$ dimensões).
2. **Transformação:** Aplicação analítica da DCT-II para reorganizar a energia dos coeficientes.
3. **Projeção:** Retenção das $K$ componentes principais ($K < N$).

### Objetivo de Pesquisa
Avaliar os limites práticos de preservação de distância e erro de reconstrução (RMSE) em comparações diretas contra métodos tradicionais como PCA e SVD.

---

## 🇬🇧 English

### Overview
**HDM (High Dimensions Map)** is an experimental project focused on vector reduction and mapping in high-dimensional spaces. Rather than relying on statistical approaches or iterative training, HDM explores the **Discrete Cosine Transform Type II (DCT-II)** to perform deterministic orthogonal projections.

> **Current Status:** Prototype phase undergoing conceptual testing and reconstruction benchmarking. This package is **not** yet published on PyPI.

### How It Works (Concept)
1. **Input:** High-dimensional dense vectors (e.g., $N$ dimensions).
2. **Transformation:** Analytical application of DCT-II to concentrate coefficient energy.
3. **Projection:** Retention of the top $K$ components ($K < N$).

### Research Goal
Evaluate the practical limits of distance preservation and reconstruction error (RMSE) in direct benchmarks against traditional methods like PCA and SVD.

---

## ⚙️ Uso Local / Local Usage

```bash
# Clonar o repositório / Clone the repository
git clone [https://github.com/reinhardtmarta/hdm.git](https://github.com/reinhardtmarta/hdm.git)
cd hdm

# Instalar dependências locais / Install local dependencies
pip install -r requirements.txt
