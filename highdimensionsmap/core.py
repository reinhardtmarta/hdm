import numpy as np


class hdm:
    """
    Motor determinista de projeção analítica ortogonal (DCT-II) para altas dimensões.
    Elimina geradores pseudoaleatórios, saturação por tanh e descarte de fase de FFT.
    """

    def __init__(self, input_dim=512, latent_modes=32, steps=None, seed=None):
        self.input_dim = int(input_dim)
        self.latent_modes = int(latent_modes)
        # steps e seed são mantidos por compatibilidade com chamadas anteriores
        self.steps = steps
        self.seed = seed
        self.projection_matrix = self._build_deterministic_basis(self.latent_modes, self.input_dim)

    def _build_deterministic_basis(self, n_modes, d):
        """Gera uma base ortonormal analítica pura (DCT-II) sem geradores aleatórios."""
        n = np.arange(d, dtype=np.float64)
        basis = np.zeros((n_modes, d), dtype=np.float64)
        for k in range(n_modes):
            if k == 0:
                basis[k, :] = 1.0 / np.sqrt(d)
            else:
                basis[k, :] = np.sqrt(2.0 / d) * np.cos((np.pi * (2.0 * n + 1.0) * k) / (2.0 * d))
        return basis

    def _align_dimensions(self, X):
        """Reamostragem linear contínua sem fatiamento destrutivo ou preenchimento com zeros."""
        X = np.asarray(X, dtype=np.float64)
        X = np.atleast_2d(X)
        d = X.shape[1]

        if d != self.input_dim:
            orig_indices = np.linspace(0.0, 1.0, d)
            target_indices = np.linspace(0.0, 1.0, self.input_dim)
            X = np.array([np.interp(target_indices, orig_indices, row) for row in X], dtype=np.float64)

        return X

    def transform(self, X):
        """Projeta os vetores reais de forma linear e ortogonal, preservando distâncias relativas."""
        X = self._align_dimensions(X)

        # Projeção ortogonal estrita
        signatures = np.dot(X, self.projection_matrix.T)

        # Normalização esférica unitária
        norms = np.linalg.norm(signatures, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        signatures = signatures / norms

        return signatures if signatures.shape[0] > 1 else signatures[0]

    def query_knn(self, dataset_signatures, query_vector, k=5):
        """Busca os k vizinhos mais próximos no espaço espectral determinado."""
        dataset_array = np.asarray(dataset_signatures, dtype=np.float64)
        query_sig = self.transform(query_vector)

        if dataset_array.ndim == 1:
            dataset_array = dataset_array.reshape(1, -1)

        # Se a entrada contiver dados brutos sem projeção, projeta antes de calcular distâncias
        if dataset_array.shape[1] != query_sig.shape[0]:
            dataset_signatures = self.transform(dataset_array)
        else:
            dataset_signatures = dataset_array

        distances = np.linalg.norm(dataset_signatures - query_sig, axis=1)
        nearest_indices = np.argsort(distances)[:k]
        return nearest_indices, distances[nearest_indices]
        
