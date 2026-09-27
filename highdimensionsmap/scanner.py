import numpy as np


class HDMScanner:
    """
    Scanner dimensional determinístico baseado em decomposição analítica ortogonal (DCT-II).
    Elimina geradores pseudoaleatórios, saturação por tanh e descarte de fase de FFT.
    """

    def __init__(self, input_dim=512, latent_modes=32, steps=None, seed=None):
        self.input_dim = int(input_dim)
        self.latent_modes = int(latent_modes)
        # Parâmetros legados mantidos como opcionais para evitar quebra de contratos de chamada
        self.steps = steps
        self.seed = seed
        self.projection_matrix = self._build_deterministic_basis(self.latent_modes, self.input_dim)

    def _build_deterministic_basis(self, n_modes, d):
        """Gera uma base ortonormal discreta pura (DCT tipo II) sem uso de rng."""
        n = np.arange(d, dtype=np.float64)
        basis = np.zeros((n_modes, d), dtype=np.float64)
        for k in range(n_modes):
            if k == 0:
                basis[k, :] = 1.0 / np.sqrt(d)
            else:
                basis[k, :] = np.sqrt(2.0 / d) * np.cos((np.pi * (2.0 * n + 1.0) * k) / (2.0 * d))
        return basis

    def _align_dimensions(self, X):
        """Reamostra analiticamente vetores de dimensão diferente sem preenchimento estático com zeros."""
        X = np.asarray(X, dtype=np.float64)
        X = np.atleast_2d(X)
        d = X.shape[1]

        if d != self.input_dim:
            orig_indices = np.linspace(0.0, 1.0, d)
            target_indices = np.linspace(0.0, 1.0, self.input_dim)
            X = np.array([np.interp(target_indices, orig_indices, row) for row in X], dtype=np.float64)

        return X

    def generate_dataset(self, n_samples=1, noise=0.0, drift=0.1):
        """
        Gera trajetórias determinísticas para calibração e teste baseadas em harmónicos ortogonais.
        Não usa geradores aleatórios.
        """
        t = np.linspace(0.0, 1.0, n_samples)
        x = np.linspace(-1.0, 1.0, self.input_dim)
        dataset = np.zeros((n_samples, self.input_dim), dtype=np.float64)

        for i, ti in enumerate(t):
            base_curve = np.cos(np.pi * ti * x) + drift * x
            dataset[i] = base_curve

        norms = np.linalg.norm(dataset, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dataset / norms

    def transform(self, X):
        """Projeção linear ortogonal estrita preservando distâncias euclidianas relativas."""
        X = self._align_dimensions(X)

        # Projeção ortogonal direta
        signatures = np.dot(X, self.projection_matrix.T)

        # Normalização de norma unitária (espaço esférico)
        norms = np.linalg.norm(signatures, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        signatures = signatures / norms

        return signatures if signatures.shape[0] > 1 else signatures[0]

    def query(self, dataset, query_vector, k=5):
        """Consulta k vizinhos mais próximos no espaço projetado."""
        dataset_array = np.asarray(dataset, dtype=np.float64)
        query_sig = self.transform(query_vector)

        if dataset_array.ndim == 1:
            dataset_array = dataset_array.reshape(1, -1)

        # Se os dados passados forem brutos (não transformados), aplica a projeção
        if dataset_array.shape[1] != query_sig.shape[0]:
            dataset_signatures = self.transform(dataset_array)
        else:
            dataset_signatures = dataset_array

        distances = np.linalg.norm(dataset_signatures - query_sig, axis=1)
        nearest = np.argsort(distances)[:k]
        return nearest, distances[nearest]
        
