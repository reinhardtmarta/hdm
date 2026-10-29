import numpy as np

class HDMScanner:
    """
    Scanner dimensional determinístico com suporte a projeções ortogonais
    adaptativas (SVD/PCA), padronização de feições e modos espectrais.
    """

    def __init__(self, input_dim=512, latent_modes=32, method="adaptive", scale=True, seed=42, steps=None):
        if int(input_dim) < 1:
            raise ValueError("input_dim precisa ser >= 1.")
        if int(latent_modes) < 1:
            raise ValueError("latent_modes precisa ser >= 1.")
        if method not in ("adaptive", "spectral", "random"):
            raise ValueError("method deve ser 'adaptive', 'spectral' ou 'random'.")
        self.input_dim = int(input_dim)
        self.latent_modes = int(latent_modes)
        self.method = method
        self.scale = scale
        self.seed = seed
        self.steps = steps  # mantido por compatibilidade com exemplos antigos
        self.projection_matrix = None
        self.mean_ = None
        self.std_ = None

        if self.method == "spectral":
            self.projection_matrix = self._build_dct_basis(self.latent_modes, self.input_dim)
        elif self.method == "random":
            self.projection_matrix = self._build_random_basis(self.latent_modes, self.input_dim)

    def _build_dct_basis(self, n_modes, d):
        n = np.arange(d, dtype=np.float64)
        basis = np.zeros((n_modes, d), dtype=np.float64)
        for k in range(n_modes):
            if k == 0:
                basis[k, :] = 1.0 / np.sqrt(d)
            else:
                basis[k, :] = np.sqrt(2.0 / d) * np.cos((np.pi * (2.0 * n + 1.0) * k) / (2.0 * d))
        return basis

    def _build_random_basis(self, n_modes, d):
        rng = np.random.default_rng(self.seed)
        gaussian_matrix = rng.standard_normal((d, n_modes))
        q, _ = np.linalg.qr(gaussian_matrix)
        return q.T

    def generate_dataset(self, n_samples=1, noise=0.05, drift=0.1):
        """Gera dados sintéticos reproduzíveis para testes e exemplos.

        Cada linha é uma amostra. ``noise`` controla o ruído gaussiano;
        ``drift`` adiciona uma tendência gradual ao longo das amostras.
        """
        if int(n_samples) < 1:
            raise ValueError("n_samples precisa ser >= 1.")
        if noise < 0 or drift < 0:
            raise ValueError("noise e drift precisam ser >= 0.")
        rng = np.random.default_rng(self.seed)
        data = rng.standard_normal((int(n_samples), self.input_dim))
        if drift:
            trend = np.linspace(0.0, float(drift), int(n_samples))[:, None]
            direction = np.ones((1, self.input_dim), dtype=np.float64) / np.sqrt(self.input_dim)
            data = data + trend * direction
        if noise:
            data = data + rng.normal(0.0, float(noise), size=data.shape)
        return data

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        X = np.atleast_2d(X)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X precisa ser uma matriz 2D não vazia.")
        if not np.isfinite(X).all():
            raise ValueError("X deve conter apenas valores finitos.")
        self.input_dim = X.shape[1]

        # Uma única amostra não permite estimar componentes principais.
        # Use uma base ortogonal determinística sem centralizar pela amostra,
        # pois isso zeraria completamente a assinatura.
        if X.shape[0] == 1:
            self.mean_ = np.zeros(self.input_dim, dtype=np.float64)
            self.std_ = np.ones(self.input_dim, dtype=np.float64)
            self.projection_matrix = self._build_random_basis(
                min(self.latent_modes, self.input_dim), self.input_dim
            )
            return self

        self.mean_ = np.mean(X, axis=0)
        X_proc = X - self.mean_

        if self.scale:
            self.std_ = np.std(X_proc, axis=0)
            self.std_[self.std_ == 0.0] = 1.0
            X_proc = X_proc / self.std_
        else:
            self.std_ = np.ones(self.input_dim, dtype=np.float64)

        _, _, Vh = np.linalg.svd(X_proc, full_matrices=False)
        self.projection_matrix = Vh[: min(self.latent_modes, Vh.shape[0]), :]
        return self

    def transform(self, X, normalize=False):
        X = np.asarray(X, dtype=np.float64)
        X = np.atleast_2d(X)

        if X.shape[1] != self.input_dim or self.projection_matrix is None:
            self.fit(X)

        X_proc = X - self.mean_ if self.mean_ is not None else X
        if self.scale and self.std_ is not None:
            X_proc = X_proc / self.std_

        signatures = np.dot(X_proc, self.projection_matrix.T)

        if normalize:
            norms = np.linalg.norm(signatures, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            signatures = signatures / norms

        return signatures if signatures.shape[0] > 1 else signatures[0]

    def query(self, dataset, query_vector, k=5):
        dataset_array = np.asarray(dataset, dtype=np.float64)
        if dataset_array.ndim == 1:
            dataset_array = dataset_array.reshape(1, -1)

        query_sig = self.transform(query_vector)

        if dataset_array.shape[1] != query_sig.shape[0]:
            dataset_signatures = self.transform(dataset_array)
        else:
            dataset_signatures = dataset_array

        distances = np.linalg.norm(dataset_signatures - query_sig, axis=1)
        nearest = np.argsort(distances)[:k]
        return nearest, distances[nearest]
