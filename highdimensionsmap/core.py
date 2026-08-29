import numpy as np


class hdm:
    """
    Motor leve de projeção geométrica não linear e indexação para altas dimensões.
    """

    def __init__(self, input_dim=512, latent_modes=32, steps=32, seed=42):
        self.input_dim = input_dim
        self.latent_modes = latent_modes
        self.steps = steps
        self.seed = seed
        self.time_grid = np.linspace(0, 2 * np.pi, steps)
        self._rebuild_projection_matrix(self.input_dim)

    def _rebuild_projection_matrix(self, input_dim):
        rng = np.random.default_rng(self.seed)
        w_raw = rng.standard_normal((self.latent_modes, input_dim))
        self.projection_matrix, _ = np.linalg.qr(w_raw.T)
        self.projection_matrix = self.projection_matrix.T

    def _align_dimensions(self, X):
        X = np.asarray(X, dtype=float)
        X = np.atleast_2d(X)

        if X.shape[1] != self.input_dim:
            if X.shape[1] > self.input_dim:
                X = X[:, : self.input_dim]
            else:
                pad_width = ((0, 0), (0, self.input_dim - X.shape[1]))
                X = np.pad(X, pad_width, mode="constant", constant_values=0.0)

        return X

    def transform(self, X):
        """Mapeia vetores D-dimensionais em assinaturas espectrais invariantes."""
        X = self._align_dimensions(X)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        X_norm = X / norms

        proj = np.dot(X_norm, self.projection_matrix.T)
        omegas = np.tanh(proj) * np.pi

        phases = omegas[:, :, np.newaxis] * self.time_grid[np.newaxis, np.newaxis, :]
        waveforms = np.cos(phases)

        fft_energy = np.abs(np.fft.rfft(waveforms, axis=2))
        signatures = fft_energy.reshape(X.shape[0], -1)

        sig_norms = np.linalg.norm(signatures, axis=1, keepdims=True)
        sig_norms[sig_norms == 0] = 1.0
        signatures = signatures / sig_norms

        return signatures if signatures.shape[0] > 1 else signatures[0]

    def query_knn(self, dataset_signatures, query_vector, k=5):
        """Busca os k vizinhos mais próximos no espaço espectral."""
        dataset_array = np.asarray(dataset_signatures, dtype=float)
        query_sig = self.transform(query_vector)

        if dataset_array.ndim == 1:
            dataset_array = dataset_array.reshape(1, -1)

        if dataset_array.shape[1] != query_sig.shape[0]:
            dataset_signatures = np.asarray([self.transform(row) for row in dataset_array], dtype=float)
        else:
            dataset_signatures = dataset_array

        distances = np.linalg.norm(dataset_signatures - query_sig, axis=1)
        nearest_indices = np.argsort(distances)[:k]
        return nearest_indices, distances[nearest_indices]

