class HDMScanner:
    """
    Scanner dimensional determinístico com suporte a projeções ortogonais
    adaptativas (SVD/PCA), padronização de feições e modos espectrais.
    """

    def __init__(
        self,
        input_dim=None,
        latent_modes=32,
        max_dim=16384,
        method="adaptive",
        scale=True,
        seed=42,
        steps=None,
    ):
        if input_dim is not None and int(input_dim) < 1:
            raise ValueError("input_dim precisa ser >= 1.")
        if int(latent_modes) < 1:
            raise ValueError("latent_modes precisa ser >= 1.")
        if int(max_dim) < 1:
            raise ValueError("max_dim precisa ser >= 1.")
        if input_dim is not None and int(input_dim) > int(max_dim):
            raise ValueError(
                f"input_dim ({input_dim}) excede o limite de segurança max_dim ({max_dim}). "
                "Aumente max_dim explicitamente se tiver recursos suficientes."
            )
        if method not in ("adaptive", "spectral", "random"):
            raise ValueError("method deve ser 'adaptive', 'spectral' ou 'random'.")

        self.input_dim = int(input_dim) if input_dim is not None else None
        self.latent_modes = int(latent_modes)
        self.max_dim = int(max_dim)
        self.method = method
        self.scale = scale
        self.seed = seed
        self.steps = steps
        self.projection_matrix = None
        self.mean_ = None
        self.std_ = None

        if self.input_dim is not None:
            self._init_basis(self.input_dim)

    def _validate_dimensions(self, n_cols):
        if n_cols > self.max_dim:
            raise ValueError(
                f"A entrada possui {n_cols} dimensoes, ultrapassando o limite aceitavel (max_dim={self.max_dim}). "
                "Ajuste 'max_dim' na inicializacao ou reduza a entrada antes de processar."
            )

    def _init_basis(self, d):
        modes = min(self.latent_modes, d)
        if self.method == "spectral":
            self.projection_matrix = self._build_dct_basis(modes, d)
        elif self.method == "random":
            self.projection_matrix = self._build_random_basis(modes, d)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        X = np.atleast_2d(X)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X precisa ser uma matriz 2D não vazia.")
        if not np.isfinite(X).all():
            raise ValueError("X deve conter apenas valores finitos.")

        self._validate_dimensions(X.shape[1])
        self.input_dim = X.shape[1]

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

        self._validate_dimensions(X.shape[1])

        if self.input_dim is None or X.shape[1] != self.input_dim or self.projection_matrix is None:
            self.input_dim = X.shape[1]
            if self.method == "adaptive":
                self.fit(X)
            else:
                self._init_basis(self.input_dim)

        X_proc = X - self.mean_ if self.mean_ is not None else X
        if self.scale and self.std_ is not None:
            X_proc = X_proc / self.std_

        signatures = np.dot(X_proc, self.projection_matrix.T)

        if normalize:
            norms = np.linalg.norm(signatures, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            signatures = signatures / norms

        return signatures if signatures.shape[0] > 1 else signatures[0]
