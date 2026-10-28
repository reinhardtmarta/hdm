import numpy as np
from .scanner import HDMScanner


class hdm(HDMScanner):
    """
    Interface de compatibilidade e atalho principal da biblioteca.
    Herda de HDMScanner com motor adaptativo (SVD/PCA) e suporte a padronização.
    """

    def __init__(self, input_dim=512, latent_modes=32, method="adaptive", scale=True, seed=42, steps=None):
        super().__init__(
            input_dim=input_dim,
            latent_modes=latent_modes,
            method=method,
            scale=scale,
            seed=seed
        )
        self.steps = steps

    def query_knn(self, dataset_signatures, query_vector, k=5):
        """
        Método de conveniência mantido para retrocompatibilidade.
        Redireciona para o método de consulta unificado do scanner.
        """
        return self.query(dataset_signatures, query_vector, k=k)
        
