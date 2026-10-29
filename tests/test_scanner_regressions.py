import numpy as np
import pytest

from highdimensionsmap import HDMScanner, MotionNoiseTracker


def test_generate_dataset_is_reproducible_and_has_expected_shape():
    a = HDMScanner(input_dim=12, seed=9).generate_dataset(7)
    b = HDMScanner(input_dim=12, seed=9).generate_dataset(7)
    assert a.shape == (7, 12)
    np.testing.assert_allclose(a, b)


def test_single_sample_transform_does_not_collapse_to_zero():
    scanner = HDMScanner(input_dim=4, latent_modes=3, seed=2)
    signature = scanner.transform([1, 2, 3, 4])
    assert signature.shape == (3,)
    assert np.linalg.norm(signature) > 0


def test_query_returns_exact_match_first():
    scanner = HDMScanner(input_dim=8, latent_modes=4, seed=5)
    data = scanner.generate_dataset(10)
    signatures = scanner.transform(data)
    indices, distances = scanner.query(signatures, data[0], k=3)
    assert indices[0] == 0
    assert distances[0] < 1e-10


def test_tracker_reports_documented_noise_key():
    scanner = HDMScanner(input_dim=8, latent_modes=4, seed=3)
    data = scanner.generate_dataset(8)
    result = MotionNoiseTracker(scanner).track(data)
    assert result["noise"].shape == (8,)
    assert result["velocity"].shape == (7,)


def test_fit_rejects_non_finite_values():
    scanner = HDMScanner(input_dim=2)
    with pytest.raises(ValueError, match="finitos"):
        scanner.fit([[1.0, np.nan]])
