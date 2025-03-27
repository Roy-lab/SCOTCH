import pytest
import torch
from initialize import nnsvd_nmf_initialize, nnsvd_nmtf_initialize


def test_nnsvd_nmf_initialize_shapes():
    X = torch.rand(10, 8)
    k = 4
    W, H = nnsvd_nmf_initialize(X, k)
    assert W.shape == (10, 4), "W does not have the expected shape"
    assert H.shape == (8, 4), "H does not have the expected shape"

def test_nnsvd_nmf_initialize_rank():
    X = torch.rand(10, 8)
    k = 3
    W, H = nnsvd_nmf_initialize(X, k)
    assert torch.linalg.matrix_rank(W) <= k, "Rank of W exceeds k"
    assert torch.linalg.matrix_rank(H) <= k, "Rank of H exceeds k"


def test_nnsvd_nmf_initialize_non_negativity():
    X = torch.rand(10, 8)
    k = 4
    W, H = nnsvd_nmf_initialize(X, k)
    assert torch.all(W >= 0), "W contains negative values"
    assert torch.all(H >= 0), "H contains negative values"


def test_nnsvd_nmf_initialize_reproducibility():
    X = torch.rand(10, 8)
    k = 4
    seed = 42
    W1, H1 = nnsvd_nmf_initialize(X, k, seed=seed)
    W2, H2 = nnsvd_nmf_initialize(X, k, seed=seed)
    assert torch.allclose(W1, W2), "W is not reproducible with the same seed"
    assert torch.allclose(H1, H2), "H is not reproducible with the same seed"


def test_nnsvd_nmf_initialize_different_seed():
    X = torch.rand(10, 8)
    k = 4
    seed1 = 42
    seed2 = 123
    W1, H1 = nnsvd_nmf_initialize(X, k, seed=seed1)
    W2, H2 = nnsvd_nmf_initialize(X, k, seed=seed2)
    assert not torch.allclose(W1, W2), "W should not be identical with different seeds"
    assert not torch.allclose(H1, H2), "H should not be identical with different seeds"


def test_nnsvd_nmtf_initialize_shapes():
    X = torch.rand(12, 9)
    k1 = 6
    k2 = 4
    W, H = nnsvd_nmtf_initialize(X, k1, k2)
    assert W.shape == (12, 6), "W does not have the expected shape"
    assert H.shape == (9, 4), "H does not have the expected shape"


def test_nnsvd_nmtf_initialize_non_negativity():
    X = torch.rand(12, 8)
    k1 = 5
    k2 = 3
    W, H = nnsvd_nmtf_initialize(X, k1, k2)
    assert torch.all(W >= 0), "W contains negative values"
    assert torch.all(H >= 0), "H contains negative values"


def test_nnsvd_nmtf_initialize_reproducibility():
    X = torch.rand(15, 10)
    k1 = 7
    k2 = 5
    seed = 42
    W1, H1 = nnsvd_nmtf_initialize(X, k1, k2, seed=seed)
    W2, H2 = nnsvd_nmtf_initialize(X, k1, k2, seed=seed)
    assert torch.allclose(W1, W2), "W is not reproducible with the same seed"
    assert torch.allclose(H1, H2), "H is not reproducible with the same seed"


def test_nnsvd_nmtf_initialize_different_seed():
    X = torch.rand(15, 10)
    k1 = 7
    k2 = 5
    seed1 = 42
    seed2 = 123
    W1, H1 = nnsvd_nmtf_initialize(X, k1, k2, seed=seed1)
    W2, H2 = nnsvd_nmtf_initialize(X, k1, k2, seed=seed2)
    assert not torch.allclose(W1, W2), "W should not be identical with different seeds"
    assert not torch.allclose(H1, H2), "H should not be identical with different seeds"

def test_nnsvd_nmtf_initialize_rank():
    X = torch.rand(10, 8)
    k1 = 3
    k2 = 4
    W, H = nnsvd_nmtf_initialize(X, k1, k2)
    assert torch.linalg.matrix_rank(W) <= k1, "Rank of W exceeds k"
    assert torch.linalg.matrix_rank(H) <= k2, "Rank of H exceeds k"






