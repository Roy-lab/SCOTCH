import pytest
import torch
from NMTF import NMTF


def test_nmtf_initialization_defaults():
    model = NMTF()
    assert model.k1 == 2, "Default k1 should be 2"
    assert model.k2 == 2, "Default k2 should be 2"
    assert model.maxIter == 100, "Default max_iter should be 100"
    assert model.termTol == 1e-5, "Default term_tol should be 1e-5"


def test_nmtf_initialization_custom_values():
    model = NMTF(k1=4, k2=5, max_iter=200, term_tol=1e-4, device="cuda")
    assert model.k1 == 4, "k1 should be set to the provided value"
    assert model.k2 == 5, "k2 should be set to the provided value"
    assert model.maxIter == 200, "max_iter should be set to the provided value"
    assert model.termTol == 1e-4, "term_tol should be set to the provided value"
    assert model.device == "cuda", "Device should be set to cuda"


def test_nmtf_assign_X_data():
    model = NMTF()
    X = torch.rand(10, 8)
    model.assign_X_data(X)
    assert model.X.shape == (10, 8), "Assigned X should retain its original shape"
    assert torch.equal(model.X, X), "Assigned X should match the input tensor"


def test_nmtf_send_to_gpu():
    model = NMTF(device="cuda:0")
    X = torch.rand(10, 8)
    model.assign_X_data(X)
    model.send_to_gpu()
    assert model.X.device.type == "cuda", "X should be moved to GPU when send_to_gpu is called"


def test_nmtf_send_to_cpu():
    model = NMTF(device="cpu")
    X = torch.rand(10, 8).to("cuda:0")
    model.assign_X_data(X)
    model.send_to_cpu()
    assert model.X.device.type == "cpu", "X should be moved to CPU when send_to_cpu is called"


def test_nmtf_fit_initialization():
    model = NMTF(k1=3, k2=4, max_iter=10, init_style="nnsvd")
    X = torch.abs(torch.rand(10, 10))  # Ensure non-negative for factorization
    model.assign_X_data(X)
    model.fit()
    assert hasattr(model, "U"), "U matrix should be initialized during fit"
    assert hasattr(model, "V"), "V matrix should be initialized during fit"
    assert model.U.shape == (10, 3), "U matrix should have correct shape based on k1"
    assert model.V.shape == (4, 10), "V matrix should have correct shape based on k2"

def test_nmtf_fit_initialization_random():
    model = NMTF(k1=3, k2=4, max_iter=10, init_style="random")
    X = torch.abs(torch.rand(10, 10))  # Ensure non-negative for factorization
    model.assign_X_data(X)
    model.fit()
    assert hasattr(model, "U"), "U matrix should be initialized during fit"
    assert hasattr(model, "V"), "V matrix should be initialized during fit"
    assert model.U.shape == (10, 3), "U matrix should have correct shape based on k1"
    assert model.V.shape == (4, 10), "V matrix should have correct shape based on k2"
