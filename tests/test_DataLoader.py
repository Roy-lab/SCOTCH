import anndata
import numpy as np
import pandas as pd
import pytest
import torch
from DataLoader import DataLoader


@pytest.fixture
def dataloader():
    return DataLoader(verbose=False)


def test_from_text_valid_file(dataloader, tmp_path):
    data = "1\t2\t3\n4\t5\t6\n7\t8\t9"
    file = tmp_path / "test_data.txt"
    file.write_text(data)

    tensor, shape = dataloader.from_text(str(file))
    assert isinstance(tensor, torch.Tensor), "Returned object is not a torch.Tensor"
    assert tensor.shape == (3, 3), "Tensor shape is not as expected"
    np.testing.assert_array_equal(tensor.numpy(), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


def test_from_pt_valid_file(dataloader, tmp_path):
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    file = tmp_path / "test_data.pt"
    torch.save(tensor, file)

    loaded_tensor, shape = dataloader.from_pt(str(file))
    assert isinstance(loaded_tensor, torch.Tensor), "Returned object is not a torch.Tensor"
    assert loaded_tensor.shape == (2, 2), "Tensor shape is not as expected"
    torch.testing.assert_allclose(loaded_tensor, tensor)


def test_from_h5ad_valid_file(dataloader, tmp_path):
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    adata = anndata.AnnData(X=x)
    file = tmp_path / "test_data.h5ad"
    adata.write_h5ad(file)

    result, shape = dataloader.from_h5ad(str(file))
    assert isinstance(result, anndata.AnnData), "Returned object is not an AnnData instance"
    assert result.shape == (2, 2), "AnnData shape is not as expected"
    np.testing.assert_array_equal(result.X, x)


def test_from_text_invalid_file(dataloader):
    with pytest.raises(FileNotFoundError):
        dataloader.from_text("non_existing_file.txt")


def test_from_pt_invalid_file(dataloader):
    with pytest.raises(FileNotFoundError):
        dataloader.from_pt("non_existing_file.pt")


def test_from_h5ad_invalid_file(dataloader):
    with pytest.raises(FileNotFoundError):
        dataloader.from_h5ad("non_existing_file.h5ad")
