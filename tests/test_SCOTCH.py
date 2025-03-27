# File: tests/test_SCOTCH.py

import os

import anndata
import pytest
import torch
from SCOTCH import SCOTCH
import numpy as np


def test_scotch_initialization():
    model = SCOTCH(k1=3, k2=4, verbose=False)
    assert model.k1 == 3, "k1 is not initialized correctly"
    assert model.k2 == 4, "k2 is not initialized correctly"


def test_scotch_add_data_from_adata():
    adata = anndata.AnnData(X=torch.rand(10, 12).numpy())
    model = SCOTCH(k1=2, k2=2, verbose=False)
    model.add_data_from_adata(adata)
    assert torch.equal(model.X, torch.tensor(adata.X)), "Data was not added to the SCOTCH object"
    assert model.X.shape == (10, 12), "Data in SCOTCH has an unexpected shape"
    assert model.num_u == 10
    assert model.num_v == 12

def test_scotch_add_scotch_embeddings_to_adata():
    adata = anndata.AnnData(X=torch.rand(10, 10).numpy())
    model = SCOTCH(k1=2, k2=2, verbose=False)
    model.add_data_from_adata(adata)
    model.fit()
    model.add_scotch_embeddings_to_adata(adata, prefix="test")
    assert "test_U_embedding" in adata.obsm.keys(), "U embedding not added to AnnData object"
    assert "test_V_embedding" in adata.varm.keys(), "V embedding not added to AnnData object"


def test_scotch_make_top_regulators_list():
    adata = anndata.AnnData(X=torch.rand(10, 15).numpy())
    adata.obs["gene_clusters"] = [1, 2] * 5
    model = SCOTCH(k1=2, k2=2, verbose=False)
    model.add_data_from_adata(adata)
    top_regulators = model.make_top_regulators_list(
        adata, gene_cluster_id="gene_clusters", top_k=3
    )
    assert isinstance(top_regulators, list), "Method did not return a list"
    assert len(top_regulators) > 0, "Top regulators list is empty"


def test_scotch_plot_element_count_heatmap():
    adata = anndata.AnnData(X=torch.rand(20, 20).numpy())
    adata.obs['cell_clusters'] = ["Cluster1"] * 10 + ["Cluster2"] * 10
    model = SCOTCH(k1=2, k2=2, verbose=False)
    try:
        model.plot_element_count_heatmap(adata, field='cell_clusters')
    except Exception as e:
        pytest.fail(f"plot_element_count_heatmap raised an exception: {e}")
