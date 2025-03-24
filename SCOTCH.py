import torch
import pandas as pd
import anndata
from scipy.sparse import issparse
import numpy as np
from DataLoader import DataLoader
from NMTF import NMTF
import os
import matplotlib.pyplot as plt


class SCOTCH(NMTF):
    """
SCOTCH Class
============

The `SCOTCH` class extends from the `NMTF` class. It has a specific `__init__` method with several input parameters. The only required inputs are `k1` and `k2`.

**__init__ Input Parameters:**

- **k1, k2** (*int*):
  Lower dimension size of `U` and `V`. *(required)*

- **verbose** (*bool*, optional):
  If `True`, prints messages. *(default: True)*

- **max_iter** (*int*, optional):
  Maximum number of iterations. *(default: 100)*

- **seed** (*int*, optional):
  Random seed for initialization. *(default: 1001)*

- **term_tol** (*float*, optional):
  Relative error threshold for convergence. *(default: 1e-5)*

- **max_l_u** (*float*, optional):
  Maximum regularization on `U`. *(default: 0)*

- **max_l_v** (*float*, optional):
  Maximum regularization on `V`. *(default: 0)*

- **max_a_u** (*float*, optional):
  Maximum sparse regularization on `U`. *(default: 0, change at own risk)*

- **max_a_v** (*float*, optional):
  Maximum sparse regularization on `V`. *(default: 0, change at own risk)*

- **var_lambda** (*bool*, optional):
  If `True`, the regularization parameters `l_U` and `l_V` increase to max value using a sigmoid scheduler. Generally set to `False`. *(default: False)*

- **var_alpha** (*bool*, optional):
  If `True`, the regularization parameters `a_U` and `a_V` increase to max value using a sigmoid scheduler. Generally set to `False`. *(default: False)*

- **shape_param** (*float*, optional):
  Controls the rate of increase for `l_U`, `l_V`, `a_U`, and `a_V` when `var_lambda=True`. *(default: 10)*

- **mid_epoch_param** (*int*, optional):
  Sets the epoch where `l_U`, `l_V`, `a_U`, and `a_V` reach half of their max values if `var_lambda=True`. *(default: 5)*

- **init_style** (*str*, optional):
  Initialization method for SCOTCH. Should be either `"random"` or `"nnsvd"`. *(default: "random")*

- **save_clust** (*bool*, optional):
  Whether to save cluster assignments after each epoch. *(default: False)*

- **draw_intermediate_graph** (*bool*, optional):
  If `True`, draws and saves the matrix representation after each epoch. These can be saved as a GIF. *(default: False)*

- **track_objective** (*bool*, deprecated):
  *(default: False)*

- **kill_factors** (*bool*, optional):
  If `True`, SCOTCH will halt updates if any factors in `U` and `V` reach zero. *(default: False)*

- **device** (*str*, optional):
  Specifies the device to run SCOTCH on: `"cpu"` or `"cuda:"`. *(default: "cpu")*

- **out_path** (*str*, optional):
  Directory to save SCOTCH output files. *(default: '.')*
    """
    def __init__(self, k1, k2, verbose=True, max_iter=100, seed=1001, term_tol=1e-5,
                 max_l_u=0, max_l_v=0, max_a_u=0, max_a_v=0, var_lambda=False,
                 var_alpha=False, shape_param=10, mid_epoch_param=5,
                 init_style="random", save_clust=False, draw_intermediate_graph=False, save_intermediate=False,
                 track_objective=False, kill_factors=False, device="cpu", out_path='.'):

        super().__init__(verbose, max_iter, seed, term_tol, max_l_u, max_l_v, max_a_u, max_a_v, k1, k2, var_lambda,
                         var_alpha, shape_param, mid_epoch_param, init_style, save_clust, draw_intermediate_graph,
                         save_intermediate, track_objective, kill_factors, device, out_path)

        self.DataLoader = DataLoader(verbose)

    def add_data_from_file(self, file):
        """
        Loads matrix representation into PyTorch tensor object to run with SCOTCH.

        :param file: The file path to load data from and should have the valid extensions like '.pt', '.txt', or '.h5ad'.
        :type file: str
        """
        if not isinstance(file, str):
            raise TypeError('file must be a string')

        if not os.path.isfile(file):
            raise ValueError('The file does not exist')

        if not os.access(file, os.R_OK):
            raise ValueError('The file is not readable')

        _, file_extension = os.path.splitext(file)
        if file_extension == '.pt':
            self.X, shape = self.DataLoader.from_pt(file)
        elif file_extension == '.txt':
            self.X, shape = self.DataLoader.from_text(file)
        elif file_extension == '.h5ad':
            adata = self.DataLoader.from_h5ad(file)
            self.X, shape = self.add_data_from_adata(adata)
        else:
            raise ValueError("Unsupported file type. Select .pt or .txt or .h5ad")
        self.num_u = shape[0]
        self.num_v = shape[1]
        print("Data loaded successfully. Shape: ", self.num_u, self.num_v)
        return None

    def add_data_from_adata(self, adata):
        """
        Loads data from AnnData object into SCOTCH framework.

        :param adata: anndata.AnnData object to extract data from. Transforms adata.X to PyTorch object.
        :type adata: anndata.AnnData
        """
        if not isinstance(adata, anndata.AnnData):
            raise TypeError("adata must be an AnnData object")

        # Extract the X matrix and covert to a torch tensor
        X = adata.X
        if issparse(X):
            X_coo = X.tocoo()
            values = torch.tensor(X_coo.data)
            indices = torch.tensor(np.vstack((X_coo.row, X_coo.col)))
            X_tensor = torch.sparse_coo_tensor(indices, values, X_coo.shape)
            X_tensor = X_tensor.to_dense()
        else:
            X_tensor = torch.tensor(X)
        self.X = X_tensor
        self.num_u = self.X.shape[0]
        self.num_v = self.X.shape[1]
        return None

    def add_scotch_embeddings_to_adata(self, adata, prefix=""):
        """
        Adds SCOTCH objects to an AnnData object.

        :param prefix: Prefix to add to AnnData objects created by SCOTCH.
        :type prefix: str

        :param adata: The AnnData object to which SCOTCH embeddings will be added.
        :type adata: anndata.AnnData
        """
        if not isinstance(adata, anndata.AnnData):
            raise TypeError("adata must be an AnnData object")

        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        if len(prefix) > 0 and prefix[-1] != '_':
            prefix = prefix + '_'

        adata.obs[prefix + 'cell_clusters'] = pd.Categorical(self.U_assign.detach().numpy())
        adata.var[prefix + "gene_clusters"] = pd.Categorical(self.V_assign.t().detach().numpy())
        adata.obsm[prefix + 'cell_embedding'] = self.U.detach().numpy()
        adata.varm[prefix + 'gene_embedding'] = self.V.t().detach().numpy()
        adata.uns[prefix + 'S_matrix'] = self.S.detach().numpy()
        adata.obsm[prefix + 'P_embedding'] = self.P.detach().numpy()
        adata.varm[prefix + 'Q_embedding'] = self.Q.t().detach().numpy()
        adata.uns[prefix + 'reconstruction_error'] = self.reconstruction_error.detach().numpy()
        adata.uns[prefix + 'error'] = self.error.detach().numpy()
        return adata

    def make_adata_from_scotch(self, prefix=""):
        """
        Create an AnnData object from the given data.

        :param self: The instance of the class containing the data.
        :type self: object

        :param prefix: A string appended to the generated AnnData objects.
        :type prefix: str

        :returns: An AnnData object containing the processed data.
        :rtype: anndata.AnnData
        """
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a str")

        if len(prefix) > 0 and prefix[-1] != '_':
            prefix = prefix + '_'
        X = self.X
        adata = anndata.AnnData(self.X.numpy())
        adata = self.add_scotch_embeddings_to_adata(adata)
        return adata

    def make_top_regulators_list(self, adata, top_k=5):
        cluster_gene_idx = [(i, torch.topk(self.V[i, :], top_k).indices.tolist()) for i in range(self.k2)]
        cluster_gene = []
        for i, indices in cluster_gene_idx:
            top_genes = adata.var_names[indices]
            cluster_gene.append((i, top_genes))

        if self.verbose:
            for c_g in cluster_gene:
                print(f"Gene Cluster {c_g[0]}: {c_g[1]}")
        return cluster_gene

    def write_gene_clusters_to_enrich_analyzer(self, adata, prefix):
        adata.prefix







