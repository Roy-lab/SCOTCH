import torch
import pandas as pd
import anndata
from scipy.sparse import issparse
import numpy as np
from DataLoader import DataLoader
from NMTF import NMTF
import os

class SCOTCH(NMTF):
    """
        This is the SCOTCH class that extends from the NMTF class. It has a specific __init__ method with several input
        parameters. The only required inputs are k1 and k2.

        __init__ Input parameters:
          - k1, k2: lower dimension size of U and V. (required)
          - verbose: If true, prints messages (default is True).
          - max_iter: Maximum number of iterations (default is 100).
          - seed: Random seed for initialization (default is 1001).
          - term_tol: Relative error threshold for convergence (default is 1e-5)
          - max_l_u: Maximum regularization on U. (default is 0)
          - max_l_v: Maximum regularization on V. (default is 0)
          - max_a_u: Maximum sparse regularization U. (default is 0, change at own risk)
          - max_a_v: Maximum sparse regularization V. (default is 0, change at own risk)
          - var_lambda: If True, the regularization parameter l_U and l_V are increased to max value using
            a sigmoid scheduler. Generally should be set to FALSE. (default is False)
         - var_alpha: If True, the regularization parameter a_U and a_V are increased to max value using
            a sigmoid scheduler. Generally should be set to FALSE. (default is False)
         - shape_param : controls rate of  l_U, l_V, a_U and a_V  increase when var_lambda = TRUE. (default is 10)
         - mid_epoch_param:  sets epoch where lU, lV, a_U and a_V achieve max/2, if var_lambda = TRUE. (default is 5)
         - init_style: Initialization method for SCOTCH. Should be "random" or "nnsvd". (default is "random")
         - save_clust: Whether to save clusters assignment after each epoch (default is False)
         - draw_intermediate_graph: If True, will draw matrix representation after each epoch and save them to SCOTCH
             object. These can be printed to a GIF file. (default is False)
         - track_objective: Depreciated (default is False)
         - kill_factors: if True, SCOTCH will halt updates if at any point factors in U and V go to zero
            (default is False).
         - device: which machine to run SCOTCH. set to "cpu" or "cuda:". default is "cpu"
         - out_path: where to save files form SCOTCH. (default is '.')

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
        Loads matrix representation into pytorch tensor object to run with SCOTCH.

        Args:
            file: The file path to load data from and should have the valid extensions like '.pt', '.txt', or '.h5ad'.
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
        Loads data from adata object into SCOTCH framework.
        Args:
            adata: anndata.AnnData object to extract data from. Transform adata.X to pytorch object.
        """
        if not isinstance(adata, anndata.AnnData):
            raise TypeError("adata must be an AnnData object")

        #Extract the X matrix and covert to a torch tensor
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
        Adds SCOTCH objects to an AnnData object
        Args:
            prefix: prefix to add to adata objects created by SCOTCH
            adata: the AnnData object to which Scotch embeddings will be added.
        """
        if not isinstance(adata, anndata.AnnData):
            raise TypeError("adata must be an AnnData object")

        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        if prefix[-1] != '_':
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

        Parameters:
            self: object
                The instance of the class containing the data.

            prefix: string
                A string appended to

        Returns:
            anndata.AnnData
                An AnnData object containing the processed data.
        """
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a str")

        if prefix[-1] != '_':
            prefix = prefix + '_'

        adata = anndata.AnnData(self.X.numpy())
        self.add_scotch_embeddings_to_adata(adata)
        return adata



