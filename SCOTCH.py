import torch
import pandas
import anndata
from scipy.sparse import issparse
import numpy as np
import umap
import matplotlib.pyplot as plt
import matplotlib.gridspec as GridSpec

from DataLoader import DataLoader
from NMTF import NMTF

import os

class SCOTCH(NMTF):
    def __init__(self, k1, k2, verbose=True, max_iter=100, seed=1001, term_tol=1e-5,
                 max_l_u=0, max_l_v=0, max_a_u=0, max_a_v=0, var_lambda=False,
                 var_alpha=False, shape_param=10, mid_epoch_param=5,
                 init_style="nnsvd", save_clust=False,
                 track_objective=False, kill_factors=False, device="cpu", out_path=None):

        super().__init__(verbose, max_iter, seed, term_tol, max_l_u, max_l_v, max_a_u, max_a_v, k1, k2, var_lambda,
                         var_alpha, shape_param, mid_epoch_param, init_style, save_clust,
                         track_objective, kill_factors, device, out_path)

        self.DataLoader = DataLoader(verbose)

    def addDataFromFile(self, file):
        _, file_extension = os.path.splitext(file)
        if file_extension == '.pt':
            self.X = self.DataLoader.from_pt(file)
        elif file_extension == '.txt':
            self.X = self.DataLoader.from_text(file)
        elif file_extension == '.h5ad':
            adata = self.DataLoader.from_h5ad(file)
            self.X = self.addDataFromAData(adata)
        else:
            raise ValueError("Unsupported file type")

    def addDataFromAData(self, adata):
        if not isinstance(adata, anndata.AnnData):
            raise TypeError("adata must be an AnnData object")

        ## Extract the X matrix and covert to a torch tensor
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

    def addScotchEmbeddingsToAnnData(self, adata):
        if not isinstance(adata, anndata.AnnData):
            raise TypeError("adata must be an AnnData object")

        adata.obs['cell_clusters'] = self.U_assign.detach().numpy()
        adata.var["gene_clusters"] = self.V_assign.t().detach().numpy()
        adata.obsm['cell_embedding'] = self.U.detach().numpy()
        adata.varm['gene_embedding'] = self.V.t().detach().numpy()
        adata.uns['S_matrix'] = self.S.detach().numpy()

        return adata

    def visualizeHeatmap(self):
        fig = plt.figure(figsize=(16, 6))
        grids = GridSpec(1, 4, fig, wspace=0.4)

        # Visualize U matrix
        ax1 = fig.add_subplot(grids[0, 0])
        ax1.imshow(self.U.detach().numpy(), aspect="auto", cmap='viridis')
        ax1.set_title("U Matrix")

        # Visualize S matrix
        ax2 = fig.add_subplot(grids[0, 1])
        ax2.imshow(self.S.detach().numpy(), aspect="auto", cmap='viridis')
        ax2.set_title("S Matrix")

        # Visualize V matrix
        ax3 = fig.add_subplot(grids[0, 2])
        ax3.imshow(self.V.detach().numpy(), aspect="auto", cmap='viridis')
        ax3.set_title("V Matrix")

        # Visualize X matrix
        ax4 = fig.add_subplot(grids[0, 3])
        ax4.imshow(self.X.detach().numpy(), aspect="auto", cmap='viridis')
        ax4.set_title("X Matrix")

        plt.show()