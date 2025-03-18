import torch
import pandas as pd
import anndata
from scipy.sparse import issparse
import numpy as np
import matplotlib.pyplot as plt


from DataLoader import DataLoader
from NMTF import NMTF
from torchmetrics.classification import MulticlassJaccardIndex
import time

import imageio.v2 as imageio
import os

class SCOTCH(NMTF):
    def __init__(self, k1, k2, verbose=True, max_iter=100, seed=1001, term_tol=1e-5,
                 max_l_u=0, max_l_v=0, max_a_u=0, max_a_v=0, var_lambda=False,
                 var_alpha=False, shape_param=10, mid_epoch_param=5,
                 init_style="nnsvd", save_clust=False, draw_intermediate_graph=False, save_intermediate=False,
                 track_objective=False, kill_factors=False, device="cpu", out_path=None):

        super().__init__(verbose, max_iter, seed, term_tol, max_l_u, max_l_v, max_a_u, max_a_v, k1, k2, var_lambda,
                         var_alpha, shape_param, mid_epoch_param, init_style, save_clust, draw_intermediate_graph,
                         save_intermediate, track_objective, kill_factors, device, out_path)

        self.DataLoader = DataLoader(verbose)

    def addDataFromFile(self, file):
        _, file_extension = os.path.splitext(file)
        if file_extension == '.pt':
            self.X, shape = self.DataLoader.from_pt(file)
        elif file_extension == '.txt':
            self.X, shape = self.DataLoader.from_text(file)
        elif file_extension == '.h5ad':
            adata = self.DataLoader.from_h5ad(file)
            self.X, shape = self.addDataFromAData(adata)
        else:
            raise ValueError("Unsupported file type")
        self.num_u = shape[0]
        self.num_v = shape[1]
        print("Data loaded successfully. Shape: ", self.num_u, self.num_v)

    def fitVideo(self):
        """
        Optimizes the selected NMTF model. Performs cluster assignment and update for U and V.
        :return: None
        """
        print("Here 1")
        frames = []
        start_time = time.time()
        curr_time = time.time()

        # Initialize factors
        self._initialize_factors()
        self._normalize_and_scale_u()
        self._normalize_and_scale_v()
        self._updateS()

        print("Here 2")

        self.track_objective_setup()

        fig = self.visualizeFactors(cmap='Reds')
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        frames.append(frame)
        plt.close(fig)
        self.citer = 0

        #U_jaccard = MulticlassJaccardIndex(num_classes=self.k1, average='weighted')
        #V_jaccard = MulticlassJaccardIndex(num_classes=self.k2, average='weighted')
        while self.citer != self.maxIter:
            self.citer += 1

            # Update
            if self.legacy:
                self.update()
                self.calculate_objective()
            else:
                self.update_unit()
                self.calculate_error_only()
            slope = (self.error[-2] - self.error[-1]) / self.error[-2]

            #self.relative_error[self.citer - 1] = slope

            if self.verbose:
                next_time = time.time()
                print("Iter: {0}\tIter Time: {1:.3f}\tTotal Time: {2:.3f}\tError: {3:.3e}\tRelative Delta "
                      "Residual: {4:.3e}".
                      format(self.citer, next_time - curr_time, next_time - start_time, self.error[-1], slope))
                curr_time = next_time

            fig = self.visualizeFactors(cmap='Reds')
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            frames.append(frame)
            plt.close(fig)

            if self.termTol > slope > 0:
                break

        self.send_to_cpu()
        if self.out_path is not None:
            outfile = self.out_path + "/fit.gif"
        else:
            outfile = "fit.gif"
        print("writing gif to {0}".format(outfile))
        imageio.mimsave(outfile, frames, fps=10)

    def addDataFromAData(self, adata):
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

    def addScotchEmbeddingsToAnnData(self, adata):
        if not isinstance(adata, anndata.AnnData):
            raise TypeError("adata must be an AnnData object")

        adata.obs['cell_clusters'] = pd.Categorical(self.U_assign.detach().numpy())
        adata.var["gene_clusters"] = pd.Categorical(self.V_assign.t().detach().numpy())
        adata.obsm['cell_embedding'] = self.U.detach().numpy()
        adata.varm['gene_embedding'] = self.V.t().detach().numpy()
        adata.uns['S_matrix'] = self.S.detach().numpy()
        adata.obsm['P_embedding'] = self.P.detach().numpy()
        adata.varm['Q_embedding'] = self.Q.t().detach().numpy()
        return adata

    def addScotchEmbeddingsToAnnData(self, adata):
        if not isinstance(adata, anndata.AnnData):
            raise TypeError("adata must be an AnnData object")

        adata.obs['cell_clusters'] = pd.Categorical(self.U_assign.detach().numpy())
        adata.var["gene_clusters"] = pd.Categorical(self.V_assign.t().detach().numpy())
        adata.obsm['cell_embedding'] = self.U.detach().numpy()
        adata.varm['gene_embedding'] = self.V.t().detach().numpy()
        adata.uns['S_matrix'] = self.S.detach().numpy()
        adata.obsm['P_embedding'] = self.P.detach().numpy()
        adata.varm['Q_embedding'] = self.Q.t().detach().numpy()
        return adata


    def makeAdataFromScotch(self):
        adata = anndata.AnnData(self.X.numpy())
        adata.obs['cell_clusters'] = pd.Categorical(self.U_assign.detach().numpy())
        adata.var["gene_clusters"] = pd.Categorical(self.V_assign.t().detach().numpy())
        adata.obsm['cell_embedding'] = self.U.detach().numpy()
        adata.varm['gene_embedding'] = self.V.t().detach().numpy()
        adata.uns['S_matrix'] = self.S.detach().numpy()
        adata.obsm['P_embedding'] = self.P.detach().numpy()
        adata.varm['Q_embedding'] = self.Q.t().detach().numpy()
        return adata



