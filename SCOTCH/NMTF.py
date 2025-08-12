from bdb import effective

import torch
import time
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as GridSpec
from matplotlib.colors import ListedColormap
import imageio.v2 as imageio
from statsmodels.tools.sm_exceptions import ValueWarning

import SCOTCH.initialize as initialize

from torchmetrics.classification import MulticlassJaccardIndex

import scipy.cluster.hierarchy as sch


class NMTF:
    """
    Base class for NMTF model. Provides minimal support functionality and returns factorized matrices.

    :param k1: Number of components for U matrix. (Default: 2)
    :type k1: int

    :param k2: Number of components for V matrix. (Default: 2)
    :type k2: int

    :param verbose: If True, displays progress messages. (Default: True)
    :type verbose: bool, optional

    :param max_iter: Maximum number of iterations for optimization. (Default: 100)
    :type max_iter: int, optional

    :param seed: Seed for random number generation. (Default: 1001)
    :type seed: int, optional

    :param term_tol: Tolerance level for convergence, defined by relative change of error. (Default: 1e-5)
    :type term_tol: float, optional

    :param max_l_u: Maximum orthogonal regularization term for U matrix. (Default: 0)
    :type max_l_u: float, optional

    :param max_l_v: Maximum orthogonal regularization term for V matrix. (Default: 0)
    :type max_l_v: float, optional

    :param max_a_u: Maximum sparsity constraint for U matrix. (Default: 0)
    :type max_a_u: float, optional

    :param max_a_v: Maximum sparsity constraint for V matrix. (Default: 0)
    :type max_a_v: float, optional

    :param var_lambda: If True, lambda increases based on a sigmoid schedule. (Default: False)
    :type var_lambda: bool, optional

    :param var_alpha: If True, alpha increases based on a sigmoid schedule. (Default: False)
    :type var_alpha: bool, optional

    :param shape_param: Controls the steepness of the sigmoid schedule for both alpha and lambda. (Default: 10)
    :type shape_param: float, optional

    :param mid_epoch_param: Epoch at which the sigmoid scheduling function achieves a mean value. (Default: 5)
    :type mid_epoch_param: int, optional

    :param init_style: Initialization method for factors; either "nnsvd" (default) or "random".
    :type init_style: str, optional

    :param save_clust: If True, saves cluster assignments after every iteration. (Default: False)
    :type save_clust: bool, optional

    :param track_objective: If True, tracks the objective function. (Default: False)
    :type track_objective: bool, optional

    :param kill_factors: If True, halts updates if factor values go to zero. (Default: False)
    :type kill_factors: bool, optional

    :param device: Device for computation, either "cpu" or "cuda". (Default: "cpu")
    :type device: str, optional

    :param out_path: Path to save output files. (Default: '.')
    :type out_path: str, optional

    :rtype: NMTF object
    """

    def __init__(self, verbose=True, max_iter=100, seed=1001, term_tol=1e-5,
                 max_l_u=0, max_l_v=0, max_a_u=0, max_a_v=0, k1=2, k2=2,
                 var_lambda=False, var_alpha=False, shape_param=10, mid_epoch_param=5,
                 init_style="random", save_clust=False, draw_intermediate_graph=False, save_intermediate=False,
                 track_objective=False, kill_factors=False, device="cpu", out_path=None, legacy=False, store_effective = False, dtype = torch.float32):

        # Initialize Parameter space
        self.verbose = verbose
        self.citer = int(0)
        self.maxIter = int(max_iter)
        self.seed = int(seed)
        self.termTol = float(term_tol)
        self.max_lU = float(max_l_u)
        self.max_lV = float(max_l_v)
        self.max_aU = float(max_a_u)
        self.max_aV = float(max_a_v)
        self.lU = float(0)
        self.lV = float(0)
        self.aU = float(0)
        self.aV = float(0)
        self.var_lambda = var_lambda
        self.var_alpha = var_alpha
        self.shape_param = float(shape_param)
        self.mid_epoch_param = int(mid_epoch_param)
        self.k1 = int(k1)
        self.k2 = int(k2)
        self.init_style = str(init_style)
        self.save_clust = save_clust
        self.kill_factors = kill_factors
        self.device = device
        self.track_objective = track_objective
        self.save_intermediate = save_intermediate
        self.draw_intermediate_graph = draw_intermediate_graph
        self.frames = [] if self.draw_intermediate_graph else None
        self.dtype = dtype
        torch.set_default_dtype(self.dtype)

        if out_path is not None:
            self.out_path = str(out_path)
        else:
            self.out_path = '.'
        self.error = []
        # self.error = torch.empty(0)
        torch.manual_seed(self.seed)

        # Initialize Matrices
        self.has_data = False
        self.num_u = 0
        self.num_v = 0
        self.R = torch.empty(0)
        self.X = torch.empty(0)
        self.U = torch.empty(0)
        self.V = torch.empty(0)
        self.S = torch.empty(0)
        self.Q = torch.empty(0)
        self.P = torch.empty(0)

        self.legacy = legacy

        # initialize matrices for storing effective lU and lV terms for norm case.
        self.store_effective = store_effective
        self.E_lU = torch.empty(0)
        self.E_lV = torch.empty(0)
        self.E_aU = torch.empty(0)
        self.E_aV = torch.empty(0)

        # Initialize matrices for saving cluster assignments throughout training
        self.U_assign = torch.empty(0)
        self.V_assign = torch.empty(0)
        self.relative_error = torch.empty(0)
        self.U_JI = torch.empty(0)
        self.V_JI = torch.empty(0)

        # Initialize matrices for tracking objective function parts
        self.reconstruction_error = torch.empty(0)
        self.lU_error = torch.empty(0)
        self.lV_error = torch.empty(0)

    def assign_X_data(self, X):
        """
        Adds a Torch data object to SCOTCH. The input `X` must be a two-dimensional, non-negative Torch tensor.

        :param X: Torch data object to add to SCOTCH. Must be a two-dimensional, non-negative Torch tensor.
        :type X: torch.Tensor
        """
        if not isinstance(X, torch.Tensor):
            raise TypeError('X must be torch tensor object')

        if len(X.shape) != 2:
            raise ValueError('X must be a two dimensional tensor')

        if torch.any(X < 0):
            raise ValueError('X must be non-negative')

        self.X = X
        self.num_u = X.shape[0]
        self.num_v = X.shape[1]
        self.has_data = True

    def _initialize_factors(self):
        """
        Initializes the parameters U, V, S, P, Q, and R based on the specified initialization style.

        :returns: None
        """
        if self.init_style == "random":
            self.U = torch.rand(self.num_u, self.k1, device=self.device)
            self.V = torch.rand(self.k2, self.num_v, device=self.device)
            self.S = self.X.max() * torch.rand((self.k1, self.k2), device=self.device)
            self.P = self.U @ self.S
            self.Q = self.S @ self.V
            self.R = self.X - self.P @ self.V

        elif self.init_style == "nnsvd":
            self.U, self.V = initialize.nnsvd_nmtf_initialize(self.X, self.k1, self.k2, self.seed)
            self.V = torch.t(self.V)

            self.send_to_gpu()

            # Not a real good way of doing this. Start with something random and let's update S first.
            # Perhaps bias toward diagonal.
            self.S = torch.rand((self.k1, self.k2), device=self.device)
            self.P = self.U @ self.S
            self.Q = self.S @ self.V
            self.R = self.X - self.P @ self.V

        else:
            print("Not a valid initialization method.")
            return -999

        return None

    def send_to_gpu(self):
        """
        Sends all tensors to GPU if CUDA is available.

        :returns: None
        """
        if torch.cuda.is_available():
            self.X = self.X.to(self.device)
            self.U = self.U.to(self.device)
            self.V = self.V.to(self.device)
            self.S = self.S.to(self.device)
            self.P = self.P.to(self.device)
            self.Q = self.Q.to(self.device)
            self.R = self.R.to(self.device)
            # self.error.to(self.device)
        return None

    def send_to_cpu(self):
        """
        Sends all tensors to the CPU if CUDA is available.

        :returns: None
        """
        if torch.cuda.is_available():
            self.X = self.X.cpu()
            self.U = self.U.cpu()
            self.V = self.V.cpu()
            self.S = self.S.cpu()
            self.P = self.P.cpu()
            self.Q = self.Q.cpu()
            self.R = self.R.cpu()
            self.U_assign = self.U_assign.cpu()
            self.V_assign = self.V_assign.cpu()
            # self.error.to(self.device)
        return None

    # Update rules
    def _update_kth_block_u(self, k):
        """
        Updates the kth factor of the U matrix.

        :param k: Index of the block to update.
        :type k: int

        :returns: None
        """
        q_norm = torch.linalg.norm(self.Q[k, :]) ** 2
        self.U[:, k] = torch.matmul(self.R, self.Q[k, :]) / q_norm

        # Apply Non-negativity
        self.U[self.U < 0] = 0

    def _update_kth_block_u_unit(self, k):
        """
                Updates the kth factor of the U matrix and normalizes this vector.

                :param k: Index of the block to update.
                :type k: int

                :returns: None
        """
        self.U[:, k] = torch.matmul(self.R, self.Q[k, :])

        # Apply Non-negativity
        self.U[self.U[:, k] < 0, k] = 0

        # Normalize to unit length
        self.U[:, k] = self.U[:, k] / torch.linalg.norm(self.U[:])

    def _apply_orthog_u(self, k):
        """
        Applies the orthogonal regularization term to the kth factor of the U matrix.

        :param k: Index of the block to update.
        :type k: int

        :returns: None
        """
        q_norm = torch.linalg.norm(self.Q[k, :]) ** 2
        # Orthogonality term
        if self.lU > 0:
            self.U[:, k] = self.U[:, k] - self.lU * torch.sum(self.U[:, [x for x in range(self.k1) if x not in [k]]],
                                                              dim=1) / q_norm
        # Apply Non-negativity
        self.U[self.U < 0] = 0

    def _apply_orthog_u_unit(self, k):
        """
               Applies the orthogonal regularization term to the kth factor of the U matrix. Assumes unit norm and uses lambda* for regularization.

               :param k: Index of the block to update.
               :type k: int

               :returns: None
               """


        if self.lU > 0:
            beta = torch.sum(self.U[:, [x for x in range(self.k1) if x not in [k]]], dim=1)
            beta_norm = torch.linalg.norm(beta)
            beta = beta / beta_norm
            self.U[:, k] = self.U[:, k] - self.lU * beta
            # apply Non-negativity
            if self.store_effective:
                self.E_lU[self.citer, k] = (torch.linalg.norm(self.Q[k, :]) ** 2 / beta_norm) * self.lU
        self.U[self.U[:, k] < 0, k] = 0
        self.U[:, k] = self.U[:, k] / torch.linalg.norm(self.U[:, k])
        return None

    def _apply_sparsity_u(self, k):
        """
        Applies the sparsity regularization term to the kth factor of the U matrix.

        :param k: Index of the kth factor.
        :type k: int

        :returns: None
        """

        q_norm = torch.linalg.norm(self.Q[k, :]) ** 2
        # Sparsity term
        if self.aU > 0:
            self.U[:, k] = self.U[:, k] - self.aU * torch.ones(self.num_u, device=self.device) / q_norm

        # Apply Non-negativity
        self.U[self.U < 0] = 0
        return None

    def _apply_sparsity_u_unit(self, k):
        """
        Applies the sparsity regularization term to the kth factor of the U matrix. Assumes unit norm of U.

        :param k: Index of the kth factor.
        :type k: int

        :returns: None
        """

        if self.aU > 0:
            self.U[:, k] = self.U[:, k] = self.aU * torch.ones(self.num_u, device=self.device)
            if self.store_effective:
                self.E_aU[self.citer, k] = (torch.linalg.norm(self.Q[k, :])** 2) * self.aU

        self.U[self.U[:, k] < 0, k] = 0
        self.U[:, k] = self.U[:, k] / torch.linalg.norm(self.U[:, k])
        return None

    def _enforce_non_zero_u(self, k):
        """
        Enforces non-zero values in column k of self.U. If the sum of the column is zero, it sets all values to 1/num_u.
        If citer is greater than 5 and kill_factors is True, the program exits with an error message.

        :param k: Index of a column in self.U to enforce non-zero values.
        :type k: int

        :returns: None
        """

        # Enforce non-zero
        if torch.sum(self.U[:, k]) == 0 or torch.isnan(self.U[:, k]).any():
            self.U[:, k] = torch.ones(self.num_u)
            self.U[:, k] = self.U[:, k] / torch.linalg.norm(self.U[:, k])
            if self.citer > 5 and self.kill_factors:
                sys.exit("Cell factor killed")
        return None

    def _update_kth_block_v(self, k):
        """
        Updates the kth block of the V matrix.

        :param k: Index of the row of V to update.
        :type k: int

        :returns: None
        """

        p_norm = torch.linalg.norm(self.P[:, k]) ** 2
        self.V[k, :] = torch.matmul(self.P[:, k], self.R) / p_norm

        # Apply Non-negativity
        self.V[self.V < 0] = 0
        return None

    def _update_kth_block_v_unit(self, k):
        """
                Updates the kth block of the V matrix and normalizes the vector to unit length.

                :param k: Index of the row of V to update.
                :type k: int

                :returns: None
                """
        self.V[k, :] = torch.matmul(self.P[:, k], self.R)
        # Apply Non-negativity
        self.V[k, self.V[k, :] < 0] = 0
        # Normalize V
        self.V[k, :] = self.V[k, :] / torch.linalg.norm(self.V[k, :])
        return None

    def _apply_orthog_v(self, k):
        """
        Applies the orthogonal regularization update to the kth factor of V.

        :param k: Index of the column to apply regularization.
        :type k: int

        :returns: None
        """

        p_norm = torch.linalg.norm(self.P[:, k]) ** 2
        # Orthogonality term
        if self.lV > 0:
            self.V[k, :] = self.V[k, :] - self.lV * torch.sum(self.V[[x for x in range(self.k2) if x not in [k]], :],
                                                              dim=0) / p_norm

        # Apply Non-negativity
        self.V[self.V < 0] = 0
        return None

    def _apply_orthog_v_unit(self, k):
        """
        Applies the orthogonal regularization update to the kth factor of V using the lambda* interpretation.

        :param k: Index of the column to apply regularization.
        :type k: int

        :returns: None
        """

        if self.lV > 0:
            beta = torch.sum(self.V[[x for x in range(self.k2) if x not in [k]], :], dim=0)
            beta_norm = torch.linalg.norm(beta)
            beta = beta / beta_norm
            self.V[k, :] = self.V[k, :] - self.lV * beta
            if self.store_effective:
                self.E_lV[self.citer, k] = (torch.linalg.norm(self.P[:, k]) ** 2/ beta_norm) * self.lV

        # Apply Non-negativity
        self.V[k, self.V[k, :] < 0] = 0
        # Normalize
        self.V[k, :] = self.V[k, :] / torch.linalg.norm(self.V[k, :])
        return None

    def _apply_sparsity_v(self, k):
        """
        Applies the sparsity regularization update to the kth factor of V.

        :param k: Index of the column to apply sparsity.
        :type k: int

        :returns: None
        """

        p_norm = torch.linalg.norm(self.P[:, k]) ** 2
        # Sparsity term
        if self.aV > 0:
            self.V[k, :] = self.V[k, :] - self.aV * torch.ones(self.num_v, device=self.device) / p_norm

        # Apply Non-negativity
        self.V[self.V < 0] = 0
        return None

    def _apply_sparsity_v_unit(self, k):
        """
        Applies the sparsity update to the kth factor of V using the lambda* interpretation.

        :param k: Index of the column to apply sparsity.
        :type k: int

        :returns: None
        """

        if self.aV > 0:
            self.V[k, :] = self.V[k, :] - self.aV * torch.ones(self.num_v, device=self.device)

            if self.store_effective:
                self.E_aV[self.citer, k]  = torch.linalg.norm(self.P[:, k]) ** 2 * self.aV

        # Apply Non-negativity
        self.V[k, self.V[k, :] < 0] = 0

        # Normalize V
        self.V[k, :] = self.V[k, :] / torch.linalg.norm(self.V[k, :])
        return None

    def _enforce_non_zero_v(self, k):
        """
        Enforces a non-zero value for the gene factor at index k. If the sum of values of the gene factor row is zero,
        it assigns equal weights to each value.

        :param k: Index of the gene.
        :type k: int

        :returns: None

        If the condition self.citer > 5 and self.kill_factors is True, the program exits with the message
        "Gene factor killed".
        """
        # Enforce non-zero
        if torch.sum(self.V[k, :]) == 0 or torch.isnan(self.V[k, :]).any():
            self.V[k, :] = torch.ones(self.num_v)
            self.V[k, :] = self.V[k, :] / torch.linalg.norm(self.V[k, :])
            if self.citer > 5 and self.kill_factors:
                sys.exit("Gene factor killed")
        return None

    def _update_ith_jth_of_s(self, i, j):
        """
        Updates each cell (i, j) of the S (sharing) matrix.

        :param i: Row index of the S matrix to update.
        :type i: int

        :param j: Column index of the S matrix to update.
        :type j: int

        :returns: None
        """

        u_norm = torch.linalg.norm(self.U[:, i]) ** 2
        v_norm = torch.linalg.norm(self.V[j, :]) ** 2
        val = torch.matmul(torch.matmul(self.U[:, i], self.R), self.V[j, :]) / (u_norm * v_norm)
        self.S[i, j] = val if val > 0 else 0
        return None

    # Update the residuals
    def _update_P(self):
        """
        Updates the P matrix (U * S). The P matrix must be updated before refining V.

        :returns: None
        """

        self.P = self.U @ self.S
        return None

    def _update_Q(self):
        """
                Updates the Q matrix (S * V). The Q matrix must be updated before refining U.

                :returns: None
                """
        self.Q = self.S @ self.V
        return None

    # Scaling functions
    def _normalize_and_scale_u(self):
        """
        Normalizes U matrix factors to 1. The scale factor is shifted to the S matrix (i, j) terms.
        This step is required before applying orthogonal regularization.

        :returns: None
        """

        for idx in range(self.k1):
            u_norm = torch.linalg.norm(self.U[:, idx])
            self.U[:, idx] = self.U[:, idx] / u_norm
            self.S[idx, :] = self.S[idx, :] * u_norm

        return None

    def _normalize_and_scale_v(self):
        """
        Normalizes V matrix factors to 1. The scale factor is shifted to the S matrix (i, j) terms.
        This step is required before applying orthogonal regularization.

        :returns: None
        """

        for idx in range(self.k2):
            v_norm = torch.linalg.norm(self.V[idx, :])
            self.V[idx, :] = self.V[idx, :] / v_norm
            self.S[:, idx] = self.S[:, idx] * v_norm

        return None

    # Update objectives
    def _calculate_objective(self):
        """
                Computes the objective function value based on the current state. Adds regularization parameter terms as necessary.

                :returns: None
                """
        # Compute reconstruction error
        error = torch.linalg.norm(self.R, ord='fro').item() ** 2
        self.reconstruction_error[:, self.citer] = error

        # Compute lU component
        if self.lU > 0:
            overlap = (torch.transpose(self.U, 0, 1) @ self.U)
            overlap = overlap - torch.diag_embed(torch.diag(overlap))
            lU_reg = self.lU / 2 * torch.norm(overlap, p=1).item()
            self.lU_error[:, self.citer] = torch.norm(overlap, p=1).item()
        else:
            lU_reg = 0
            self.lU_error[:, self.citer] = 0

        # Compute lV component
        if self.lV > 0:
            overlap = self.V @ torch.transpose(self.V, 0, 1)
            overlap = overlap - torch.diag_embed(torch.diag(overlap))
            lV_reg = self.lV / 2 * torch.norm(overlap, p=1).item()
            self.lV_error[:, self.citer] = torch.norm(overlap, p=1).item()
        else:
            lV_reg = 0
            self.lV_error[:, self.citer] = 0

        # Compute aU component
        if self.aU > 0:
            aU_reg = self.aU / 2 * torch.sum(self.U).item()
            self.aU_error[:, self.citer] = torch.sum(self.U).item()
        else:
            aU_reg = 0
            self.aU_error[:, self.citer] = 0

        # Compute aV component
        if self.aU > 0:
            aV_reg = self.aV / 2 * torch.sum(self.V).item()
            self.aV_error[:, self.citer] = torch.sum(self.V).item()
        else:
            aV_reg = 0
            self.aV_error[:, self.citer] = 0

        # Compute error
        self.error[:, self.citer] = error + lU_reg + lV_reg + aU_reg + aV_reg
        if self.citer > 0:
            cur_error = self.error[:, self.citer]
            prev_error = self.error[:, self.citer - 1]
            self.relative_error[:, self.citer] = ((prev_error - cur_error) / prev_error)
        else:
            self.relative_error[:, self.citer] = float('inf')
        return None

    def _calculate_error_only(self):
        """
        Computes error term corresponding the frobenius norm term of the objective. This measures the inaccuracy of the
        reconstruction of X, given the product U, S, V^T

        :return: None
        """

        # Compute reconstruction error
        error = torch.linalg.norm(self.R, ord='fro').item() ** 2
        self.reconstruction_error[:, self.citer] = error
        self.error[:, self.citer] = error
        self.relative_error = (self.error[:, self.citer - 1] - self.error[:, self.citer]) / (self.error[:, self - 1])
        return None

    def _update_U(self):
        """
                Update the U matrix.

                This method iterates through k1 number of columns and performs the following operations:
                1. Updates the R matrix by adding the outer product of U[:, idx_i] and Q[idx_i, :]
                2. Calls the '_update_kth_block_u' method to update the kth block of U
                3. Updates the R matrix by subtracting the outer product of U[:, idx_i] and Q[idx_i, :]

                After iterating through all columns, it performs the following operations on each column:
                1. Applies orthogonal regularization if lU > 0 by calling '_apply_orthog_u'
                2. Applies sparsity control if aU > 0 by calling '_apply_sparsity_u'
                3. Enforces non-zero elements in the column by calling '_enforce_non_zero_u'

                :return: None
                """
        for idx_i in range(self.k1):
            self.R = self.R + torch.outer(self.U[:, idx_i], self.Q[idx_i, :])
            self._update_kth_block_u(idx_i)
            self.R = self.R - torch.outer(self.U[:, idx_i], self.Q[idx_i, :])

        for idx_i in range(self.k1):
            if self.lU > 0:
                self._apply_orthog_u(idx_i)
            if self.aU > 0:
                self._apply_sparsity_u(idx_i)
            self._enforce_non_zero_u(idx_i)
        return None

    def _update_U_unit(self):
        """
               Updates U matrix by iterating over k1 range and performing several operations on it:
               1. Updates R matrix by adding the outer product of the selected U column and Q row.
               2. Calls _update_kth_block_u_unit method for further updates.
               3. Calls _apply_orthog_u_unit method to apply orthogonal constraint.
               4. Calls _apply_sparsity_u_unit method to enforce sparsity constraint.
               5. Calls _enforce_non_zero_u method to ensure non-zero values in U matrix.
               6. Finally, updates R matrix by subtracting the outer product of the selected U column and Q row.

               :return: None
               """
        for idx_i in range(self.k1):
            self.R = self.R + torch.outer(self.U[:, idx_i], self.Q[idx_i, :])
            self._update_kth_block_u_unit(idx_i)
            self._apply_orthog_u_unit(idx_i)
            self._apply_sparsity_u_unit(idx_i)
            self._enforce_non_zero_u(idx_i)
            self.R = self.R - torch.outer(self.U[:, idx_i], self.Q[idx_i, :])
        return None

    def _update_V(self):
        """
                Update the V matrix.

                This method iterates through k2 number of rows and performs the following operations:
                1. Updates the R matrix by adding the outer product of P[:, idx_j] and V[idx_j, :]
                2. Calls the '_update_kth_block_v' method to update the kth block of V
                3. Updates the R matrix by subtracting the outer product of P[:, idx_j] and V[idx_j, :]

                After iterating through all rows, it performs the following operations on each row:
                1. Applies orthog reg if lV > 0 by calling '_apply_orthog_v'
                2. Applies sparsity control if aV > 0 by calling '_apply_sparsity_v'
                3. Enforces non-zero elements in the row by calling '_enforce_non_zero_v'

                Returns:
                    None
                """
        for idx_j in range(self.k2):
            self.R = self.R + torch.outer(self.P[:, idx_j], self.V[idx_j, :])
            self._update_kth_block_v(idx_j)
            self.R = self.R - torch.outer(self.P[:, idx_j], self.V[idx_j, :])

        for idx_j in range(self.k2):
            if self.lV > 0:
                self._apply_orthog_v(idx_j)
            if self.aV > 0:
                self._apply_sparsity_v(idx_j)
            self._enforce_non_zero_v(idx_j)
        return None

    def _update_V_unit(self):
        """
                Updates the V matrix by iterating over k2 range and performing several operations on it:
                1. Updates the R matrix by adding the outer product of P[:, idx_j] and V[idx_j, :].
                2. Calls the '_update_kth_block_v_unit' method for further updates.
                3. Calls the '_apply_orthog_v_unit' method to apply orthogonal constraint.
                4. Calls the '_apply_sparsity_v_unit' method to enforce sparsity constraint.
                5. Calls the '_enforce_non_zero_v' method to ensure non-zero values in V matrix.
                6. Finally, updates the R matrix by subtracting the outer product of P[:, idx_j] and V[idx_j, :].

                Returns:
                    None
                """
        for idx_j in range(self.k2):
            self.R = self.R + torch.outer(self.P[:, idx_j], self.V[idx_j, :])
            self._update_kth_block_v_unit(idx_j)
            self._apply_orthog_v_unit(idx_j)
            self._apply_sparsity_v_unit(idx_j)
            self._enforce_non_zero_v(idx_j)
            self.R = self.R - torch.outer(self.P[:, idx_j], self.V[idx_j, :])
        return None

    def _updateS(self):
        """
        Updates the matrix R based on the values in matrix S and the matrices U and V.

        This method performs the following operations:
        1. Computes the updated value of matrix R by calculating the product of matrices U, S, and V.
        2. Adjusts the matrix R according to the current state of S, U, and V.

        Returns:
            None
        """

        for idx_i in range(self.k1):
            for idx_j in range(self.k2):
                self.R = self.R + self.S[idx_i, idx_j] * torch.outer(self.U[:, idx_i], self.V[idx_j, :])
                self._update_ith_jth_of_s(idx_i, idx_j)
                self.R = self.R - self.S[idx_i, idx_j] * torch.outer(self.U[:, idx_i], self.V[idx_j, :])
            if torch.sum(self.S[idx_i, :]) == 0:
                self.S[idx_i, :] = 1e-5

        for idx_j in range(self.k2):
            if torch.sum(self.S[:, idx_j]) == 0:
                self.S[:, idx_j] = 1e-5
        return None

    def update(self):
        """
        Defines one update step for the U, V, and S factors.

        This method updates the U, V, and S matrices in one iteration by performing the necessary operations for
        each matrix, including applying regularization, sparsity constraints, and other updates to ensure the
        factors are optimized. It also updates the residual matrix (R) as part of the optimization process.

        Steps:
            1. Updates the U matrix using the '_update_U' method.
            2. Updates the P matrix.
            3. If lU or aU is greater than 0, recalculates the residual matrix R.
            4. Updates the V matrix using the '_update_V' method.
            5. Updates the Q matrix.
            6. Recalculates the residual matrix R if necessary.
            7. Updates the S matrix.
            8. Normalizes and scales U and V matrices.
            9. Re-updates the P and Q matrices.

        Returns:
            None
        """
        self._update_U()
        self._update_P()

        if self.lU > 0 or self.aU > 0:
            self.R = self.X - self.P @ self.V

        self._update_V()
        self._update_Q()

        if self.lU > 0 or self.aU > 0:
            self.R = self.X - self.P @ self.V

        self._updateS()
        self._normalize_and_scale_u()
        self._normalize_and_scale_v()
        self._update_P()
        self._update_Q()
        return None

    def update_unit(self):
        """
        Defines one update step for U, V, and S, using the unit rules.

        This method updates the U, V, and S matrices in one iteration using the unit-based update rules. The update
        steps ensure that regularization, sparsity constraints, and other necessary updates are applied in the manner
        that follows the unit rule approach.

        Steps:
            1. Updates the U matrix using the '_update_U_unit' method.
            2. Updates the P matrix.
            3. Updates the V matrix using the '_update_V_unit' method.
            4. Updates the Q matrix.
            5. Updates the S matrix.
            6. Re-updates the P and Q matrices.

        Returns:
            None
        """

        self._update_U_unit()
        self._update_P()

        self._update_V_unit()
        self._update_Q()

        self._updateS()
        self._update_P()
        self._update_Q()
        return None

    def _determine_reg_state(self):
        """
            Determines the registration state based on the given parameters such as var_lambda, max_lU, sigmoid_schedule,
            mid_epoch_param, shape_param, var_alpha, max_aU, max_aV. Updates the values of lU, lV, aU, aV accordingly.

            Steps:
            1. Checks the value of `var_lambda` to determine if the regularization parameters lU and lV should be adjusted
               using the sigmoid schedule function or if they should be set to the maximum values.

            2. Checks the value of `var_alpha` to determine if the sparsity parameters aU and aV should be adjusted
               using the sigmoid schedule function or if they should be set to the maximum values.

            Uses instance variables: var_lambda, max_lU, sigmoid_schedule, mid_epoch_param, shape_param, var_alpha, max_aU, max_aV.

            Returns:
                None
        """
        if self.var_lambda:
            self.lU = self.max_lU * self._sigmoid_schedule(self.mid_epoch_param, self.shape_param)
            self.lV = self.max_lV * self._sigmoid_schedule(self.mid_epoch_param, self.shape_param)
        else:
            self.lU = self.max_lU
            self.lV = self.max_lV

        if self.var_alpha:
            self.aU = self.max_aU * self._sigmoid_schedule(self.mid_epoch_param, self.shape_param)
            self.aV = self.max_aV * self._sigmoid_schedule(self.mid_epoch_param, self.shape_param)
        else:
            self.aU = self.max_aU
            self.aV = self.max_aV
        return None


    def visualize_effective_lU(self):
        iterations = torch.arange(self.E_lU.size(0))
        plt.figure(figsize = (10, 6))
        for factor_idx in range(self.E_lU.size(1)):
            plt.plot(iterations.numpy(), self.E_lU[:, factor_idx].numpy(), label = f'Factor {factor_idx}')

        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Effective lU')
        plt.title(f'Relation between Effective lU and lU = {self.lU}')


    def visualize_effective_aU(self):
        iterations = torch.arange(self.E_aU.size(0))
        plt.figure(figsize=(10, 6))
        for factor_idx in range(self.E_aU.size(1)):
            plt.plot(iterations.numpy(), self.E_aU[:, factor_idx].numpy(), label=f'Factor {factor_idx}')

        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Effective aU')
        plt.title(f'Relation between Effective aU and aU = {self.aU}')

    def visualize_effective_lV(self):
        iterations = torch.arange(self.E_lV.size(0))
        plt.figure(figsize=(10, 6))
        for factor_idx in range(self.E_lV.size(1)):
            plt.plot(iterations.numpy(), self.E_lV[:, factor_idx].numpy(), label=f'Factor {factor_idx}')

        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Effective LU')
        plt.title(f'Relation between Effective LU and LU = {self.lU}')

    def visualize_effective_aV(self):
        iterations = torch.arange(self.E_aV.size(0))
        plt.figure(figsize=(10, 6))
        for factor_idx in range(self.E_aV.size(1)):
            plt.plot(iterations.numpy(), self.E_aV[:, factor_idx].numpy(), label=f'Factor {factor_idx}')

        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Effective LU')
        plt.title(f'Relation between Effective LU and LU = {self.lU}')


    def fit(self):
        """
                Fits the data using the optimization algorithm.

                This method executes the necessary steps to fit the model to the data using an optimization algorithm. It begins by
                initializing factors, normalizing, and scaling them, and then updates the S matrix. The NMTF algorithm is then started
                and iterated upon. It tracks the objective function setup and updates the model's factors at each iteration.

                Steps:

                1. Initializes the factors (U, V, and S).
                2. Normalizes and scales the U and V factors.
                3. Updates the S matrix.
                4. Tracks the objective function setup.
                5. Begins the NMTF optimization algorithm.
                6. During each iteration:
                    - Updates U, V, and S using the specified update method (legacy or unit-based).
                    - Calculates the objective value.
                    - Optionally prints detailed information about the iteration, including time, objective value, and reconstruction error.
                    - Optionally saves intermediate values of U, S, and V.
                    - Optionally tracks cluster convergence using the Jaccard Index for both U and V assignments.
                    - Optionally visualizes and saves intermediate graphical representations of the factors.
                7. Stops when the relative error falls below a specified tolerance (termTol).

                Returns:
                    None
        """

        self.citer = 0
        start_time = time.time()
        curr_time = time.time()
        stop_marker = 0

        if self.store_effective:
            self.E_lU = torch.empty(self.maxIter+1, self.k1)
            self.E_lV = torch.empty(self.maxIter+1, self.k2)
            self.E_aU = torch.empty(self.maxIter+1, self.k1)
            self.E_aV = torch.empty(self.maxIter+1, self.k2)

        if self.verbose:
            print("Initializing NMTF factors")
        # Initialize factors
        self._initialize_factors()
        self._normalize_and_scale_u()
        self._normalize_and_scale_v()
        self._updateS()
        self._track_objective_setup()

        U_jaccard = MulticlassJaccardIndex(num_classes=self.k1, average='weighted')
        V_jaccard = MulticlassJaccardIndex(num_classes=self.k2, average='weighted')

        if self.verbose:
            print("Beginning NMTF")

        if self.save_clust:
            self._track_clusters_setup()

        if self.draw_intermediate_graph:
            self.frames = []
            fig = self.visualize_factors()
            fig.canvas.draw_idle()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            self.frames.append(frame)
            plt.close(fig)

        while self.citer != self.maxIter:
            self.citer += 1
            self._determine_reg_state()
            if self.legacy:
                self.update()
            else:
                self.update_unit()

            self._calculate_objective()

            if self.verbose:
                next_time = time.time()
                print(
                    "Iter: {0}\tIter Time: {1:.3f}\tTotal Time: {2:.3f}\tObjective: {3:.3e}\tRelative Delta Objective: {4:.3e}\tReconstruction Error: {5:.3e}".
                    format(self.citer, next_time - curr_time, next_time - start_time,
                           self.error[:, self.citer].item(), self.relative_error[:, self.citer].item(),
                           self.reconstruction_error[:, self.citer].item()))
                curr_time = next_time

            # If we want intermediate values in U S and V
            if self.save_intermediate:
                out_path = f"{self.out_path}/ITER_{self.citer}"
                self.print_USV(out_path)

            # If we want to know about cluster convergence.
            if self.save_clust:
                self.U_assign[:, self.citer] = torch.argmax(self.U, dim=1)
                self.V_assign[:, self.citer] = torch.argmax(self.V, dim=0)
                U_target = self.U_assign[:, self.citer - 1]
                U_predict = self.U_assign[:, self.citer]
                V_target = self.V_assign[:, self.citer - 1]
                V_predict = self.V_assign[:, self.citer]
                self.U_JI[:, self.citer - 1] = U_jaccard(U_target, U_predict).item()
                self.V_JI[:, self.citer - 1] = V_jaccard(V_target, V_predict).item()

            if self.draw_intermediate_graph:
                fig = self.visualize_factors()
                fig.canvas.draw()
                frame = np.array(fig.canvas.renderer.buffer_rgba())
                self.frames.append(frame)
                plt.close(fig)

            if self.termTol > self.relative_error[:, self.citer].item() >= 0:
                stop_marker = stop_marker + 1
                if stop_marker >= 5:
                    break
            else:
                stop_marker = 0
        return None

    def fit_U(self):
        """
                Fits the data using the optimization algorithm.

                This method executes the necessary steps to fit the model to the data using an optimization algorithm. It begins by
                initializing factors, normalizing, and scaling them, and then updates the S matrix. The NMTF algorithm is then started
                and iterated upon. It tracks the objective function setup and updates the model's factors at each iteration.

                Steps:

                1. Initializes the factors (U, V, and S).
                2. Normalizes and scales the U and V factors.
                3. Updates the S matrix.
                4. Tracks the objective function setup.
                5. Begins the NMTF optimization algorithm.
                6. During each iteration:
                    - Updates U, holding V, and S using the specified constant.
                    - Calculates the objective value.
                    - Optionally prints detailed information about the iteration, including time, objective value, and reconstruction error.
                    - Optionally saves intermediate values of U, S, and V.
                    - Optionally tracks cluster convergence using the Jaccard Index for both U and V assignments.
                    - Optionally visualizes and saves intermediate graphical representations of the factors.
                7. Stops when the relative error falls below a specified tolerance (termTol).

                Returns:
                    None
        """

        self.citer = 0
        start_time = time.time()
        curr_time = time.time()
        stop_marker = 0

        if self.verbose:
            print("Initializing NMTF factors")
        # Initialize factors
        self.U = torch.rand(self.num_u, self.k1, device=self.device)
        self._normalize_and_scale_u()
        self.P = self.U @ self.S
        self.Q = self.S @ self.V
        self.R = self.X - self.P @ self.V

        self._track_objective_setup()

        U_jaccard = MulticlassJaccardIndex(num_classes=self.k1, average='weighted')
        V_jaccard = MulticlassJaccardIndex(num_classes=self.k2, average='weighted')

        if self.verbose:
            print("Beginning NMTF")

        if self.save_clust:
            self._track_clusters_setup()

        if self.draw_intermediate_graph:
            self.frames = []
            fig = self.visualize_factors()
            fig.canvas.draw_idle()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            self.frames.append(frame)
            plt.close(fig)

        while self.citer != self.maxIter:
            self.citer += 1
            self._determine_reg_state()
            if self.legacy:
                #
                self.update()
            else:
                self.update_unit()
                #self._updateS()


            self._calculate_objective()

            if self.verbose:
                next_time = time.time()
                print(
                    "Iter: {0}\tIter Time: {1:.3f}\tTotal Time: {2:.3f}\tObjective: {3:.3e}\tRelative Delta Objective: {4:.3e}\tReconstruction Error: {5:.3e}".
                    format(self.citer, next_time - curr_time, next_time - start_time,
                           self.error[:, self.citer].item(), self.relative_error[:, self.citer].item(),
                           self.reconstruction_error[:, self.citer].item()))
                curr_time = next_time

            # If we want intermediate values in U S and V
            if self.save_intermediate:
                out_path = f"{self.out_path}/ITER_{self.citer}"
                self.print_USV(out_path)

            # If we want to know about cluster convergence.
            if self.save_clust:
                self.U_assign[:, self.citer] = torch.argmax(self.U, dim=1)
                self.V_assign[:, self.citer] = torch.argmax(self.V, dim=0)
                U_target = self.U_assign[:, self.citer - 1]
                U_predict = self.U_assign[:, self.citer]
                V_target = self.V_assign[:, self.citer - 1]
                V_predict = self.V_assign[:, self.citer]
                self.U_JI[:, self.citer - 1] = U_jaccard(U_target, U_predict).item()
                self.V_JI[:, self.citer - 1] = V_jaccard(V_target, V_predict).item()

            if self.draw_intermediate_graph:
                fig = self.visualize_factors()
                fig.canvas.draw()
                frame = np.array(fig.canvas.renderer.buffer_rgba())
                self.frames.append(frame)
                plt.close(fig)

            if self.termTol > self.relative_error[:, self.citer].item() >= 0:
                stop_marker = stop_marker + 1
                if stop_marker >= 5:
                    break
            else:
                stop_marker = 0
        return None

    def print_USV(self, file_pre=""):
        """
               Write the lower-dimensional matrices (U, V, and S) to tab-delimited text files.

               This method saves the U, V, and S matrices to text files with names based on the
               provided prefix. The matrices are saved in tab-delimited format and will be named
               `file_pre_U.txt`, `file_pre_V.txt`, and `file_pre_S.txt`.

               Args:
                   file_pre (str): Prefix to append to the file names.

               Returns:
                   None
        """

        if not isinstance(file_pre, str):
            raise TypeError('file_pre must be a string')

        if len(file_pre) > 0:
            if file_pre[-1] != '_':
                file_pre = file_pre + '_'

        U_out = self.U.cpu()
        U_out = pd.DataFrame(U_out.numpy())
        U_out.to_csv(self.out_path + '/' + file_pre + "U.txt", sep='\t', header=False, index=False)

        V_out = self.V.cpu()
        V_out = torch.transpose(V_out, 0, 1)
        V_out = pd.DataFrame(V_out.numpy())
        V_out.to_csv(self.out_path + '/' + file_pre + "V.txt", sep="\t", header=False, index=False)

        S_out = self.S.cpu()
        S_out = pd.DataFrame(S_out.numpy())
        S_out.to_csv(self.out_path + '/' + file_pre + "S.txt", sep="\t", header=False, index=False)
        return None

    def print_output(self, out_path):
        """
        Write output files related to the factorization and clustering results.

        This method writes multiple output files, including the lower-dimensional matrices (U, S, V),
        terms associated with the objective function (e.g., reconstruction error, lambda regularization terms),
        and the assignment of U and V at every iteration. It also tracks the stepwise convergence of cluster assignments.

        The output files include:
            - `reconstruction_error.txt`: The reconstruction error over iterations.
            - `lU_error.txt`: The lambda regularization error for U.
            - `lV_error.txt`: The lambda regularization error for V.
            - `relative_error.txt`: The relative error over iterations.
            - `U_assign.txt`: The U assignments at each iteration (if `save_clust` is enabled).
            - `V_assign.txt`: The V assignments at each iteration (if `save_clust` is enabled).
            - `V_JI.txt`: The Jaccard Index for V assignments (if `save_clust` is enabled).
            - `U_JI.txt`: The Jaccard Index for U assignments (if `save_clust` is enabled).

        :param out_path: The path where the output files will be saved.
        :type out_path: str

        :return: None
        """

        self.print_USV(out_path)

        # if self.track_objective:
        reconstruction_error_out = self.reconstruction_error.cpu()
        reconstruction_error_out = pd.DataFrame(reconstruction_error_out.numpy())
        reconstruction_error_out.to_csv(out_path + "/reconstruction_error.txt", sep="\t", header=False, index=False)

        lU_error_out = self.lU_error.cpu()
        lU_error_out = pd.DataFrame(lU_error_out.numpy())
        lU_error_out.to_csv(out_path + '/lU_error.txt', sep='\t', header=False, index=False)

        lV_error_out = self.lV_error.cpu()
        lV_error_out = pd.DataFrame(lV_error_out.numpy())
        lV_error_out.to_csv(out_path + "/lV_error.txt", sep='\t', header=False, index=False)

        aU_error_out = self.aU_error.cpu()
        aU_error_out = pd.DataFrame(aU_error_out.numpy())
        aU_error_out.to_csv(out_path + "/aU_error.txt", sep='\t', header=False, index=False)

        aV_error_out = self.aV_error.cpu()
        aV_error_out = pd.DataFrame(aV_error_out.numpy())
        aV_error_out.to_csv(out_path + "/aV_error.txt", sep='\t', header=False, index=False)

        if self.store_effective:
            effective_lU_out = self.E_lU
            effective_lU_out = pd.DataFrame(effective_lU_out.numpy())
            effective_lU_out.to_csv(out_path + "/effective_lU.txt", sep='\t', header=False, index=False)

            effective_lV_out = self.E_lV
            effective_lV_out = pd.DataFrame(effective_lV_out.numpy())
            effective_lV_out.to_csv(out_path + "/effective_lV.txt", sep='\t', header=False, index=False)

            effective_aU_out = self.E_aU
            effective_aU_out = pd.DataFrame(effective_aU_out.numpy())
            effective_aU_out.to_csv(out_path + "/effective_aU.txt", sep='\t', header=False, index=False)

            effective_aV_out = self.E_aV
            effective_aV_out = pd.DataFrame(effective_aV_out.numpy())
            effective_aV_out.to_csv(out_path + "/effective_aV.txt", sep='\t', header=False, index=False)

        if self.save_clust:
            U_test_out = self.U_assign.cpu()
            U_test_out = pd.DataFrame(U_test_out.numpy())
            U_test_out.to_csv(out_path + "/U_assign.txt", sep='\t', header=False, index=False)

            V_test_out = self.V_assign.cpu()
            V_test_out = pd.DataFrame(V_test_out.numpy())
            V_test_out.to_csv(out_path + "/V_assign.txt", sep='\t', header=False, index=False)

            V_JI_out = self.V_JI.cpu()
            V_JI_out = pd.DataFrame(V_JI_out.numpy())
            V_JI_out.to_csv(out_path + "/V_JI.txt", sep='\t', header=False, index=False)

            U_JI_out = self.U_JI.cpu()
            U_JI_out = pd.DataFrame(U_JI_out.numpy())
            U_JI_out.to_csv(out_path + "/U_JI.txt", sep='\t', header=False, index=False)

        relative_error_out = self.relative_error.cpu()
        relative_error_out = pd.DataFrame(relative_error_out.numpy())
        relative_error_out.to_csv(out_path + "/relative_error.txt", sep='\t', header=False, index=False)

    def _track_objective_setup(self):
        """
        Initialize and track the objective values for the algorithm's error terms across iterations.

        This method sets up tensors to store the reconstruction error, U regularization error, V regularization error,
        relative error, and overall error for each iteration of the algorithm. It then calls `calculate_objective`
        to compute the initial objective values.

        Attributes:
            reconstruction_error (torch.Tensor): Stores the reconstruction error at each iteration.
            lU_error (torch.Tensor): Stores the U regularization error at each iteration.
            lV_error (torch.Tensor): Stores the V regularization error at each iteration.
            relative_error (torch.Tensor): Stores the relative error at each iteration.
            error (torch.Tensor): Stores the overall error at each iteration.

        :return: None
        """
        self.reconstruction_error = torch.zeros(size=[1, self.maxIter + 1])
        self.lU_error = torch.zeros(size=[1, self.maxIter + 1])
        self.lV_error = torch.zeros(size=[1, self.maxIter + 1])
        self.aU_error = torch.zeros(size=[1, self.maxIter + 1])
        self.aV_error = torch.zeros(size=[1, self.maxIter + 1])
        self.relative_error = torch.zeros(size=[1, self.maxIter + 1])
        self.error = torch.zeros(size=[1, self.maxIter + 1])
        self._calculate_objective()

    def _track_clusters_setup(self):
        """
        Initialize the necessary tensors for tracking clusters setup including U_assign, V_assign, U_JI, V_JI.
        Set the initial values for U_JI and V_JI as infinity.
        """
        self.U_assign = torch.zeros(size=[self.num_u, self.maxIter + 1])
        self.V_assign = torch.zeros(size=[self.num_v, self.maxIter + 1])
        self.U_JI = torch.zeros(size=[self.num_u, self.maxIter + 1])
        self.V_JI = torch.zeros(size=[self.num_v, self.maxIter + 1])
        self.U_JI[:, 0] = float('inf')
        self.V_JI[:, 0] = float('inf')

    def save_cluster(self):
        """
        Save cluster assignments and errors for each iteration of the algorithm.

        This method initializes tensors to store the cluster assignments for both U and V matrices
        at each iteration of the algorithm. It also initializes tensors for the Jaccard Index (JI)
        for both U and V and tracks the relative error over iterations.

        Steps:
        1. Initializes tensors for storing U cluster assignments (`U_assign`) and Jaccard Index (`U_JI`).
        2. Initializes tensors for storing V cluster assignments (`V_assign`) and Jaccard Index (`V_JI`).
        3. Initializes tensor to store the relative error over iterations (`relative_error`).


        Returns:
            None
       """
        self.U_assign = torch.zeros(size=[self.num_u, self.maxIter + 1], dtype=torch.uint8)
        self.U_assign[:, 0] = torch.argmax(self.U, dim=1)
        self.U_JI = torch.zeros(size=[self.num_u, self.maxIter])
        self.V_assign = torch.zeros(size=[self.num_v, self.maxIter + 1], dtype=torch.uint8)
        self.V_assign[:, 0] = torch.argmax(self.V, dim=0)
        self.V_JI = torch.zeros(size=[self.num_v, self.maxIter])
        self.relative_error = torch.zeros(size=[1, self.maxIter])

    def assign_cluster(self):
        """
        Assign clusters based on the lower-dimensional embedding matrices U and V.

        This method assigns clusters by taking the `argmax` along the appropriate dimensions of the
        lower-dimensional embedding matrices `U` and `V`. Specifically, it assigns clusters to each
        data point based on the maximum value in the corresponding row of `U` (for the U assignments)
        and the maximum value in the corresponding column of `V` (for the V assignments).

        The cluster assignments are stored in `U_assign` and `V_assign`.

        :return: None
        """
        self.U_assign = torch.argmax(self.U, dim=1)
        self.V_assign = torch.argmax(self.V, dim=0)

    def print_cluster(self, file_pre=''):
        """
        Write the lower-dimensional matrices (U, V, and S) to tab-delimited text files.

        This method saves the U, V, and S matrices to text files with names based on the
        provided prefix. The matrices are saved in tab-delimited format and will be named
       `file_pre_U_assign.txt` and `file_pre_V_assign.txt`.

       Args:
       file_pre (str): Prefix to append to the file names.

       Returns:
           None
       """


        if not isinstance(file_pre, str):
            raise TypeError('file_pre must be a string')

        if len(file_pre) > 0:
            if file_pre[-1] != '_':
                file_pre = file_pre + '_'

        self.assign_cluster() ## For safety. Assign clusters.
        U_out = self.U_assign.cpu()
        U_out = pd.DataFrame(U_out.numpy())
        U_out.to_csv(self.out_path + '/' + file_pre + "U_assign.txt", sep='\t', header=False, index=False)

        V_out = self.V_assign.cpu()
        #V_out = V_out.transpose(0, 1)
        V_out = pd.DataFrame(V_out.numpy())
        V_out.to_csv(self.out_path + '/' + file_pre + "V_assign.txt", sep="\t", header=False, index=False)

    def _sigmoid_schedule(self, mid_iter=5, shape=10.0):
        """
        Generates a sigmoid scheduling function for the lambda U and lambda V regularization parameters.

        This function creates a sigmoid schedule for the regularization parameters `LU` and `LV`, where the values
        of these parameters achieve half of their maximum value at the `mid_iter` (the midpoint iteration). The
        steepness of the curve is controlled by the `shape` parameter.

        :param mid_iter: The midpoint iteration where the schedule reaches half of the maximum value.
        :type mid_iter: int

        :param shape: The shape parameter that controls the steepness of the sigmoid curve.
        :type shape: float

        :return: The value of the sigmoid schedule at the current iteration.
        :rtype: float
        """
        return 1 / (1 + np.exp(-shape * (self.citer - mid_iter)))

    def visualize_factors(self, cmap='viridis', interp='nearest', max_u=1, max_v=1, max_x=1, n_cells = None, n_genes =None):
        """
        This function generates a visual representation of the NMTF factors, allowing users to specify
        the colormap and interpolation method used for image display.

        :param cmap: The colormap to be used for visualization. Default is 'viridis'.
        :type cmap: str, optional

        :param interp: The interpolation method to be used for image display. Default is 'nearest'.
        :type interp: str, optional

        :param max_u: The maximum for color scale. Value between [0, 1] where 1 represents the max value in U.
            Default is 1.
        :type max_u: float, optional

        :param max_v: The maximum for color scale. Value between [0, 1] where 1 represents the max value in V.
            Default is 1.
        :type max_v: float, optional

        :param max_x: The maximum for color scale. Value between [0, 1] where 1 represents the max value in X.
            Default is 1.
        :type max_x: float, optional

        :return: U, S, V  matrix heatmaps with X and product.
        :rtype: matplotlib.figure.Figure

        """

        fig = plt.figure(figsize=(16, 6))
        grids = GridSpec.GridSpec(2, 3, wspace=0.1, width_ratios=(0.2, 0.4, 0.4), height_ratios=(0.3, 0.7))

        U = self.U.clone()
        V = self.V.clone().t()
        S = self.S.clone()

        if n_cells is not None:
            n_cells = min(n_cells, U.shape[0])
            cell_sample_indices = np.random.choice(U.shape[0], n_cells, replace=False)
            U = U[cell_sample_indices, :]

        if n_genes is not None:
            n_genes = min(n_genes, V.shape[1])
            gene_sample_indices = np.random.choice(V.shape[1], n_genes, replace=False)
            V = V[:, gene_sample_indices]

        U_viz = U.detach().numpy()
        U_viz = (U_viz - U_viz.min()) / (U_viz.max() - U_viz.min())
        ax1 = fig.add_subplot(grids[1, 0])
        ax1.imshow(U_viz, aspect="auto", cmap=cmap, interpolation=interp,
                  vmin=0, vmax=max_u)
        ax1.set_axis_off()
        # ax1.set_title("U Matrix")

        # Visualize S matrix
        ax2 = fig.add_subplot(grids[0, 0])
        ax2.imshow(S.t().detach().numpy(), aspect="auto", cmap=cmap, interpolation=interp)
        ax2.set_axis_off()
        # ax2.set_title("S Matrix")

        # Visualize V matrix
        V_viz = V.detach().numpy()
        V_viz = (V_viz - V_viz.min()) / (V_viz.max() - V_viz.min())
        ax3 = fig.add_subplot(grids[0, 1])
        ax3.imshow(V_viz, aspect="auto", cmap=cmap, interpolation=interp,
                   vmin=0, vmax=max_v)
        ax3.set_axis_off()
        # ax3.set_title("V Matrix")

        # Visualize X matrix
        X_est_viz = (U @ S @ V).detach().numpy()
        X_est_viz = (X_est_viz - X_est_viz.min())/(X_est_viz.max() - X_est_viz.min())
        ax4 = fig.add_subplot(grids[1, 1])
        ax4.imshow(X_est_viz, aspect="auto", cmap=cmap,
                   interpolation=interp, vmin=0, vmax=max_x)
        # ax4.set_title("X Matrix")
        ax4.set_axis_off()

        X_temp = self.X.clone()
        # Ensure X_temp is a true copy, in float32
        if not isinstance(X_temp, np.ndarray):
            X_temp = np.array(X_temp, dtype=np.float32, copy=True)

        if n_cells is not None and 'cell_sample_indices' in locals():
            X_temp = X_temp[cell_sample_indices, :]
        if n_genes is not None and 'gene_sample_indices' in locals():
            X_temp = X_temp[:, gene_sample_indices]

        X_temp = (X_temp - X_temp.min()) / (X_temp.max() - X_temp.min())
        ax5 = fig.add_subplot(grids[1, 2])
        ax5.imshow(X_temp, aspect="auto", cmap=cmap, interpolation=interp,
                   vmin=0, vmax=max_x)
        ax5.set_axis_off()
        plt.close(fig)
        return fig

    def visualize_factors_sorted(self, cmap='viridis', interp='nearest', max_u=1, max_v=1, max_x=1, n_cells = None, n_genes = None):
        """
        This function generates a visual representation of the NMTF factors, allowing users to specify
        the colormap and interpolation method used for image display.

        :param cmap: Colormap for the visualization. Default is 'viridis'.
        :type cmap: str, optional

        :param interp: Interpolation method for image display. Default is 'nearest'.
        :type interp: str, optional

        :param max_u: The maximum for color scale. Value between [0, 1] where 1 represents the max value in U. Default is 1.
        :type max_u: float, optional

        :param max_v: The maximum for color scale. Value between [0, 1] where 1 represents the max value in V. Default is 1.
        :type max_v: float, optional

        :param max_x: The maximum for color scale. Value between [0, 1] where 1 represents the max value in X. Default is 1.
        :type max_x: float, optional

        :return: U, S, V  matrix heatmaps with X and product.
        :rtype: matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=(16, 6))
        grids = GridSpec.GridSpec(2, 3, wspace=0.1, width_ratios=(0.2, 0.4, 0.4), height_ratios=(0.3, 0.7))

        U = self.U.clone()
        V = self.V.clone()
        S = self.S.clone()

        if n_cells is not None:
            n_cells = min(n_cells, U.shape[0])
            cell_sample_indices = np.random.choice(U.shape[0], size=n_cells, replace=False)
            U = U[cell_sample_indices, :]

        if n_genes is not None:
            n_genes = min(n_genes, V.shape[1])
            gene_sample_indices = np.random.choice(V.shape[1], size=n_genes, replace=False)
            V = V[:, gene_sample_indices]

        # Generate Sorting for U
        max_U, max_U_idx = U.max(dim=1)
        sorting_criteria = torch.stack([max_U_idx, max_U], dim=1)
        sorted_U_indices = torch.argsort(sorting_criteria, dim=0, stable=True)[:, 0]

        # Generate Sorting for V
        max_V, max_V_idx = V.max(dim=0)
        sorting_criteria = torch.stack([max_V_idx, max_V], dim=1)
        sorted_V_indices = torch.argsort(sorting_criteria, dim=0, stable=True)[:, 0]

        U_viz = U[sorted_U_indices, :].detach().numpy()
        U_viz = (U_viz - U_viz.min()) / (U_viz.max() - U_viz.min())
        ax1 = fig.add_subplot(grids[1, 0])
        ax1.imshow(U_viz, aspect="auto", cmap=cmap, interpolation=interp,
                   vmin=0, vmax=max_u) # set color scale
        ax1.set_axis_off()
        # ax1.set_title("U Matrix")

        # Visualize S matrix
        ax2 = fig.add_subplot(grids[0, 0])
        ax2.imshow(S.t().detach().numpy(), aspect="auto", cmap=cmap, interpolation=interp)
        ax2.set_axis_off()
        # ax2.set_title("S Matrix")

        # Visualize V matrix
        V_viz = V[:, sorted_V_indices].detach().numpy()
        V_viz = (V_viz - V_viz.min())/(V_viz.max() - V_viz.min())
        ax3 = fig.add_subplot(grids[0, 1])
        ax3.imshow(V_viz, aspect="auto", cmap=cmap, interpolation=interp,
                   vmin=0, vmax=max_v) # set color scale
        ax3.set_axis_off()
        # ax3.set_title("V Matrix")

        # Visualize X matrix
        X_est = U @ S @ V
        X_est = X_est[sorted_U_indices, :]
        X_est = X_est[:, sorted_V_indices]
        X_est = (X_est - X_est.min()) / (X_est.max() - X_est.min())
        ax4 = fig.add_subplot(grids[1, 1])
        ax4.imshow(X_est, aspect="auto", cmap=cmap,
                   interpolation=interp, vmin=0, vmax=max_x) # set color scale
        ax4.set_axis_off()

        # ax4.set_title("X Matrix")
        X_temp = self.X.clone()

        if not isinstance(X_temp, np.ndarray):
            X_temp =np.array(X_temp, dtype=np.float32, copy=True)

        if n_cells is not None and 'cell_sample_indices' in locals():
            X_temp = X_temp[cell_sample_indices, :]
        if n_genes is not None and 'gene_sample_indices' in locals():
            X_temp = X_temp[:, gene_sample_indices]


        X_temp = X_temp[sorted_U_indices, :]
        X_temp = X_temp[:, sorted_V_indices]
        X_temp = (X_temp - X_temp.min()) / (X_temp.max() - X_temp.min())
        ax5 = fig.add_subplot(grids[1, 2])
        ax5.imshow(X_temp, aspect="auto", cmap=cmap, interpolation=interp,
                   vmin=0, vmax=max_x) # set color scale
        ax5.set_axis_off()
        plt.close(fig)
        return fig


    def write_gif(self, filename="NMTF_fit.gif", fps=5):
        """
        Save frames of NMTF fit to a GIF figure.

        This method generates and saves a GIF showing the intermediate steps of the NMTF fitting process.
        It is important that the `draw_interm ediate_graph` parameter is set to `True` during the fit to
        capture these frames.

        :param filename: The file name to save the GIF. Default is "NMTF_fit.gif".
        :type filename: str, optional

        :param fps: The desired frames per second for the GIF. Default is 5.
        :type fps: int, optional

        :return: None
        """

        if not isinstance(filename, str):
            raise TypeError("filename must be a str")

        if not isinstance(fps, int):
            raise TypeError("fps must be a int")

        if fps < 0:
            raise ValueError("fps must be positive integer")

        if self.out_path is not None:
            outfile = self.out_path + '/' + filename
        else:
            outfile = "fit.gif"
        print("writing gif to {0}".format(outfile))
        imageio.mimsave(outfile, self.frames, fps=fps, loop=0)

    def recluster_V(self, linkage_type="average", dist_metric='euclidean'):
        """
        Clusters the V matrix using hierarchical clustering, with the specified linkage type and distance metric.
        Afterward, it reapplies SCOTCH based on the cluster representations to remove overly redundant factors from S.

        This process involves performing hierarchical clustering on the V matrix to group similar factors and
        reduce redundancy. SCOTCH is then reapplied to the clustered data to improve the factorization.

        :param linkage_type: The type of linkage method to use for hierarchical clustering.
            Must be one of the following: 'single', 'complete', 'average', or 'ward'.
            Default is 'average'.
        :type linkage_type: str

        :param dist_metric: The distance metric used for calculating pairwise distances in clustering.
            It can be one of the following: 'cosine', 'euclidean', 'city_block', 'chebyshev',
            or an integer for a p-metric. Default is 'euclidean'.
        :type dist_metric: str or int

        :return: None
        """
        # Use a pseudo-Q representation to recluster.
        # This is to make sure we don't have any additive representations in V
        # Row normalize

        if dist_metric == "cosine":
            cosine_sim = torch.nn.functional.cosine_similarity(self.Q.T[:, None, :], self.Q.T[None, :, :], dim=-1)
            D = 1 - cosine_sim
        elif dist_metric == "euclidean":
            D = torch.cdist(self.Q.T, self.Q.T, p=2)
        elif dist_metric == "city_block":
            D = torch.cdist(self.Q.T, self.Q.T, p=1)
        elif dist_metric == "chebyshev":
            D = torch.cdist(self.Q.T, self.Q.T, p=float('inf'))
        elif isinstance(dist_metric, int):
            D = torch.cdist(self.Q.T, self.Q.T, p=dist_metric)
        else:
            raise ValueError("Dist metric is invalid. Value must be one of cosine, euclidean, city_block, chebyshev, or"
                             "an integer for a p metric")

        D = D.cpu().numpy()
        # Cluster V by S representation for each gene
        Z = sch.linkage(D, method=linkage_type)
        k = self.k2
        clusters = sch.fcluster(Z, k, criterion='maxclust')
        clusters = torch.tensor(clusters) - 1
        for i in range(self.V.shape[1]):
            self.V[clusters[i], i] = 1

        # Estimate a new S based on the mean S per cluster
        for i in range(self.k2):
            cluster_indices = (clusters == i).nonzero(as_tuple=True)[0]
            if cluster_indices.numel() > 0:
                self.S[:, i] = self.Q[:, cluster_indices].mean(dim=1)

        # refit
        start_time = time.time()
        curr_time = time.time()
        self.citer = 0
        self._normalize_and_scale_v()
        self._updateS()
        self._track_objective_setup()
        U_jaccard = MulticlassJaccardIndex(num_classes=self.k1, average='weighted')
        V_jaccard = MulticlassJaccardIndex(num_classes=self.k2, average='weighted')

        if self.save_clust:
            self._track_clusters_setup()

        if self.draw_intermediate_graph:
            self.frames = []
            fig = self.visualize_factors()
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            self.frames.append(frame)
            plt.close(fig)

        while self.citer != self.maxIter:
            self.citer += 1
            self._determine_reg_state()
            if self.legacy:
                self.update()
            else:
                self.update_unit()
            self._calculate_objective()

            if self.verbose:
                next_time = time.time()
                print(
                    "Iter: {0}\tIter Time: {1:.3f}\tTotal Time: {2:.3f}\tObjective: {3:.3e}\tRelative Delta Objective: {4:.3e}\tReconstruction Error: {5:.3e}".
                    format(self.citer, next_time - curr_time, next_time - start_time,
                           self.error[:, self.citer].item(), self.relative_error[:, self.citer].item(),
                           self.reconstruction_error[:, self.citer].item()))
                curr_time = next_time

            # If we want intermediate values in U S and V
            if self.save_intermediate:
                out_path = f"{self.out_path}/ITER_{self.citer}"
                self.print_USV(out_path)

            # If we want to know about cluster convergence.
            if self.save_clust:
                self.U_assign[:, self.citer] = torch.argmax(self.U, dim=1)
                self.V_assign[:, self.citer] = torch.argmax(self.V, dim=0)
                U_target = self.U_assign[:, self.citer - 1]
                U_predict = self.U_assign[:, self.citer]
                V_target = self.V_assign[:, self.citer - 1]
                V_predict = self.V_assign[:, self.citer]
                self.U_JI[:, self.citer - 1] = U_jaccard(U_target, U_predict).item()
                self.V_JI[:, self.citer - 1] = V_jaccard(V_target, V_predict).item()

            if self.draw_intermediate_graph:
                fig = self.visualize_factors()
                fig.canvas.draw()
                frame = np.array(fig.canvas.renderer.buffer_rgba())
                self.frames.append(frame)
                plt.close(fig)

            if self.termTol > self.relative_error[:, self.citer].item() >= 0:
                break

    def visualize_clusters(self, cmap='viridis', interp='nearest', max_x=1):
        """
        Visualizes the factors from the NMTF model.

        This function generates a visualization of the factors resulting from the NMTF model. It supports customizing the
        color scheme, interpolation method, and the scaling of the visualization.

        :param factor_name: The name of the factor to visualize (e.g., 'U', 'V').
        :type factor_name: str

        :param cmap: The colormap to use for the visualization. Default is 'viridis'.
        :type cmap: str, optional

        :param interp: The interpolation method for rendering. Default is 'nearest'.
        :type interp: str, optional

        :param max_val: The maximum value for scaling the color map. Default is 1.
        :type max_val: float, optional

        :return: The matplotlib figure object representing the factor visualization.
        :rtype: matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=(8, 6))
        grids = GridSpec.GridSpec(2, 2, hspace=0.1, wspace=0.1, width_ratios=(0.05, 0.95), height_ratios=(0.05, 0.95))

        # Setup safe color palette for U
        n_u_clusters = max(self.U_assign)
        tab_20 = plt.get_cmap('tab20')
        if n_u_clusters > 20:
            raise ValueWarning('Number of U clusters exceeds maximum number of supported by palette (tab20). Repeat '
                               'colors will be used.')
        colors = [tab_20(i % 20) for i in range(n_u_clusters + 1)]
        u_cmap = ListedColormap(colors)

        # Visualize U matrix
        ax1 = fig.add_subplot(grids[1, 0])
        ax1.imshow(self.U_assign.view(-1, 1).detach().numpy(), norm='linear', aspect="auto", cmap=u_cmap,
                   interpolation=interp)
        ax1.set_axis_off()

        n_v_clusters = max(self.V_assign)
        if n_v_clusters > 20:
            raise ValueWarning('Number of V clusters exceeds maximum of supported by palette (tab20). Repeat '
                               "colors will be used.")
        colors = [tab_20(i % 20) for i in range(n_v_clusters + 1)]
        v_cmap = ListedColormap(colors)

        # Visualize V matrix
        ax3 = fig.add_subplot(grids[0, 1])
        ax3.imshow(self.V_assign.view(1, -1).detach().numpy(), norm='linear', aspect="auto", cmap=v_cmap,
                   interpolation=interp)
        ax3.set_axis_off()

        ax4 = fig.add_subplot(grids[1, 1])
        X_viz = self.X.detach().numpy()
        X_viz = (X_viz - X_viz.min()) / (X_viz.max() - X_viz.min())
        ax4.imshow(X_viz, norm='linear', aspect="auto", cmap=cmap, interpolation=interp,
                   vmin=0, vmax=max_x)
        ax4.set_axis_off()
        plt.close(fig)
        return fig

    def visualize_clusters_sorted(self, cmap='viridis', interp='nearest', max_x=1):
        """
            Visualizes the clusters by ordering elements of the matrix based on their cluster assignments.

            The function sorts the elements of the matrix by their cluster order and alternates the color of each
                cluster between grey and black. This approach avoids potential issues with limited color palettes, ensuring
                better visual distinction between clusters.

            :param cmap: The colormap to be used for visualization. Defaults to 'viridis'.
            :type cmap: str, optional
            :param interp: The interpolation method for rendering the image. Defaults to 'nearest'.
            :type interp: str, optional
            :param max_x: The maximum for color scale. Value between [0, 1] where 1 represents the max value in X.  Default is 1.
            :type max_x: int, optional
            :return: Sorted clusters heatmap representation.
            :rtype: matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=(8, 6))
        grids = GridSpec.GridSpec(2, 2, hspace=0.1, wspace=0.1, width_ratios=(0.05, 0.95), height_ratios=(0.05, 0.95))

        # Generate Sorting for U
        max_U, max_U_idx = self.U.max(dim=1)
        sorting_criteria = torch.stack([max_U_idx, max_U], dim=1)
        sorted_U_indices = torch.argsort(sorting_criteria, dim=0, stable=True)[:, 0]

        # Generate Sorting for V
        max_V, max_V_idx = self.V.max(dim=0)
        sorting_criteria = torch.stack([max_V_idx, max_V], dim=1)
        sorted_V_indices = torch.argsort(sorting_criteria, dim=0, stable=True)[:, 0]

        barcode_U = torch.zeros_like(self.U_assign)
        for i, class_value in enumerate(torch.unique(self.U_assign)):
            barcode_U[self.U_assign == class_value] = 0 if i % 2 == 0 else 1.0

        barcode_V = torch.zeros_like(self.V_assign)
        for i, class_value in enumerate(torch.unique(self.V_assign)):
            barcode_V[self.V_assign == class_value] = 0 if i % 2 == 0 else 1.0

        ax1 = fig.add_subplot(grids[1, 0])
        ax1.imshow(barcode_U[sorted_U_indices].view(-1, 1).detach().numpy(), aspect="auto", cmap='gray', vmin=0,
                   vmax=2, interpolation=interp)
        ax1.set_axis_off()

        # Visualize V matrix
        ax3 = fig.add_subplot(grids[0, 1])
        ax3.imshow(barcode_V[sorted_V_indices].view(1, -1).detach().numpy(), aspect="auto", cmap='gray',
                   vmin=0, vmax=2, interpolation=interp)
        ax3.set_axis_off()
        # ax3.set_title("V Matrix")

        X_temp = self.X.detach().numpy()[sorted_U_indices, :][:, sorted_V_indices]
        X_temp = (X_temp - X_temp.min()) / (X_temp.max() - X_temp.min())
        ax5 = fig.add_subplot(grids[1, 1])
        ax5.imshow(X_temp, aspect="auto", cmap=cmap, interpolation=interp,
                   vmin=0, vmax=max_x)
        ax5.set_axis_off()
        plt.close(fig)
        return fig