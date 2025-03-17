import anndata
import torch
import os
import time
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as GridSpec
import imageio.v2 as imageio
import initialize
from torchmetrics.classification import MulticlassJaccardIndex

import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd


class NMTF:
    """
    Base class for NMTF model. Minimal support functionality. Returns matrices.


    :rtype: NMTF object
    :param verbose: Sets whether to display progress messages
    :param max_iter: Maximum number of iterations for optimization. Default: 100
    :param seed: Seed for random number generation Default 1001
    :param term_tol: Tolerance level for convergence. Defined by relative change of error. Default 1e-5
    :param max_l_u: Max orthogonal regularization  term for U matrix. Default 0
    :param max_l_v: Max orthogonal regularization  term for V matrix. Default 0
    :param max_a_u: Max sparsity constraint for U matrix. Default 0.
    :param max_a_v: Max sparsity constraint for V matrix. Default 0.
    :param k1: Number of components for U matrix. Default 2.
    :param k2: Number of components for V matrix. Default 2.
    :param var_lambda: If set to True, allow for lambda to increase based on sigmoid schedule.
    :param var_alpha: If set to True, allow for alpha to increase as a function on sigmoid schedule.
    :param shape_param: shape factor that controls steepness of sigmoid schedule for both alpha and lambda. Default 10.
    :param mid_epoch_param: The epoch at which the sigmoid scheduling function achieves a mean value, Default = 5,
    :param init_style: Initialization method for factors. Set to either nnsvd (default) or random.
    :param save_clust: Flag to save cluster assignments after every iteration.
    :param track_objective: Flag to track objective function.
    :param kill_factors: Flag to stop update if factor all factor values go to zero.
    :param device: Device to run computation on (cpu/gpu)
    :param out_path: Path to save output files
    """

    def __init__(self, verbose=True, max_iter=100, seed=1001, term_tol=1e-5,
                 l_u=0, l_v=0, a_u=0, a_v=0, k1=2, k2=2,
                 var_lambda=False, var_alpha=False, shape_param=10, mid_epoch_param=5,
                 init_style="random", save_clust=False,
                 track_objective=False, kill_factors=False, device="cpu", out_path=None, legacy=False):

        # Initialize Parameter space
        self.verbose = verbose
        self.citer = int(0)
        self.maxIter = int(max_iter)
        self.seed = int(seed)
        self.termTol = float(term_tol)
        self.max_lU = float(l_u)
        self.max_lV = float(l_v)
        self.max_aU = float(a_u)
        self.max_aV = float(a_v)
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
        self.save_intermediate = False
        self.draw_intermediate_graph = False
        self.frames = [] if self.draw_intermediate_graph else None

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
        self.X = X
        self.num_u = X.shape[0]
        self.num_v = X.shape[1]
        self.has_data = True

    def _initialize_factors(self):
        """
        Initialize the parameters U, V, S, P, Q, and R based on the specified initialization style.

        :return: None
        """
        if self.init_style == "random":
            self.U = torch.rand(self.num_u, self.k1, device=self.device, dtype=torch.float64)
            self.V = torch.rand(self.k2, self.num_v, device=self.device, dtype=torch.float64)
            self.S = self.X.max() * torch.rand((self.k1, self.k2), device=self.device, dtype=torch.float64)
            self.P = self.U @ self.S
            self.Q = self.S @ self.V
            self.R = self.X - self.P @ self.V

        elif self.init_style == "nnsvd":
            self.U, self.V = initialize.nnsvd_nmtf_initialize(self.X, self.k1, self.k2, self.seed)
            self.V = torch.t(self.V)

            # Not a real good way of doing this. Start with something random and let's update S first.
            # Perhaps bias toward diagonal.
            self.S = torch.rand((self.k1, self.k2), device=self.device, dtype=torch.float64)
            self.P = self.U @ self.S
            self.Q = self.S @ self.V
            self.R = self.X - self.P @ self.V

        else:
            print("Not a valid initialization method.")
            return -999

    def send_to_gpu(self):
        """
        Send full tensors to GPU if CUDA is available.

        :return: None
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

    def send_to_cpu(self):
        """
        Send full tensors to GPU if CUDA is available.
        :return: None
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

    # Update rules
    def _update_kth_block_u(self, k):
        """
        Update the kth factor of the U matrix.

        :param k: index of the block to update
        :return: None
        """
        q_norm = torch.linalg.norm(self.Q[k, :]) ** 2
        self.U[:, k] = torch.matmul(self.R, self.Q[k, :]) / q_norm

        # Apply Non-negativity
        self.U[self.U < 0] = 0

    def _update_kth_block_u_unit(self, k):
        """
        Update the kth factor of the U matrix. Normalizes this vector
        :param k: kidex of the block to update
        :return: None
        """
        self.U[:, k] = torch.matmul(self.R, self.Q[k, :])

        # Apply Non-negativity
        self.U[self.U[:, k] < 0, k] = 0

        # Normalize to unit length
        self.U[:, k] = self.U[:, k] / torch.linalg.norm(self.U[:])

    def _apply_orthog_u(self, k):
        """
        Apply orthogonal regularization term to the kth factor of the U matrix

        :param k: index of the block to update
        :return: None
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
        Apply orthogonal regularization term to the kth factor of the U matrix. Assumes unit norm. Uses lambda* def.
        :param k:
        :return:
        """
        if self.lU > 0:
            beta = torch.sum(self.U[:, [x for x in range(self.k1) if x not in [k]]], dim=1)
            beta = beta / torch.linalg.norm(beta)
            self.U[:, k] = self.U[:, k] - self.lU * beta
        # apply Non-negativity
        self.U[self.U[:, k] < 0, k] = 0
        self.U[:, k] = self.U[:, k] / torch.linalg.norm(self.U[:, k])

    def _apply_sparsity_u(self, k):
        """
        Apply the sparsity regularization term to the kth factor of the U.

        :param k: index of the kth factor.
        :return: None
        """
        q_norm = torch.linalg.norm(self.Q[k, :]) ** 2
        # Sparsity term
        if self.aU > 0:
            self.U[:, k] = self.U[:, k] - self.aU * torch.ones(self.num_u, device=self.device) / q_norm

        # Apply Non-negativity
        self.U[self.U < 0] = 0

    def _apply_sparsity_u_unit(self, k):
        """
        Apply the sparsity regularization term to the kth factor of U. Assumes unit norm of U.
        :param k: index of the kth factor
        :return: None
        """

        if self.aU > 0:
            self.U[:, k] = self.U[:, k] = self.aU * torch.ones(self.num_u, device=self.device)

        self.U[self.U[:, k] < 0, k] = 0
        self.U[:, k] = self.U[:, k] / torch.linalg.norm(self.U[:, k])

    def _enforce_non_zero_u(self, k):
        """
        :param k: index of a column in self.U to enforce non-zero values
        :return: None

        Enforces non-zero values in column k of self.U. If the sum of the column is zero, it sets all values to 1/num_u.
        If citer is greater than 5 and kill_factors is True, the program exits with an error message.
        """
        # Enforce non-zero
        if torch.sum(self.U[:, k]) == 0 or torch.isnan(self.U[:, k]).any():
            self.U[:, k] = torch.ones(self.num_u)
            self.U[:, k] = self.U[:, k] / torch.linalg.norm(self.U[:, k])
            if self.citer > 5 and self.kill_factors:
                sys.exit("Cell factor killed")

    def _update_kth_block_v(self, k):
        """
        Updates the kth block of the V matrix
        :param k: index of the row of V to update
        return: None
        """
        p_norm = torch.linalg.norm(self.P[:, k]) ** 2
        self.V[k, :] = torch.matmul(self.P[:, k], self.R) / p_norm

        # Apply Non-negativity
        self.V[self.V < 0] = 0

    def _update_kth_block_v_unit(self, k):
        """
        Update the kth block of the V matrix. Normalizes the vector to unit length.
        :param k:  index of the row of V to update
        :return:  None
        """
        self.V[k, :] = torch.matmul(self.P[:, k], self.R)
        # Apply Non-negativity
        self.V[k, self.V[k, :] < 0] = 0
        # Normalize V
        self.V[k, :] = self.V[k, :] / torch.linalg.norm(self.V[k, :])

    def _apply_orthog_v(self, k):
        """
        Apply orthogonal regularization update to the kth factor of v.

        :params k: index of the column to apply sparsity.
        :return: None
        """
        p_norm = torch.linalg.norm(self.P[:, k]) ** 2
        # Orthogonality term
        if self.lV > 0:
            self.V[k, :] = self.V[k, :] - self.lV * torch.sum(self.V[[x for x in range(self.k2) if x not in [k]], :],
                                                              dim=0) / p_norm

        # Apply Non-negativity
        self.V[self.V < 0] = 0

    def _apply_orthog_v_unit(self, k):
        """
        Apply orthogonal regularization update to the kth factor of V.  This is using the lambda* interpretation.
        :param k:
        :return:
        """
        if self.lV > 0:
            beta = torch.sum(self.V[[x for x in range(self.k2) if x not in [k]], :], dim=0)
            beta = beta / torch.linalg.norm(beta)
            self.V[k, :] = self.V[k, :] - self.lV * beta
        # Apply Non-negativity
        self.V[k, self.V[k, :] < 0] = 0
        # Normalize
        self.V[k, :] = self.V[k, :] / torch.linalg.norm(self.V[k, :])

    def _apply_sparsity_v(self, k):
        """
        Apply sparsity regularization update to the kth factor of V.

        :param k: index of the column to apply sparsity
        :return: None
        """
        p_norm = torch.linalg.norm(self.P[:, k]) ** 2
        # Sparsity term
        if self.aV > 0:
            self.V[k, :] = self.V[k, :] - self.aV * torch.ones(self.num_v, device=self.device) / p_norm

        # Apply Non-negativity
        self.V[self.V < 0] = 0

    def _apply_sparsity_v_unit(self, k):
        """
        Applying sparsity update to the kth factor of V.  This is using the lambda* interpretation.
        :param k: index of the column to apply sparsity
        :return: None
        """
        if self.aV > 0:
            self.V[k, :] = self.V[k, :] - self.aV * torch.ones(self.num_v, device=self)

        # Apply Non-negativity
        self.V[k, self.V[k, :] < 0] = 0

        # Normalize V
        self.V[k, :] = self.V[k, :] / torch.linalg.norm(self.V[k, :])

    def _enforce_non_zero_v(self, k):
        """
        :param k: Index of the gene
        :return: None

        Enforces non-zero value for the gene factor at index k. If the sum of values of the gene factor row is zero, it
        assigns equal weights to each value.
        If the condition self.citer > 5 and self.kill_factors is True, the program exits with the message
        "Gene factor killed".
        """
        # Enforce non-zero
        if torch.sum(self.V[k, :]) == 0 or torch.isnan(self.V[k, :]).any():
            self.V[k, :] = torch.ones(self.num_v)
            self.V[k, :] = self.V[k, :] / torch.linalg.norm(self.V[k, :])
            if self.citer > 5 and self.kill_factors:
                sys.exit("Gene factor killed")

    def _update_ith_jth_of_s(self, i, j):
        """
        Update each cell (i, j) of the S (sharing) matrix
        :param i: row index of the S matrix to update.
        :param j: column index of the S matrix to update.
        :return: None
        """
        u_norm = torch.linalg.norm(self.U[:, i]) ** 2
        v_norm = torch.linalg.norm(self.V[j, :]) ** 2
        val = torch.matmul(torch.matmul(self.U[:, i], self.R), self.V[j, :]) / (u_norm * v_norm)
        self.S[i, j] = val if val > 0 else 0



    # Update the residuals
    def _update_P(self):
        """
        Update the P matrix (U * S). The P matrix must be updated before refining V.
        :return: None
        """
        self.P = self.U @ self.S

    def _update_Q(self):
        """
        Update the Q matrix (S *V). The Q matrix must be updated before refining U.
        :return: None
        """
        self.Q = self.S @ self.V

    # Scaling functions
    def _normalize_and_scale_u(self):
        """
        Normalize U matrix factors to 1. Scale factor shifted to S matrix (i, j) terms.
        Required to run this before apply orthogonal regularization.

        :return: None
        """
        for idx in range(self.k1):
            u_norm = torch.linalg.norm(self.U[:, idx])
            self.U[:, idx] = self.U[:, idx] / u_norm
            self.S[idx, :] = self.S[idx, :] * u_norm

    def _normalize_and_scale_v(self):
        """
        Normalize V matrix factors to 1. Scale factor shifted to S matrix (i, j) terms.
        Required to run this before apply orthogonal regularization.
        :return: None
        """
        for idx in range(self.k2):
            v_norm = torch.linalg.norm(self.V[idx, :])
            self.V[idx, :] = self.V[idx, :] / v_norm
            self.S[:, idx] = self.S[:, idx] * v_norm

    # Update objectives
    def calculate_objective(self):
        """
        Computes the objective function value given current state. Adds in regularization parameter terms as needed
        :return: None
        """
        # Compute reconstruction error
        error = torch.linalg.norm(self.R, ord='fro').item() ** 2
        self.reconstruction_error[:, self.citer] = error

        # Compute lU component
        if self.lU > 0:
            overlap = (torch.transpose(self.U, 0, 1) @ self.U)
            overlap = overlap - torch.diag_embed(torch.diag(overlap))
            lU_reg = self.lU / 2 * torch.norm(overlap, p=1).item()
        else:
            lU_reg = 0
        self.lU_error[:, self.citer] = lU_reg

        # Compute lV component
        if self.lV > 0:
            overlap = self.V @ torch.transpose(self.V, 0, 1)
            overlap = overlap - torch.diag_embed(torch.diag(overlap))
            lV_reg = self.lV / 2 * torch.norm(overlap, p=1).item()
        else:
            lV_reg = 0
        self.lV_error[:, self.citer] = lV_reg
        # Compute aU component
        if self.aU > 0:
            aU_reg = self.aU / 2 * torch.sum(self.U).item()
        else:
            aU_reg = 0

        # Compute aV component
        if self.aU > 0:
            aV_reg = self.aV / 2 * torch.sum(self.V).item()
        else:
            aV_reg = 0
        self.error[:, self.citer] = error + lU_reg + lV_reg + aU_reg + aV_reg
        if self.citer > 0:
            self.relative_error[:, self.citer] = ((self.error[:, self.citer - 1] - self.error[:, self.citer]) /
                                                   self.error[:, self.citer - 1])
        else:
            self.relative_error[:, self.citer] = float('inf')

    def calculate_error_only(self):
        """
        Computes the objective function value given current state. Adds in regularization parameter terms as needed
        :return: None
        """
        # Compute reconstruction error
        error = torch.linalg.norm(self.R, ord='fro').item() ** 2
        self.reconstruction_error[:, self.citer] = error
        self.error[:, self.citer] = error
        self.relative_error = (self.error[-2] - self.error[-1])

    def _updateU(self):
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

    def _updateU_unit(self):
        for idx_i in range(self.k1):
            self.R = self.R + torch.outer(self.U[:, idx_i], self.Q[idx_i, :])
            self._update_kth_block_u_unit(idx_i)
            self._apply_orthog_u_unit(idx_i)
            self._apply_sparsity_u_unit(idx_i)
            self._enforce_non_zero_u(idx_i)
            self.R = self.R - torch.outer(self.U[:, idx_i], self.Q[idx_i, :])

    def _updateV(self):
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

    def _updateV_unit(self):
        for idx_j in range(self.k2):
            self.R = self.R + torch.outer(self.P[:, idx_j], self.V[idx_j, :])
            self._update_kth_block_v_unit(idx_j)
            self._apply_orthog_v_unit(idx_j)
            self._apply_sparsity_v_unit(idx_j)
            self._enforce_non_zero_v(idx_j)
            self.R = self.R - torch.outer(self.P[:, idx_j], self.V[idx_j, :])

    def _updateS(self):
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

    def update(self):
        """
        Define one update step for the U, V and S factors.
        :return: None
        """
        self._updateU()
        self._update_P()

        if self.lU > 0 or self.aU > 0:
            self.R = self.X - self.P @ self.V

        self._updateV()
        self._update_Q()

        if self.lU > 0 or self.aU > 0:
            self.R = self.X - self.P @ self.V

        self._updateS()
        self._normalize_and_scale_u()
        self._normalize_and_scale_v()
        self._update_P()
        self._update_Q()

    def update_unit(self):
        """
        Define one update step for U, V and S, using the unit rules.
        :return: None
        """

        self._updateU_unit()
        self._update_P()

        self._updateV_unit()
        self._update_Q()

        self._updateS()
        self._update_P()
        self._update_Q()

    def determine_reg_state(self):
        if self.var_lambda:
            self.lU = self.max_lU * self.sigmoid_schedule(self.mid_epoch_param, self.shape_param)
            self.lV = self.max_lV * self.sigmoid_schedule(self.mid_epoch_param, self.shape_param)
        else:
            self.lU = self.max_lU
            self.lV = self.max_lV

        if self.var_alpha:
            self.aU = self.max_aU * self.sigmoid_schedule(self.mid_epoch_param, self.shape_param)
            self.aV = self.max_aV * self.sigmoid_schedule(self.mid_epoch_param, self.shape_param)
        else:
            self.aU = self.max_aU
            self.aV = self.max_aV

    def fit(self):
        self.citer = 0
        start_time = time.time()
        curr_time = time.time()

        if self.verbose:
            print("Initializing NMTF factors")
        # Initialize factors
        self._initialize_factors()
        self._normalize_and_scale_u()
        self._normalize_and_scale_v()
        self._updateS()
        self.track_objective_setup()

        if self.verbose:
            print("Beginning NMTF")

        if self.save_clust:
            U_jaccard = MulticlassJaccardIndex(num_classes=self.k1, average='weighted')
            V_jaccard = MulticlassJaccardIndex(num_classes=self.k2, average='weighted')
            self.track_clusters_setup()

        if self.draw_intermediate_graph:
            self.frames = []
            fig = self.visualizeFactors()
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            self.frames.append(frame)
            plt.close(fig)

        while self.citer != self.maxIter:
            self.citer += 1
            self.determine_reg_state()
            if self.legacy:
                self.update()
            else:
                self.update_unit()
            self.calculate_objective()

            if self.verbose:
                next_time = time.time()
                print("Iter: {0}\tIter Time: {1:.3f}\tTotal Time: {2:.3f}\tError: {3:.3e}\tRelative Delta "
                      "Residual: {4:.3e}".
                      format(self.citer, next_time - curr_time, next_time - start_time, self.error[:, self.citer].item(),
                             self.relative_error[:, self.citer].item()))
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
                fig = self.visualizeFactors()
                fig.canvas.draw()
                frame = np.array(fig.canvas.renderer.buffer_rgba())
                self.frames.append(frame)
                plt.close(fig)

            if self.termTol > self.relative_error[:, self.citer].item() >= 0:
                break

    def print_USV(self, file_pre):
        """
        Write the lower dimensional matrices to file. U.txt, V.txt, and S.txt.
        Files are tab delimited text files.

        :param out_path: Path
        """
        U_out = self.U.cpu()
        U_out = torch.transpose(U_out, 0, 1)
        U_out = pd.DataFrame(U_out.numpy())
        U_out.to_csv(self.out_path + '/'+ file_pre +  "_U.txt", sep='\t', header=False, index=False)

        V_out = self.V.cpu()
        V_out = pd.DataFrame(V_out.numpy())
        V_out.to_csv(self.out_path + '/' + file_pre + "_V.txt", sep="\t", header=False, index=False)

        S_out = self.S.cpu()
        S_out = pd.DataFrame(S_out.numpy())
        S_out.to_csv(self.out_path + '/' + file_pre + "_S.txt", sep="\t", header=False, index=False)

    def print_output(self, out_path):
        """
        Write output files. This includes the lower dimensional matrices U, S, V; the terms associated with the
        objective function (the residual, the lambda regularization terms);
        the Assignment of U an V at every iteration. The stepwise convergence of cluster assignments of U S and V.

        """
        self.print_USV(out_path)

        if self.track_objective:
            reconstruction_error_out = self.reconstruction_error.cpu()
            reconstruction_error_out = pd.DataFrame(reconstruction_error_out.numpy())
            reconstruction_error_out.to_csv(out_path + "/reconstruction_error.txt", sep="\t", header=False, index=False)

            lU_error_out = self.lU_error.cpu()
            lU_error_out = pd.DataFrame(lU_error_out.numpy())
            lU_error_out.to_csv(out_path + '/lU_error.txt', sep='\t', header=False, index=False)

            lV_error_out = self.lV_error.cpu()
            lV_error_out = pd.DataFrame(lV_error_out.numpy())
            lV_error_out.to_csv(out_path + "/lV_error.txt", sep='\t', header=False, index=False)

        if self.save_clust:
            U_test_out = self.U_assign.cpu()
            U_test_out = pd.DataFrame(U_test_out.numpy())
            U_test_out = U_test_out.loc[:, (U_test_out != 0).any(axis=0)]
            U_test_out.to_csv(out_path + "/U_assign.txt", sep='\t', header=False, index=False)

            V_test_out = self.V_assign.cpu()
            V_test_out = pd.DataFrame(V_test_out.numpy())
            V_test_out = V_test_out.loc[:, (V_test_out != 0).any(axis=0)]
            V_test_out.to_csv(out_path + "/V_assign.txt", sep='\t', header=False, index=False)

            relative_error_out = self.relative_error.cpu()
            relative_error_out = pd.DataFrame(relative_error_out.numpy())
            relative_error_out.to_csv(out_path + "/relative_error.txt", sep='\t', header=False, index=False)

            V_JI_out = self.V_JI.cpu()
            V_JI_out = pd.DataFrame(V_JI_out.numpy())
            V_JI_out.to_csv(out_path + "/V_JI.txt", sep='\t', header=False, index=False)

            U_JI_out = self.U_JI.cpu()
            U_JI_out = pd.DataFrame(U_JI_out.numpy())
            U_JI_out.to_csv(out_path + "/U_JI.txt", sep='\t', header=False, index=False)

    def track_objective_setup(self):
        """
        Save the objective values for error, U regularization and V regularization for each iteration of th algorithm
        : return: None
        """
        self.reconstruction_error = torch.zeros(size=[1, self.maxIter + 1], dtype=torch.float32)
        self.lU_error = torch.zeros(size=[1, self.maxIter + 1], dtype=torch.float32)
        self.lV_error = torch.zeros(size=[1, self.maxIter + 1], dtype=torch.float32)
        self.relative_error = torch.zeros(size=[1, self.maxIter + 1], dtype=torch.float32)
        self.error = torch.zeros(size=[1, self.maxIter + 1], dtype=torch.float32)
        self.calculate_objective()

    def track_clusters_setup(self):
        self.U_assign = torch.zeros(size=[self.num_u, self.maxIter + 1], dtype=torch.float32)
        self.V_assign = torch.zeros(size=[self.num_v, self.maxIter + 1], dtype=torch.float32)
        self.U_JI = torch.zeros(size=[self.num_u, self.maxIter + 1], dtype=torch.float32)
        self.V_JI = torch.zeros(size=[self.num_v, self.maxIter + 1], dtype=torch.float32)
        self.U_JI[:, 0] = float('inf')
        self.V_JI[:, 0] = float('inf')

    def save_cluster(self):
        """
        Save cluster assignments and errors for each iteration of the algorithm.

        :return: None
        """
        self.U_assign = torch.zeros(size=[self.num_u, self.maxIter + 1], dtype=torch.uint8)
        self.U_assign[:, 0] = torch.argmax(self.U, dim=1)
        self.U_JI = torch.zeros(size=[self.num_u, self.maxIter], dtype=torch.float32)
        self.V_assign = torch.zeros(size=[self.num_v, self.maxIter + 1], dtype=torch.uint8)
        self.V_assign[:, 0] = torch.argmax(self.V, dim=0)
        self.V_JI = torch.zeros(size=[self.num_v, self.maxIter], dtype=torch.float32)
        self.relative_error = torch.zeros(size=[1, self.maxIter], dtype=torch.float32)


    def assign_cluster(self):
        self.U_assign = torch.argmax(self.U, dim=1)
        self.V_assign = torch.argmax(self.V, dim=0)
    def sigmoid_schedule(self, mid_iter=5, shape=10.0):
        """
        Generates a sigmoid scheduling function for the lambda U and Lambda V regularization parameter.
         LU and LV achieve half value ad mid_iter.

        :param mid_iter: The midpoint iteration where the schedule peaks.
        :param shape: The shape parameter that controls the steepness of the sigmoid curve.

        :return: The value of the sigmoid schedule at the current iteration.
        """
        return 1 / (1 + np.exp(-shape * (self.citer - mid_iter)))

    def visualizeFactors(self, cmap='viridis', interp='nearest'):
        fig = plt.figure(figsize=(16, 6))
        grids = GridSpec.GridSpec(2, 3, wspace=0.1, width_ratios=(0.2, 0.4, 0.4), height_ratios=(0.3, 0.7))

        ax1 = fig.add_subplot(grids[1, 0])
        ax1.imshow(self.U.detach().numpy(), norm='linear', aspect="auto", cmap=cmap, interpolation=interp)
        ax1.set_axis_off()
        # ax1.set_title("U Matrix")

        # Visualize S matrix
        ax2 = fig.add_subplot(grids[0, 0])
        ax2.imshow(self.S.t().detach().numpy(), norm='linear', aspect="auto", cmap=cmap, interpolation=interp)
        ax2.set_axis_off()
        # ax2.set_title("S Matrix")

        # Visualize V matrix
        ax3 = fig.add_subplot(grids[0, 1])
        ax3.imshow(self.V.detach().numpy(), norm='linear', aspect="auto", cmap=cmap, interpolation=interp)
        ax3.set_axis_off()
        # ax3.set_title("V Matrix")

        # Visualize X matrix
        ax4 = fig.add_subplot(grids[1, 1])
        ax4.imshow((self.U @ self.S @ self.V).detach().numpy(), norm='linear', aspect="auto", cmap=cmap,
                   interpolation=interp)
        # ax4.set_title("X Matrix")
        ax4.set_axis_off()

        ax5 = fig.add_subplot(grids[1, 2])
        ax5.imshow(self.X,  norm='linear', aspect="auto", cmap=cmap, interpolation=interp)
        ax5.set_axis_off()
        plt.close(fig)
        return fig

    def visualizeFactorsSorted(self, cmap='viridis', interp='nearest'):
        fig = plt.figure(figsize=(16, 6))
        grids = GridSpec.GridSpec(2, 3, wspace=0.1, width_ratios=(0.2, 0.4, 0.4), height_ratios=(0.3, 0.7))


        # Generate Sorting for U
        max_U, max_U_idx = self.U.max(dim=1)
        sorting_criteria = torch.stack([max_U_idx, max_U], dim=1)
        sorted_U_indices = torch.argsort(sorting_criteria, dim=0, stable=True)[:, 0]

        # Generate Sorting for V
        max_V, max_V_idx = self.V.max(dim=0)
        sorting_criteria = torch.stack([max_V_idx, max_V], dim=1)
        sorted_V_indices = torch.argsort(sorting_criteria, dim=0, stable=True)[:, 0]

        ax1 = fig.add_subplot(grids[1, 0])
        ax1.imshow(self.U[sorted_U_indices, :].detach().numpy(), aspect="auto", cmap=cmap, interpolation=interp)
        ax1.set_axis_off()
        # ax1.set_title("U Matrix")

        # Visualize S matrix
        ax2 = fig.add_subplot(grids[0, 0])
        ax2.imshow(self.S.t().detach().numpy(), aspect="auto", cmap=cmap, interpolation=interp)
        ax2.set_axis_off()
        # ax2.set_title("S Matrix")

        # Visualize V matrix
        ax3 = fig.add_subplot(grids[0, 1])
        ax3.imshow(self.V[:, sorted_V_indices].detach().numpy(), aspect="auto", cmap=cmap, interpolation=interp)
        ax3.set_axis_off()
        # ax3.set_title("V Matrix")


        # Visualize X matrix
        X_est = self.U @ self.S @ self.V
        X_est = X_est[sorted_U_indices, :]
        X_est = X_est[:, sorted_V_indices]
        ax4 = fig.add_subplot(grids[1, 1])
        ax4.imshow(X_est.detach().numpy(), aspect="auto", cmap=cmap,
                   interpolation=interp)
        ax4.set_axis_off()


        # ax4.set_title("X Matrix")
        X_temp = self.X
        X_temp = X_temp[sorted_U_indices, :]
        X_temp = X_temp[:, sorted_V_indices]
        ax5 = fig.add_subplot(grids[1, 2])
        ax5.imshow(X_temp, aspect="auto", cmap=cmap, interpolation=interp)
        ax5.set_axis_off()
        plt.close(fig)
        return fig

    def writeGIF(self, filename = "NMTF_fit.gif", fps = 5):
        if self.out_path is not None:
            outfile = self.out_path + '/' + filename
        else:
            outfile = "fit.gif"
        print("writing gif to {0}".format(outfile))
        imageio.mimsave(outfile, self.frames, fps=fps, loop = 0)

    def reclusterV(self, linkage_type="average", dist_metric='euclidean'):
        # Use a psuedo-Q representation to recluster.
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
            D = torch.cdist(self.Q.T, self.Q.T, p = dist_metric)
        else:
            print("invalid distance metric.")
            return

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

        # refit.
        start_time = time.time()
        curr_time = time.time()
        self.citer = 0
        self._normalize_and_scale_v()
        self._updateS()
        self.track_objective_setup()


        if self.save_clust:
            U_jaccard = MulticlassJaccardIndex(num_classes=self.k1, average='weighted')
            V_jaccard = MulticlassJaccardIndex(num_classes=self.k2, average='weighted')
            self.track_clusters_setup()

        if self.draw_intermediate_graph:
            self.frames = []
            fig = self.visualizeFactors()
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            self.frames.append(frame)
            plt.close(fig)

        while self.citer != self.maxIter:
            self.citer += 1
            self.determine_reg_state()
            if self.legacy:
                self.update()
            else:
                self.update_unit()
            self.calculate_objective()

            if self.verbose:
                next_time = time.time()
                print("Iter: {0}\tIter Time: {1:.3f}\tTotal Time: {2:.3f}\tError: {3:.3e}\tRelative Delta "
                      "Residual: {4:.3e}".
                      format(self.citer, next_time - curr_time, next_time - start_time,
                             self.error[:, self.citer].item(),
                             self.relative_error[:, self.citer].item()))
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
                fig = self.visualizeFactors()
                fig.canvas.draw()
                frame = np.array(fig.canvas.renderer.buffer_rgba())
                self.frames.append(frame)
                plt.close(fig)

            if self.termTol > self.relative_error[:, self.citer].item() >= 0:
                break

    def visualizeClusters(self, cmap='viridis', interp='nearest'):
        fig = plt.figure(figsize=(8, 6))
        grids = GridSpec.GridSpec(2, 2, wspace=0.1, width_ratios=(0.2, 0.8), height_ratios=(0.3, 0.7))

        ax1 = fig.add_subplot(grids[1, 0])
        ax1.imshow(self.U_assign.view(-1, 1).detach().numpy(), norm='linear', aspect="auto", cmap='Dark2', interpolation=interp)
        ax1.set_axis_off()

        # Visualize V matrix
        ax3 = fig.add_subplot(grids[0, 1])
        ax3.imshow(self.V_assign.view(1, -1).detach().numpy(), norm='linear', aspect="auto", cmap='Dark2', interpolation=interp)
        ax3.set_axis_off()

        ax4 = fig.add_subplot(grids[1, 1])
        ax4.imshow(self.X,  norm='linear', aspect="auto", cmap=cmap, interpolation=interp)
        ax4.set_axis_off()
        plt.close(fig)
        return fig

    def visualizeClustersSorted(self, cmap='viridis', interp='nearest'):
        fig = plt.figure(figsize=(8, 6))
        grids = GridSpec.GridSpec(2, 2, wspace=0.1, width_ratios=(0.05, 0.95), height_ratios=(0.05, 0.95))

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
        ax3.imshow(self.V_assign[sorted_V_indices].view(1, -1).detach().numpy(), aspect="auto", cmap='gray',
                   vmin=0, vmax=2, interpolation=interp)
        ax3.set_axis_off()
        # ax3.set_title("V Matrix")

        X_temp = self.X[sorted_U_indices, :][:, sorted_V_indices]
        ax5 = fig.add_subplot(grids[1, 1])
        ax5.imshow(X_temp, aspect="auto", cmap=cmap, interpolation=interp)
        ax5.set_axis_off()
        plt.close(fig)
        return fig