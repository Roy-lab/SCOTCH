import torch
import time
import pandas as pd
import numpy as np
import sys
from statistics import mean


class NMTF:
    def __init__(self, verbose=True, max_iter=100, seed=1001, term_tol=1e-5, l_u=0, l_v=0, a_u=0, a_v=0, k1=2, k2=2,
                 device="cpu"):
        # Initialize Parameter space
        self.verbose = verbose
        self.maxIter = int(max_iter)
        self.seed = int(seed)
        self.termTol = float(term_tol)
        self.lU = float(l_u)
        self.lV = float(l_v)
        self.aU = float(a_u)
        self.aV = float(a_v)
        self.k1 = int(k1)
        self.k2 = int(k2)
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

        # Initialize Device
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

    def load_data_from_text(self, datafile, delimiter='\t', header=None):
        if self.verbose:
            start_time = time.time()
            print('Starting to read data from {0:s}'.format(datafile))
        df = pd.read_csv(datafile, sep=delimiter, header=header, dtype=np.float32)
        df = df.to_numpy()
        self.num_u = df.shape[0]
        self.num_v = df.shape[1]
        self.X = torch.from_numpy(df)
        self.U = torch.randint(2, size=[self.num_u, self.k1], dtype=torch.float32)
        self.V = torch.randint(2, size=[self.k2, self.num_v], dtype=torch.float32)
        self.S = df.max() * torch.rand((self.k1, self.k2))
        self.P = self.U @ self.S
        self.Q = self.S @ self.V
        self.R = self.X - self.P @ self.V
        self.calculate_objective()
        if self.verbose:
            end_time = time.time()
            print('Time to read file: {0:.3f}'.format(end_time - start_time))
        # Change loaded data status
        self.has_data = True

    def load_data_from_pt(self, datafile):
        if self.verbose:
            start_time = time.time()
            print('Starting to read data from {0:s}'.format(datafile))
        self.X = torch.load(datafile)
        self.num_u = self.X.shape[0]
        self.num_v = self.X.shape[1]
        self.U = torch.rand(self.num_u, self.k1)
        self.V = torch.rand(self.k2, self.num_v)
        self.S = self.X.max() * torch.rand((self.k1, self.k2))
        self.P = self.U @ self.S
        self.Q = self.S @ self.V
        self.R = self.X - self.P @ self.V
        self.calculate_objective()
        if self.verbose:
            end_time = time.time()
            print('Time to read file: {0:.3f}'.format(end_time - start_time))
        # Change loaded data status
        self.has_data = True

    def send_to_gpu(self):
        if torch.cuda.is_available():
            self.U = self.U.to(self.device)
            self.V = self.V.to(self.device)
            self.S = self.S.to(self.device)
            self.P = self.P.to(self.device)
            self.Q = self.Q.to(self.device)
            self.R = self.R.to(self.device)
            # self.error.to(self.device)

    def update_kth_block_u(self, k):
        q_norm = torch.linalg.norm(self.Q[k, :]) ** 2
        self.U[:, k] = torch.matmul(self.R, self.Q[k, :]) / q_norm

        # Apply Non-negativity
        self.U[self.U < 0] = 0

    def apply_orthog_u(self, k):
        q_norm = torch.linalg.norm(self.Q[k, :]) ** 2
        # Orthogonality term
        if self.lU > 0:
            self.U[:, k] = self.U[:, k] - self.lU * torch.sum(self.U[:, [x for x in range(self.k1) if x not in [k]]],
                                                              dim=1) / q_norm
        # Apply Non-negativity
        self.U[self.U < 0] = 0

    def apply_sparsity_u(self, k):
        q_norm = torch.linalg.norm(self.Q[k, :]) ** 2
        # Sparsity term
        if self.aU > 0:
            self.U[:, k] = self.U[:, k] - self.aU * torch.ones(self.num_u) / q_norm

        # Apply Non-negativity
        self.U[self.U < 0] = 0

    def enforce_non_zero_u(self, k):
        # Enforce non-zero
        if torch.sum(self.U[:, k]) == 0:
            self.U[:, k] = 1 / self.num_u * torch.ones(self.num_u)

    def update_kth_block_v(self, k):
        p_norm = torch.linalg.norm(self.P[:, k]) ** 2
        self.V[k, :] = torch.matmul(self.P[:, k], self.R) / p_norm

        # Apply Non-negativity
        self.V[self.V < 0] = 0

    def apply_orthog_v(self, k):
        p_norm = torch.linalg.norm(self.P[:, k]) ** 2
        # Orthogonality term
        if self.lV > 0:
            self.V[k, :] = self.V[k, :] - self.lV * torch.sum(self.V[[x for x in range(self.k2) if x not in [k]], :],
                                                              dim=0) / p_norm

        # Apply Non-negativity
        self.V[self.V < 0] = 0

    def apply_sparsity_v(self, k):
        p_norm = torch.linalg.norm(self.P[:, k]) ** 2
        # Sparsity term
        if self.aV > 0:
            self.V[k, :] = self.V[k, :] - self.aV * torch.ones(self.num_v) / p_norm

        # Apply Non-negativity
        self.V[self.V < 0] = 0

    def enforce_non_zero_v(self, k):
        # Enforce non-zero
        if torch.sum(self.V[k, :]) == 0:
            self.V[k, :] = (1 / self.num_v) * torch.ones(self.num_v)

    def update_ith_jth_of_s(self, i, j):
        u_norm = torch.linalg.norm(self.U[:, i]) ** 2
        v_norm = torch.linalg.norm(self.V[j, :]) ** 2
        val = torch.matmul(torch.matmul(self.U[:, i], self.R), self.V[j, :]) / (u_norm * v_norm)
        self.S[i, j] = val if val > 0 else 0

    def update_P(self):
        self.P = self.U @ self.S

    def update_Q(self):
        self.Q = self.S @ self.V

    def normalize_and_scale_u(self):
        for idx in range(self.k1):
            u_norm = torch.linalg.norm(self.U[:, idx])
            self.U[:, idx] = self.U[:, idx] / u_norm
            self.S[idx, :] = self.S[idx, :] * u_norm

    def normalize_and_scale_v(self):
        for idx in range(self.k2):
            v_norm = torch.linalg.norm(self.V[idx, :])
            self.V[idx, :] = self.V[idx, :] / v_norm
            self.S[:, idx] = self.S[:, idx] * v_norm

    def calculate_objective(self):
        # Compute reconstruction error
        error = torch.linalg.norm(self.R, ord='fro').item()

        # Compute lU component
        if self.lU > 0:
            overlap = (torch.transpose(self.U, 0, 1) @ self.U)
            overlap = overlap - torch.diag_embed(torch.diag(overlap))
            lU_reg = self.lU / 2 * torch.norm(overlap, p=1).item()
        else:
            lU_reg = 0

        # Compute lV component
        if self.lV > 0:
            overlap = self.V @ torch.transpose(self.V, 0, 1)
            overlap = overlap - torch.diag_embed(torch.diag(overlap))
            lV_reg = self.lV / 2 * torch.norm(overlap, p=1).item()
        else:
            lV_reg = 0

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

        self.error.append(error + lU_reg + lV_reg + aU_reg + aV_reg)
        # torch.cat((self.error, torch.linalg.norm(self.R, ord='fro')), 0)

    def update(self):
        for idx_i in range(self.k1):
            self.R = self.R + torch.outer(self.U[:, idx_i], self.Q[idx_i, :])
            self.update_kth_block_u(idx_i)
            self.R = self.R - torch.outer(self.U[:, idx_i], self.Q[idx_i, :])

        for idx_i in range(self.k1):
            if self.lU > 0:
                self.apply_orthog_u(idx_i)
            if self.aU > 0:
                self.apply_sparsity_u(idx_i)
            self.enforce_non_zero_u(idx_i)

        self.update_P()
        if self.lU > 0 or self.aU > 0:
            self.R = self.X - self.P @ self.V

        for idx_j in range(self.k2):
            self.R = self.R + torch.outer(self.P[:, idx_j], self.V[idx_j, :])
            self.update_kth_block_v(idx_j)
            self.R = self.R - torch.outer(self.P[:, idx_j], self.V[idx_j, :])

        for idx_j in range(self.k2):
            if self.lV > 0:
                self.apply_orthog_v(idx_j)
            if self.aV > 0:
                self.apply_sparsity_v(idx_j)
            self.enforce_non_zero_v(idx_j)

        self.update_Q()
        if self.lU > 0 or self.aU > 0:
            self.R = self.X - self.P @ self.V

        for idx_i in range(self.k1):
            for idx_j in range(self.k2):
                self.R = self.R + self.S[idx_i, idx_j] * torch.outer(self.U[:, idx_i], self.V[idx_j, :])
                self.update_ith_jth_of_s(idx_i, idx_j)
                self.R = self.R - self.S[idx_i, idx_j] * torch.outer(self.U[:, idx_i], self.V[idx_j, :])
            if torch.sum(self.S[idx_i, :]) == 0:
                self.S[idx_i, :] = 1e-5

        for idx_j in range(self.k2):
            if torch.sum(self.S[:, idx_j]) == 0:
                self.S[:, idx_j] = 1e-5

        self.normalize_and_scale_u()
        self.normalize_and_scale_v()
        self.update_P()
        self.update_Q()

    def fit(self):
        self.send_to_gpu()
        start_time = time.time()
        curr_time = time.time()
        for counter in range(self.maxIter):
            self.update()
            self.calculate_objective()
            slope = (self.error[-2] - self.error[-1]) / self.error[-2]
            if self.verbose:
                next_time = time.time()
                print("Iter Time: {0:.3f}\tTotal Time: {1:.3f}\tError: {2:.3e}\tRelative Delta Residual: {3:.3e}".
                      format(next_time - curr_time, next_time - start_time, self.error[-1], slope))
                curr_time = next_time
            if self.termTol > slope > 0:
                break

    def print_output(self, outpath):
        U_out = self.U.cpu()
        U_out = torch.transpose(U_out, 0, 1)
        U_out = pd.DataFrame(U_out.numpy())
        U_out.to_csv(outpath + "/U.txt", sep='\t', header=None, index=False)

        V_out = self.V.cpu()
        # V_out = torch.transpose(V_out, 0 , 1)
        V_out = pd.DataFrame(V_out.numpy())
        V_out.to_csv(outpath + "/V.txt", sep='\t', header=None, index=False)

        S_out = self.S.cpu()
        S_out = pd.DataFrame(S_out.numpy())
        S_out.to_csv(outpath + "/S.txt", sep='\t', header=None, index=False)
