import torch
import numpy as np
import pandas as pd


def nnsvd_nmf_initialize(X, k, seed=None):
    """
        NNDSVD-based initialization for Non-negative Matrix Factorization (NMF)
        using PyTorch.

        Parameters:
        A (torch.Tensor): The input non-negative matrix to factorize, shape (m, n).
        k (int): The rank for the factorization (number of components).
        seed (int, optional): Random seed for reproducibility.

        Returns:
        W (torch.Tensor): Initialized matrix of shape (m, k).
        H (torch.Tensor): Initialized matrix of shape (n, k).
        """
    if seed is not None:
        torch.manual_seed(seed)

    # Dimensions of A
    m, n = X.shape

    # Calculate the mean of A to use for filling non-positive values
    a = torch.mean(X)

    # Perform SVD on A using torch's SVD function
    ## R
    U, S_svd, V = torch.svd_lowrank(X, q=k)

    # Initialize W and H matrices
    W = torch.zeros((m, k), dtype=X.dtype, device=X.device)
    H = torch.zeros((n, k), dtype=X.dtype, device=X.device)

    # We need to initialize the first comp because technically we could have double negatives... This is resolved here.
    for i in range(0, k):
        u = U[:, i]
        v = V[:, i]

        # Split into positive and negative parts
        up = u.clamp(min=0)
        un = (-u).clamp(min=0)
        vp = v.clamp(min=0)
        vn = (-v).clamp(min=0)

        # Calculate norms
        up_norm = torch.norm(up)
        un_norm = torch.norm(un)
        vp_norm = torch.norm(vp)
        vn_norm = torch.norm(vn)

        # Compute positive and negative products
        mp = up_norm * vp_norm
        mn = un_norm * vn_norm

        # Choose the component with a larger product
        if mp > mn:
            W[:, i] = torch.sqrt(S_svd[i] * mp) * (up / up_norm)
            H[:, i] = torch.sqrt(S_svd[i] * mp) * (vp / vp_norm)
        else:
            W[:, i] = torch.sqrt(S_svd[i] * mn) * (un / un_norm)
            H[:, i] = torch.sqrt(S_svd[i] * mn) * (vn / vn_norm)

    # This is in the C++ code but not in the original paper.
    # Replace non-positive values in W and H with the mean of A. Prevents too many zeros.
    W = torch.where(W >= 0, W, a)
    H = torch.where(H >= 0, H, a)

    return W, H


def nnsvd_nmtf_initialize(X, k1, k2, seed=None):
    """
    NNDSVD-based initialization for Non-negative Matrix Tri-Factorization (NMTF)
    where the internal dimensions of W and H differ, using PyTorch.

    Parameters:
    A (torch.Tensor): The input non-negative matrix to factorize, shape (m, n).
    k_W (int): The rank for the row factorization (number of components in W).
    k_H (int): The rank for the column factorization (number of components in H).
    seed (int, optional): Random seed for reproducibility.

    Returns:
    W (torch.Tensor): Initialized matrix of shape (m, k_W).
    H (torch.Tensor): Initialized matrix of shape (n, k_H).
    """
    if seed is not None:
        torch.manual_seed(seed)

    W, _ = nnsvd_nmf_initialize(X, k1, seed)  # Optimal initialization of the rows as ortho bases
    H, _ = nnsvd_nmf_initialize(X.t(), k2, seed)  # Optimal initialization of the columns as ortho bases
    return W, H
