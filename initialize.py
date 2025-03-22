import torch



def nnsvd_nmf_initialize(X, k, seed=None):
    """
    Initializes the factors W and H for Non-negative Matrix Factorization (NMF)
    using the Non-negative Double Singular Value Decomposition (NNDSVD) method
    with PyTorch.

    This initialization helps improve convergence speed and avoid local minima
    by providing structured non-negative starting points for W and H.

    :param torch.Tensor X:
        The input non-negative matrix to be factorized, with shape (m, n).
        All elements of X should be non-negative.
    :param int k:
        The rank for the factorization, representing the number of components.
    :param int seed:
        (Optional) The random seed for reproducibility of the initialization.
        If not specified, a random seed will be used.

    :return:
        A tuple containing:
        - **W** (*torch.Tensor*): Initialized factor matrix of shape (m, k).
        - **H** (*torch.Tensor*): Initialized factor matrix of shape (n, k).
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """

    if seed is not None:
        torch.manual_seed(seed)

    # Dimensions of A
    m, n = X.shape

    # Calculate the mean of A to use for filling non-positive values
    a = torch.mean(X)

    # Perform SVD on A using torch's SVD function
    # R
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
    using PyTorch. Unlike standard NMF, the internal dimensions of W and H can differ.

    This method aims to provide a structured, non-negative initialization for
    W and H, improving convergence speed and stability.

    :param torch.Tensor X:
        The input non-negative matrix to factorize, with shape (m, n).
        All elements of X should be non-negative.
    :param int k_W:
        The rank for the row factorization, representing the number of components in W.
    :param int k_H:
        The rank for the column factorization, representing the number of components in H.
    :param int seed:
        (Optional) The random seed for reproducibility. If not specified, a random
        seed will be used.

    :return:
        A tuple containing the initialized factor matrices:
        - **W** (*torch.Tensor*): Factor matrix of shape (m, k_W).
        - **H** (*torch.Tensor*): Factor matrix of shape (n, k_H).
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    if seed is not None:
        torch.manual_seed(seed)

    W, _ = nnsvd_nmf_initialize(X, k1, seed)  # Optimal initialization of the rows as ortho bases
    H, _ = nnsvd_nmf_initialize(X.t(), k2, seed)  # Optimal initialization of the columns as ortho bases
    return W, H
