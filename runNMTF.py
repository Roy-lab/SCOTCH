import argparse

import DataLoader
import NMTF as fact
import os


def runNMTF(args):
    """
    Runs Non-negative Matrix Tri-Factorization (NMTF) on an input dataset and saves the results.

    This function initializes the NMTF model using the provided arguments, loads the input data
    (either from a PyTorch `.pt` file or a tab-delimited text file), fits the model to the data,
    and saves the output to the specified directory.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments with the following attributes:

        - **in_file** (str): Path to the input file (tab-delimited matrix or .pt file).
        - **k1** (int, optional): Dimension of the row factors. Default is -999.
        - **k2** (int, optional): Dimension of the column factors. Default is -999.
        - **lU** (float, optional): Orthogonal regularization for the U factor. Default is 0.
        - **lV** (float, optional): Orthogonal regularization for the V factor. Default is 0.
        - **aU** (float, optional): Sparsity (L1) regularization for the U factor. Default is 0.
        - **aV** (float, optional): Sparsity (L1) regularization for the V factor. Default is 0.
        - **verbose** (bool, optional): If True, print progress to the terminal. Default is False.
        - **seed** (int, optional): Random seed for reproducibility. Default is 1010.
        - **max_iter** (int, optional): Maximum number of iterations. Default is 100.
        - **term_tol** (float, optional): Termination tolerance for relative error change. Default is 1e-25.
        - **out_dir** (str, optional): Directory for saving output files. Default is '.'.
        - **save_clust** (bool, optional): Save cluster assignments for each iteration. Default is False.
        - **kill_factors** (bool, optional): Option to kill unused factors. Default is False.
        - **track_objective** (bool, optional): Track objective function values during training. Default is False.
        - **save_USV** (bool, optional): Save factorization components (U, S, V) at each iteration. Default is False.
        - **device** (str, optional): Compute device for PyTorch ('cuda:0', 'cuda:1', 'cpu'). Default is 'cuda:0'.
        - **legacy** (bool, optional): Use legacy update method for factorization. Default is False.

    Returns
    -------
    None
    """
    if args.save_USV:
        mod = fact.NMTF(verbose=args.verbose, max_iter=args.max_iter,
                        seed=args.seed, term_tol=args.term_tol, max_l_u=args.lU,
                        max_l_v=args.lV, max_a_u=args.aU, max_a_v=args.aV, k1=args.k1,
                        k2=args.k2, save_clust=args.save_clust, track_objective=args.track_objective,
                        kill_factors=args.kill_factors, write_intermediate=args.save_intermediate,
                        out_path=args.out_dir, store_effective=args.store_effective, device=args.device)
    else:
        mod = fact.NMTF(verbose=args.verbose, max_iter=args.max_iter,
                        seed=args.seed, term_tol=args.term_tol, max_l_u=args.lU,
                        max_l_v=args.lV, max_a_u=args.aU, max_a_v=args.aV, k1=args.k1,
                        k2=args.k2, save_clust=args.save_clust, track_objective=args.track_objective,
                        kill_factors=args.kill_factors, store_effective=args.store_effective,
                        device=args.device)

    dl = DataLoader.DataLoader(verbose=args.verbose)
    file_parts = os.path.splitext(args.in_file)
    if file_parts[1] == '.pt':
        X, x_shape = dl.from_pt(datafile=args.in_file)
    else:
        X, x_shape = dl.from_text(datafile=args.in_file)
    mod.assign_X_data(X)
    mod.send_to_gpu()
    mod.fit()
    mod.print_output(args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--in_file',
                        help='tab-delimited input matrix. Or pytorch .pt file',
                        required=True)
    parser.add_argument('--k1',
                        help='lower dimension of the row factors (U).',
                        required=False,
                        default=-999)
    parser.add_argument('--k2',
                        help='lower dimension of the column factors (V).',
                        required=False,
                        default=-999)
    parser.add_argument('--test_multiple',
                        help="file containing test k1 and k2 in two tab delimited columns.",
                        required=False,
                        default='')
    parser.add_argument('--lU',
                        help='Ortho regularization of U term.',
                        required=False,
                        default=0)
    parser.add_argument('--lV',
                        help='Ortho regularization of V term.',
                        required=False,
                        default=0)
    parser.add_argument('--aU',
                        help='Sparsity (L1) regularization of U term.',
                        required=False,
                        default=0)
    parser.add_argument('--aV',
                        help='Sparsity (L1) regularization of V term.',
                        required=False,
                        default=0)
    parser.add_argument('--verbose',
                        help="Print current status to terminal. (True/False).",
                        required=False,
                        action="store_true")
    parser.add_argument('--seed',
                        help="Random seed.",
                        required=False,
                        default=1010)
    parser.add_argument('--max_iter',
                        help="Maximum number of iterations.",
                        required=False,
                        default=100)
    parser.add_argument('--term_tol',
                        help="Relative change in error before finish.",
                        required=False,
                        default=1e-25)
    parser.add_argument('--out_dir',
                        help="Path to output directory",
                        required=False,
                        default='.')
    parser.add_argument('--save_clust',
                        help="Save cluster assignments for each interation to an assignment matrix",
                        required=False,
                        action="store_true")
    parser.add_argument('--kill_factors',
                        help="Save cluster assignments for each interation to an assignment matrix",
                        required=False,
                        action="store_true")
    parser.add_argument('--track_objective',
                        help="Save cluster assignments for each interation to an assignment matrix",
                        required=False,
                        action="store_true")
    parser.add_argument('--save_USV',
                        help="Save lower dimensional matrices at every iteration",
                        required=False,
                        action="store_true")
    parser.add_argument('--device',
                        help="Select Device. Default is cuda:0. Options are cuda:0/cuda:1/cpu",
                        required=False,
                        default="cuda:0")
    parser.add_argument('--store_effective',
                        help="Save effective regularization parameters when not using legacy mode. Default is False",
                        required=False,
                        default=False)
    parser.add_argument('--legacy',
                        help="Use the legacy update. The new update improves selection of lU and lV.",
                        required=False,
                        action='store_true')
    args = parser.parse_args()
    runNMTF(args)
