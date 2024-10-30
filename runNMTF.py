import argparse
import NMTF as fact
import os
import cProfile


def main(args):
    if args.save_USV:
        mod = fact.NMTF(verbose=args.verbose, max_iter=args.max_iter,
                        seed=args.seed, term_tol=args.term_tol, l_u=args.lU,
                        l_v=args.lV, a_u=args.aU, a_v=args.aV, k1=args.k1,
                        k2=args.k2, save_clust=args.save_clust, track_objective=args.track_objective, kill_factors=args.kill_factors,
                        out_path=args.out_dir, device=args.device)
    else:
        mod = fact.NMTF(verbose=args.verbose, max_iter=args.max_iter,
                        seed=args.seed, term_tol=args.term_tol, l_u=args.lU,
                        l_v=args.lV, a_u=args.aU, a_v=args.aV, k1=args.k1,
                        k2=args.k2, save_clust=args.save_clust, track_objective=args.track_objective, kill_factors=args.kill_factors,
                        device=args.device)
    file_parts = os.path.splitext(args.in_file)
    if file_parts[1] == '.pt':
        mod.load_data_from_pt(args.in_file)
    else:
        mod.load_data_from_text(args.in_file)
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
                        default = -999)
    parser.add_argument('--k2',
                        help='lower dimension of the column factors (V).',
                        required=False,
                        default = -999)
    parser.add_argument('--test_multiple',
                        help="file containing test k1 and k2 in two tab delimited columns.",
                        required=False,
                        default = '')
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
                        default=1e-5)
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
    args = parser.parse_args()
    main(args)
