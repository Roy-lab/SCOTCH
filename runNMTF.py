import argparse
import NMTF as fact
import cProfile


def main(args):
    mod = fact.NMTF(verbose=args.verbose, max_iter=args.max_iter,
                    seed=args.seed, term_tol=args.term_tol, l_u=args.lU,
                    l_v=args.lV, a_u=args.aU, a_v=args.aV, k1=args.k1, k2=args.k2, cpu=args.cpu)
    mod.load_data(args.in_file)
    mod.fit()
    mod.print_output(args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--in_file',
                        help='tab-delimited input matrix.',
                        required=True)
    parser.add_argument('--k1',
                        help='lower dimension of the row factors (U).',
                        required=True)
    parser.add_argument('--k2',
                        help='lower dimension of the column factors (V).',
                        required=True)
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
    parser.add_argument('--cpu',
                        help="if true default to CPU",
                        required=False,
                        action="store_true")
    args = parser.parse_args()
    main(args)