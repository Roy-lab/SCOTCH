import torch
import pandas as pd
import numpy as np
import argparse 
import os

def main(args):
    file_parts = os.path.splitext(args.in_file)
    df = pd.read_csv(args.in_file, sep=args.delimiter, header=args.header, dtype=args.dtype)
    df = df.to_numpy()
    X = torch.from_numpy(df)
    torch.save(X, file_parts[0] + '.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = __doc__,
        formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--in_file',
                        help="Deliminated file to convert to a .pt file",
                        required=True)
    parser.add_argument('--delimiter',
                        help="Delimiter of .txt file. Default is tab",
                        required=False,
                        default='\t')
    parser.add_argument('--header',
                        help="Number of lines before data. Default is zero",
                        required = False,
                        default = None)
    parser.add_argument('--data_type',
                        help="Data type for .pt file. Default is np.float32",
                        required=False,
                        default=np.float32)
    args = parser.parse_args()
    main(args)