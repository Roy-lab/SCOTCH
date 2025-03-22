import torch
import pandas as pd
import numpy as np
import argparse 
import os


def convert_text_to_pt(args):
    """
    Converts a delimited text file to a PyTorch tensor (.pt) file.

    This function reads a delimited file using pandas, converts the data to a NumPy array,
    then to a PyTorch tensor, and finally saves it as a `.pt` file. The output file will
    have the same name as the input file, but with a `.pt` extension.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments with the following attributes:

        - **in_file** (str): Path to the input delimited text file.
        - **delimiter** (str, optional): Delimiter used in the text file. Default is tab (`'\t'`).
        - **header** (int or None, optional): Number of header lines before data. Default is None.
        - **dtype** (numpy.dtype, optional): Data type for the PyTorch tensor. Default is `np.float32`.

    Returns
    -------
    None
    """
    file_parts = os.path.splitext(args.in_file)
    df = pd.read_csv(args.in_file, sep=args.delimiter, header=args.header, dtype=args.dtype)
    df = df.to_numpy()
    X = torch.from_numpy(df)
    torch.save(X, file_parts[0] + '.pt')


if __name__ == "__name__":
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
    parser.add_argument('--dtype',
                        help="Data type for .pt file. Default is np.float32",
                        required=False,
                        default=np.float32)
    args = parser.parse_args()
    convert_text_to_pt(args)
