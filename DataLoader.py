import time
import torch
import pandas as pd
import numpy as np
import anndata


class DataLoader:
    def __init__(self, verbose):
        self.verbose = verbose

    def from_text(self, datafile, delimiter='\t', header=None):
        """
        Matrix file that is plain text.
        Args:
            datafile: Path to the text file containing data.
            delimiter: Delimiter used to separate values in the file. Default is '\t' (tab).
            header: Row number(s) to use as the column names. Default is None.
        """
        start_time = time.time() if self.verbose else None
        if self.verbose:
            print('Starting to read data from {0:s}'.format(datafile))
        df = pd.read_csv(datafile, sep=delimiter, header=header, dtype=np.float32)
        df = df.to_numpy()
        if self.verbose:
            print('Time to read file: {0:.3f}'.format(time.time() - start_time))
        return torch.from_numpy(df), df.shape

    def from_pt(self, datafile):
        """
        Args:
            datafile: The path to the data file to be loaded using torch.load(). These should be .pt files containing
            a torch.tensor that will you wish to factorize.
        """
        start_time = time.time() if self.verbose else None
        if self.verbose:
            print('Starting to read data from {0:s}'.format(datafile))
        x = torch.load(datafile)
        if self.verbose:
            print('Time to read file: {0:.3f}'.format(time.time() - start_time))
        return x, x.shape

    def from_h5ad(self, datafile):
        """
        Loads in an anndata X data to SCOTCH.

        Args:
            datafile: A string specifying the path to the h5ad file that contains the data to be read.

        Returns:
            A tuple containing the AnnData object created from the h5ad file and its shape.
        """
        start_time = time.time() if self.verbose else None
        if self.verbose:
            print("Start reading data from {0:s}".format(datafile))
        x = anndata.read_h5ad(datafile)
        if self.verbose:
            print("Time to read file: {0:.3f}".format(time.time() - start_time))
        return x, x.shape
