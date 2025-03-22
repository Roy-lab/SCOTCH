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

        :param datafile: Path to the text file containing data.
        :type datafile: str

        :param delimiter: Delimiter used to separate values in the file. Default is '\t' (tab).
        :type delimiter: str, optional

        :param header: Row number(s) to use as the column names. Default is None.
        :type header: int or list of int, optional
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
        Load data from a PyTorch `.pt` file into SCOTCH.

        :param datafile: The path to the data file to be loaded using `torch.load()`.
                         It should be a `.pt` file containing a `torch.Tensor` to be factorized.
        :type datafile: str
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
        Loads an AnnData X data to SCOTCH.

        :param datafile: Path to the h5ad file containing the data to be read.
        :type datafile: str

        :returns: A tuple containing the AnnData object created from the h5ad file and its shape.
        :rtype: tuple (anndata.AnnData, tuple)
        """
        start_time = time.time() if self.verbose else None
        if self.verbose:
            print("Start reading data from {0:s}".format(datafile))
        x = anndata.read_h5ad(datafile)
        if self.verbose:
            print("Time to read file: {0:.3f}".format(time.time() - start_time))
        return x, x.shape
