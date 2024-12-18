import time
import torch
import pandas as pd
import numpy as np
import anndata


class DataLoader:
    def __init__(self, verbose):
        self.verbose = verbose

    def from_text(self, datafile, delimiter='\t', header=None):
        start_time = time.time() if self.verbose else None
        if self.verbose:
            print('Starting to read data from {0:s}'.format(datafile))
        df = pd.read_csv(datafile, sep=delimiter, header=header, dtype=np.float32)
        df = df.to_numpy()
        if self.verbose:
            print('Time to read file: {0:.3f}'.format(time.time() - start_time))
        return torch.from_numpy(df), df.shape

    def from_pt(self, datafile):
        start_time = time.time() if self.verbose else None
        if self.verbose:
            print('Starting to read data from {0:s}'.format(datafile))
        x = torch.load(datafile)
        if self.verbose:
            print('Time to read file: {0:.3f}'.format(time.time() - start_time))
        return x, x.shape

    def from_h5ad(self, datafile):
        start_time = time.time() if self.verbose else None
        if self.verbose:
            print("Start reading data from {0:s}".format(datafile))
        x = anndata.read_h5ad(datafile)
        if self.verbose:
            print("Time to read file: {0:.3f}".format(time.time() - start_time))
        return x, x.shape
