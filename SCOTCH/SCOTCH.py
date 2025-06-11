from matplotlib.colors import ListedColormap
from statsmodels.tools.sm_exceptions import ValueWarning
from sympy.codegen import aug_assign

from SCOTCH.DataLoader import DataLoader
from SCOTCH.NMTF import NMTF

from scipy.sparse import issparse
import torch
import pandas as pd
import anndata
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np


import PyEnrichAnalyzer


class SCOTCH(NMTF):
    """
SCOTCH Class
============

The `SCOTCH` class extends from the `NMTF` class. It has a specific `__init__` method with several input parameters. The only required inputs are `k1` and `k2`.

**__init__ Input Parameters:**

- **k1, k2** (*int*):
  Lower dimension size of `U` and `V`. *(required)*

- **verbose** (*bool*, optional):
  If `True`, prints messages. *(default: True)*

- **max_iter** (*int*, optional):
  Maximum number of iterations. *(default: 100)*

- **seed** (*int*, optional):
  Random seed for initialization. *(default: 1001)*

- **term_tol** (*float*, optional):
  Relative error threshold for convergence. *(default: 1e-5)*

- **max_l_u** (*float*, optional):
  Maximum regularization on `U`. *(default: 0)*

- **max_l_v** (*float*, optional):
  Maximum regularization on `V`. *(default: 0)*

- **max_a_u** (*float*, optional):
  Maximum sparse regularization on `U`. *(default: 0, change at own risk)*

- **max_a_v** (*float*, optional):
  Maximum sparse regularization on `V`. *(default: 0, change at own risk)*

- **var_lambda** (*bool*, optional):
  If `True`, the regularization parameters `l_U` and `l_V` increase to max value using a sigmoid scheduler. Generally set to `False`. *(default: False)*

- **var_alpha** (*bool*, optional):
  If `True`, the regularization parameters `a_U` and `a_V` increase to max value using a sigmoid scheduler. Generally set to `False`. *(default: False)*

- **shape_param** (*float*, optional):
  Controls the rate of increase for `l_U`, `l_V`, `a_U`, and `a_V` when `var_lambda=True`. *(default: 10)*

- **mid_epoch_param** (*int*, optional):
  Sets the epoch where `l_U`, `l_V`, `a_U`, and `a_V` reach half of their max values if `var_lambda=True`. *(default: 5)*

- **init_style** (*str*, optional):
  Initialization method for SCOTCH. Should be either `"random"` or `"nnsvd"`. *(default: "random")*

- **save_clust** (*bool*, optional):
  Whether to save cluster assignments after each epoch. *(default: False)*

- **draw_intermediate_graph** (*bool*, optional):
  If `True`, draws and saves the matrix representation after each epoch. These can be saved as a GIF. *(default: False)*

- **track_objective** (*bool*, deprecated):
  *(default: False)*

- **kill_factors** (*bool*, optional):
  If `True`, SCOTCH will halt updates if any factors in `U` and `V` reach zero. *(default: False)*

- **device** (*str*, optional):
  Specifies the device to run SCOTCH on: `"cpu"` or `"cuda:"`. *(default: "cpu")*

- **out_path** (*str*, optional):
  Directory to save SCOTCH output files. *(default: '.')*
"""

    def __init__(self, k1, k2, verbose=True, max_iter=100, seed=1001, term_tol=1e-5,
                 max_l_u=0, max_l_v=0, max_a_u=0, max_a_v=0, var_lambda=False,
                 var_alpha=False, shape_param=10, mid_epoch_param=5,
                 init_style="random", save_clust=False, draw_intermediate_graph=False, save_intermediate=False,
                 track_objective=False, kill_factors=False, device="cpu", out_path='.'):

        super().__init__(verbose, max_iter, seed, term_tol, max_l_u, max_l_v, max_a_u, max_a_v, k1, k2, var_lambda,
                         var_alpha, shape_param, mid_epoch_param, init_style, save_clust, draw_intermediate_graph,
                         save_intermediate, track_objective, kill_factors, device, out_path)

        self.DataLoader = DataLoader(verbose)

    def add_data_from_file(self, file):
        """
        Loads matrix representation into PyTorch tensor object to run with SCOTCH.

        :param file: The file path to load data from and should have the valid extensions like '.pt', '.txt', or '.h5ad'.
        :type file: str
        """
        if not isinstance(file, str):
            raise TypeError('file must be a string')

        if not os.path.isfile(file):
            raise ValueError('The file does not exist')

        if not os.access(file, os.R_OK):
            raise ValueError('The file is not readable')

        shape = None
        _, file_extension = os.path.splitext(file)
        if file_extension == '.pt':
            self.X, shape = self.DataLoader.from_pt(file)
        elif file_extension == '.txt':
            self.X, shape = self.DataLoader.from_text(file)
        elif file_extension == '.h5ad':
            adata = self.DataLoader.from_h5ad(file)
            self.add_data_from_adata(adata)
        else:
            raise ValueError("Unsupported file type. Select .pt or .txt or .h5ad")

        self.X = self.X.to(dtype = self.dtype)

        if shape is not None:
            self.num_u = shape[0]
            self.num_v = shape[1]
        print("Data loaded successfully. Shape: ", self.num_u, self.num_v)
        return None

    def add_data_from_adata(self, adata):
        """
        Loads data from AnnData object into SCOTCH framework.

        :param adata: anndata.AnnData object to extract data from. Transforms adata.X to PyTorch object.
        :type adata: anndata.AnnData
        """
        if not isinstance(adata, anndata.AnnData):
            raise TypeError("adata must be an AnnData object")

        # Extract the X matrix and covert to a torch tensor
        X = adata.X
        if issparse(X):
            X_coo = X.tocoo()
            values = torch.tensor(X_coo.data)
            indices = torch.tensor(np.vstack((X_coo.row, X_coo.col)))
            X_tensor = torch.sparse_coo_tensor(indices, values, X_coo.shape)
            X_tensor = X_tensor.to_dense()
        else:
            X_tensor = torch.tensor(X)
        self.X = X_tensor
        self.X = self.X.to(dtype = self.dtype)
        self.num_u = self.X.shape[0]
        self.num_v = self.X.shape[1]
        return None

    def add_scotch_embeddings_to_adata(self, adata, prefix=""):
        """
        Adds SCOTCH objects to an AnnData object.

        :param prefix: Prefix to add to AnnData objects created by SCOTCH.
        :type prefix: str

        :param adata: The AnnData object to which SCOTCH embeddings will be added.
        :type adata: anndata.AnnData
        """
        if not isinstance(adata, anndata.AnnData):
            raise TypeError("adata must be an AnnData object")

        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        if len(prefix) > 0 and prefix[-1] != '_':
            prefix = prefix + '_'

        adata.obs[prefix + 'cell_clusters'] = pd.Categorical(self.U_assign.detach().numpy())
        adata.var[prefix + "gene_clusters"] = pd.Categorical(self.V_assign.t().detach().numpy())
        adata.obsm[prefix + 'cell_embedding'] = self.U.detach().numpy()
        adata.varm[prefix + 'gene_embedding'] = self.V.t().detach().numpy()
        adata.uns[prefix + 'S_matrix'] = self.S.detach().numpy()
        adata.obsm[prefix + 'P_embedding'] = self.P.detach().numpy()
        adata.varm[prefix + 'Q_embedding'] = self.Q.t().detach().numpy()
        adata.uns[prefix + 'reconstruction_error'] = self.reconstruction_error.detach().numpy()
        adata.uns[prefix + 'error'] = self.error.detach().numpy()
        return adata

    def add_scotch_embeddings_to_adata_from_file(self, adata, dir):
        if not isinstance(adata, anndata.AnnData):
            raise TypeError("adata must be an AnnData object")
        if  not isinstance(dir, str):
            raise TypeError("dir must be a string")
        if len(dir) == 0:
            raise ValueError("dir must be a non-empty string")
        if not os.path.isdir(dir):
            raise ValueError("dir must be a valid directory")

        U = pd.read_csv(dir + '/U.txt', header=None, sep='\t')
        V = pd.read_csv(dir + '/V.txt', header=None, sep='\t')
        S = pd.read_csv(dir + '/S.txt', header=None, sep='\t')

        #Fix possible \t\n end line which expresses itself in NaN. C++ version output have this... (Need to fix there)
        if U.iloc[:, -1].isna().all():
            U = U.iloc[:, :-1]

        if V.iloc[:, -1].isna().all():
            V= V.iloc[:, :-1]

        if S.iloc[:, -1].isna().all():
            S = S.iloc[:, :-1]


        error = pd.read_csv(dir + '/err.txt', header=None, sep='\t')

        self.U = torch.tensor(U.values).t()
        self.V = torch.tensor(V.values)
        self.S = torch.tensor(S.values)

        self.P = self.U @ self.S
        self.Q = self.S @ self.V

        self.reconstruction_error = torch.tensor(error.values)
        self.error = torch.tensor(error.values)

        self.assign_cluster()
        self.add_scotch_embeddings_to_adata(adata)
        return adata

    def make_adata_from_scotch(self, prefix=""):
        """
        Create an AnnData object from the given data.

        :param self: The instance of the class containing the data.
        :type self: object

        :param prefix: A string appended to the generated AnnData objects.
        :type prefix: str

        :returns: An AnnData object containing the processed data.
        :rtype: anndata.AnnData
        """
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a str")

        if len(prefix) > 0 and prefix[-1] != '_':
            prefix = prefix + '_'
        X = self.X
        adata = anndata.AnnData(X.numpy())
        adata = self.add_scotch_embeddings_to_adata(adata, prefix)
        return adata

    def make_top_regulators_list(self, adata, gene_cluster_id="gene_clusters", gene_embedding_id="gene_embedding",
                                 prefix=None, top_k=5):
        """
            Create a list of top regulators for each gene cluster based on gene embeddings.

            :param self: The instance of the class containing the data.
            :type self: object

            :param adata: An AnnData object containing single-cell gene expression data.
            :type adata: anndata.AnnData

            :param gene_cluster_id: The key for the gene clusters stored in `adata.var`. (default is "gene_clusters")
            :type gene_cluster_id: str

            :param gene_embedding_id: The key for the gene embedding matrix stored in `adata.varm`. (default is "gene_embedding")
            :type gene_embedding_id: str

            :param prefix: The string utilized when adding SCOTCH data to anndata. Use instead of gene_cluster_id and gene_embedding_id. (default is None)
            :type prefix: str

            :param top_k: The number of top genes to select per cluster (default is 5).
            :type top_k: int, optional

            :returns: A list of tuples, each containing the cluster index and the top `top_k` genes for that cluster.
            :rtype: list of tuples
        """
        if not isinstance(prefix, str) and prefix is not None:
            raise TypeError("prefix must be a string if passed.")

        if prefix is not None:
            if len(prefix) > 0 and prefix[-1] != '_':
                prefix = prefix + '_'
            gene_cluster_id = prefix + gene_cluster_id
            gene_embedding_id = prefix + gene_embedding_id

        if not isinstance(adata, anndata.AnnData):
            raise TypeError("adata must be an AnnData object")

        if not isinstance(gene_cluster_id, str):
            raise TypeError("gene_cluster_id must be a string")

        if gene_cluster_id not in adata.var.columns:
            raise ValueError(f"gene_cluster_id '{gene_embedding_id}' not found in adata.var.columns")

        if not isinstance(gene_embedding_id, str):
            raise TypeError("gene_embedding_id must be a string")

        if gene_embedding_id not in adata.varm.keys():
            raise ValueError(f"gene_embedding_id '{gene_embedding_id}' not found in adata.varm.keys()")

        if not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")

        if not top_k > 0:
            raise TypeError("top_k must greater than zero")

        cluster_gene = []
        k2 = adata.varm[gene_embedding_id].shape[1]

        for i in range(k2):
            adata_slice = adata[:, adata.var[gene_cluster_id] == i].copy()
            if adata_slice.shape[1] < top_k:
                print(
                    f"Warning: only {adata_slice.shape[1]} genes are assigned to cluster {i}. Including only these genes")
                top_slice = adata_slice.shape[1]
            else:
                top_slice = top_k
            embedding = adata_slice.varm[gene_embedding_id]
            indices = np.argsort(embedding[:, i])[-top_slice:, ][::-1]
            top_genes = adata_slice.var_names[indices]
            cluster_gene.append((i, top_genes))

        if self.verbose:
            for c_g in cluster_gene:
                print(f"Gene Cluster {c_g[0]}: {c_g[1]}")
        return cluster_gene

    def run_enrich_analyzer(self, adata, gene_cluster_id, go_regnet_file, fdr=0.05, test_type='persg', prefix='GO'):
        """
            Perform gene ontology (GO) enrichment analysis and store results in the AnnData object.

            :param self: The instance of the class containing the data.
            :type self: object

            :param adata: An AnnData object containing single-cell gene expression data.
            :type adata: anndata.AnnData

            :param gene_cluster_id: Identifier for the gene cluster to be analyzed.
            :type gene_cluster_id: str

            :param go_regnet_file: File path to the gene ontology (GO) enrichment file.
            :type go_regnet_file: str

            :param fdr: The false discovery rate threshold (default is 0.05).
            :type fdr: float, optional

            :param test_type: Type of statistical test to be performed (default is 'persg'). Valid options are 'persg' and 'fullgraph'.
            :type test_type: str, optional

            :param prefix: Prefix for storing enrichment results in the AnnData object (default is 'GO').
            :type prefix: str, optional

            :returns: None. Enrichment results are stored in `adata.uns[prefix + "enrichment"]`.
            :rtype: None
        """

        if not isinstance(adata, anndata.AnnData):
            raise TypeError("adata must be an AnnData object")

        if not isinstance(gene_cluster_id, str):
            raise TypeError("gene_cluster_id must be a string")

        if not gene_cluster_id in adata.var.columns:
            raise ValueError(f"gene_cluster_id '{gene_cluster_id}' not found in adata.var")

        if not isinstance(go_regnet_file, str):
            raise TypeError("go_regnet_file must be a string")

        if not os.path.isfile(go_regnet_file):
            raise ValueError(f"The go_regnet_file '{go_regnet_file}' does not exist")

        if not os.access(go_regnet_file, os.R_OK):
            raise ValueError(f"The go_regnet_file '{go_regnet_file}' is not readable")

        if not isinstance(fdr, float):
            raise TypeError("fdr must be a float")

        if fdr < 0 or fdr > 1:
            raise ValueError("fdr must between 0 and 1")

        if not isinstance(test_type, str):
            raise TypeError("test_type must be a string")

        if test_type != "persg" and test_type != "fullgraph":
            raise ValueError("test_type must be 'persg' or 'fullgraph'")

        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")

        if len(prefix) > 0 and prefix[-1] != '_':
            prefix = prefix + '_'


        EA = PyEnrichAnalyzer.PyEnrichAnalyzer_core.Framework()
        enrichment = EA.runEnrichAnalyzer(
            adata.var.to_dict(orient='index'),
            gene_cluster_id,
            #adata.var_names.to_list(),
            go_regnet_file,
            fdr,
            test_type)
        df = pd.DataFrame(enrichment)
        df.rename({'SubGraphName': 'gene cluster'}, axis=1, inplace=True)
        adata.uns[prefix + 'enrichment'] = df

    def visualize_enrichment_bubbleplots(self, adata, enrich_object_id,
                                         gene_cluster_id='gene cluster',
                                         term_id='TermName',
                                         FC_id='Foldenr',
                                         q_val_id='CorrPval',
                                         top_k=5,
                                         max_point_size=100,
                                         palette='viridis',
                                         gene_cluster_set=None,
                                         ax=None):
        """
        Visualize enrichment results as a bubble plot, where the size of the bubbles represents log2 fold change
        and the color represents -log10 p-values.

        :param self: The instance of the class containing the data.
        :type self: object

        :param adata: An AnnData object containing single-cell gene expression data.
        :type adata: anndata.AnnData

        :param enrich_object_id: The key in `adata.uns` containing the enrichment data.
        :type enrich_object_id: str

        :param gene_cluster_id: The column name representing gene clusters in the enrichment data (default is 'gene cluster').
        :type gene_cluster_id: str, optional

        :param term_id: The column name representing terms in the enrichment data (default is 'TermName').
        :type term_id: str, optional

        :param FC_id: The column name representing fold change values in the enrichment data (default is 'Foldenr').
        :type FC_id: str, optional

        :param q_val_id: The column name representing the corrected p-values in the enrichment data (default is 'CorrPval').
        :type q_val_id: str, optional

        :param top_k: The number of top terms to select per gene cluster (default is 5).
        :type top_k: int, optional

        :param max_point_size: The maximum size of the bubbles in the plot (default is 100).
        :type max_point_size: int, optional

        :param palette: The color palette used for the plot (default is 'viridis').
        :type palette: str, optional

        :returns: A `matplotlib.figure.Figure` object containing the bubble plot.
        :rtype: matplotlib.figure.Figure
        """

        # Validate adata input
        if adata is None or not hasattr(adata, 'uns'):
            raise ValueError(
                "The 'adata' parameter is required and must be a valid AnnData object with an 'uns' attribute.")

        # Validate enrich_object_id
        if enrich_object_id not in adata.uns:
            raise KeyError(f"The key '{enrich_object_id}' does not exist in 'adata.uns'. Please check the input.")

        # Validate that 'adata.uns[enrich_object_id]' contains data
        enrich_data = adata.uns[enrich_object_id]
        if not isinstance(enrich_data, dict) or len(enrich_data) == 0:
            raise ValueError(
                f"The 'adata.uns[{enrich_object_id}]' object is empty or invalid. Ensure it contains enrichment data.")

        # Validate top_k and max_point_size
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("The 'top_k' parameter must be a positive integer.")

        if not isinstance(max_point_size, (int, float)) or max_point_size <= 0:
            raise ValueError("The 'max_point_size' parameter must be a positive number.")

        # Validate ax if provided
        if ax is not None and not hasattr(ax, 'plot'):
            raise ValueError("The 'ax' parameter, if provided, must be a matplotlib Axes object.")

        # If gene_cluster_set is provided, validate it
        if gene_cluster_set is not None and not isinstance(gene_cluster_set, (list, set)):
            raise ValueError("The 'gene_cluster_set' parameter must be a list or set of genes.")

        enrichment = adata.uns[enrich_object_id]


        enrichment_topk_terms = enrichment.groupby(gene_cluster_id).apply(lambda grp: grp.nsmallest(top_k, q_val_id),
                                                                          include_groups=False)[term_id]
        filtered_enrichment = enrichment[enrichment[term_id].isin(enrichment_topk_terms)].copy()
        filtered_enrichment["log2FC"] = np.log2(filtered_enrichment[FC_id])
        filtered_enrichment["-log10QVal"] = -np.log10(filtered_enrichment[q_val_id])

        filtered_enrichment[term_id] = pd.Categorical(filtered_enrichment[term_id])
        if gene_cluster_set is not None:
            filtered_enrichment[gene_cluster_id] = pd.Categorical(filtered_enrichment[gene_cluster_id],
                                                                  categories=gene_cluster_set, ordered=True)
        else:
            filtered_enrichment[gene_cluster_id] = pd.Categorical(filtered_enrichment[gene_cluster_id])

        filtered_enrichment = self._fill_missing_categories(filtered_enrichment,
                                                            categorical_columns=[gene_cluster_id, term_id],
                                                            fill_value=0)

        categories = filtered_enrichment[gene_cluster_id].unique()

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 15))

        sns.scatterplot(
            data=filtered_enrichment,
            x=gene_cluster_id,
            y=term_id,
            size='log2FC',
            hue='-log10QVal',
            sizes=(0, max_point_size),
            palette=palette,
            zorder=3
        )
        plt.grid(True, zorder=1)
        plt.xlim(-0.5, len(categories) - 0.5)

        if fig is not None:
            ax.set_ylabel(None)
            ax.set_xlabel('Gene Cluster')
            plt.show()
        return fig

    def visualize_marker_gene_bubbleplot_per_cell_cluster(self, adata,
                                                          cell_cluster_id='cell_clusters',
                                                          gene_cluster_id='gene_clusters',
                                                          gene_embedding_id='gene_embedding',
                                                          prefix=None,
                                                          top_k=5,
                                                          max_point_size=300,
                                                          palette='viridis',
                                                          ax=None):
        """
        Visualize marker gene expression as a bubble plot, where the size of the bubbles represents the
        percent of non-zero counts and the color represents the mean marker expression. Top_k are selected for each
        gene cluster. Genes names are follows by the gene cluster that each gene corresponds to.

        :param self: The instance of the class containing the data.
        :type self: object

        :param adata: An AnnData object containing single-cell gene expression data.
        :type adata: anndata.AnnData

        :param cell_cluster_id: The column name representing cell clusters in `adata.obs`. (default is 'cell_clusters')
        :type cell_cluster_id: str

        :param gene_cluster_id: The column name representing gene clusters in `adata.var`. (default is 'gene_clusters')
        :type gene_cluster_id: str

        :param gene_embedding_id: The identifier for the gene embedding used for selecting top markers. (default is 'gene_embedding')
        :type gene_embedding_id: str

        :param prefix: The string utilized when adding SCOTCH data to anndata. Use instead of gene_cluster_id and gene_embedding_id. (default is None)
        :type prefix: str

        :param top_k: The number of top markers to consider per gene cluster (default is 5).
        :type top_k: int, optional

        :param max_point_size: The maximum size of the bubbles in the plot (default is 300).
        :type max_point_size: int, optional

        :param palette: The color palette used for the plot (default is 'viridis').
        :type palette: str, optional

        :param ax: The matplotlib.axes.axes object to plot in. If none, new figure is generated and returned (default is None)
        :type ax: matplotlib.axes.Axes, optional

        :returns: A `matplotlib.figure.Figure` object containing the bubble plot if ax not passed. Else returns none.
        :rtype: matplotlib.figure.Figure or None
        """
        if prefix is not None:
            if not isinstance(prefix, str):
                raise TypeError("`prefix` must be a string if provided.")
            if len(prefix) > 0 and prefix[-1] != '_':
                prefix = prefix + '_'
            cell_cluster_id = prefix + cell_cluster_id
            gene_cluster_id = prefix + gene_cluster_id
            gene_embedding_id = prefix + gene_embedding_id

        # Safety checks for `adata`
        if not isinstance(adata, anndata.AnnData):
            raise TypeError("`adata` must be an AnnData object.")

        # Safety checks for `cell_cluster_id`
        if not isinstance(cell_cluster_id, str):
            raise TypeError("`cell_cluster_id` must be a string.")
        if cell_cluster_id not in adata.obs.columns:
            raise ValueError(f"`cell_cluster_id` '{cell_cluster_id}' was not found in `adata.obs.columns`.")

        # Safety checks for `gene_cluster_id`
        if not isinstance(gene_cluster_id, str):
            raise TypeError("`gene_cluster_id` must be a string.")
        if gene_cluster_id not in adata.var.columns:
            raise ValueError(f"`gene_cluster_id` '{gene_cluster_id}' was not found in `adata.var.columns`.")

        # Safety checks for `gene_embedding_id`
        if not isinstance(gene_embedding_id, str):
            raise TypeError("`gene_embedding_id` must be a string.")
        if gene_embedding_id not in adata.varm.keys():
            raise ValueError(f"`gene_embedding_id` '{gene_embedding_id}' was not found in `adata.varm.keys()`.")

        # Safety checks for `top_k`
        if not isinstance(top_k, int):
            raise TypeError("`top_k` must be an integer.")
        if top_k <= 0:
            raise ValueError("`top_k` must be greater than zero.")

        # Safety checks for `max_point_size`
        if not isinstance(max_point_size, (int, float)):
            raise TypeError("`max_point_size` must be a number (int or float).")
        if max_point_size <= 0:
            raise ValueError("`max_point_size` must be greater than zero.")

        # Safety checks for `palette`
        if not isinstance(palette, str):
            raise TypeError("`palette` must be a string.")
        try:
            _ = plt.get_cmap(palette)  # Check if palette exists
        except ValueError:
            raise ValueError(f"'{palette}' is not a valid color palette. Please choose a valid matplotlib colormap.")

        # Safety checks for `ax`
        if ax is not None and not isinstance(ax, plt.Axes):
            raise TypeError("`ax` must be a matplotlib.axes.Axes object or None.")

        markers = self.make_top_regulators_list(adata=adata, gene_cluster_id=gene_cluster_id,
                                                gene_embedding_id=gene_embedding_id, top_k=top_k)

        marker_list = [gene for top_gene in markers for gene in top_gene[1]]
        marker_list_idx = adata.var_names.isin(marker_list)

        marker_expression = adata[:, marker_list_idx]

        mean_marker_expression = marker_expression.to_df().groupby(adata.obs[cell_cluster_id], observed=False).mean()
        percent_marker_expression = (marker_expression.to_df() > 0).groupby(adata.obs[cell_cluster_id],
                                                                            observed=False).mean()

        data = []
        for marker in markers:
            gene_cluster = marker[0]
            genes = marker[1]

            # Iterate over the genes
            for gene in genes:
                # Get the mean expression for this gene in this cluster
                mean_exp = mean_marker_expression.loc[:, gene]

                # Get the percent non-zero expression for this gene in this cluster
                non_zero = percent_marker_expression.loc[:, gene]

                for idx, cluster in enumerate(mean_marker_expression.index):
                    data.append([gene_cluster, gene, mean_exp[idx], cluster, non_zero[idx]])

        # Convert the data list into a DataFrame
        df = pd.DataFrame(data, columns=["Gene Cluster", "Gene", "Mean Marker Expression", "Cell Cluster",
                                         "Percent Nonzero Count"])
        df['Gene Cluster'] = df['Gene Cluster'].astype('category')
        df['Cell Cluster'] = df['Cell Cluster'].astype('string').astype('category')
        df['Gene'] = df['Gene'].astype('category')
        df = df.sort_values(by=["Gene Cluster", 'Gene'])
        df["Gene_Gene_Cluster"] = df.apply(lambda row: f'{row["Gene"]}: {row["Gene Cluster"]}', axis=1)

        max_expression = np.ceil(max(df['Mean Marker Expression']))

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 15))

        sns.scatterplot(
            data=df,
            x='Cell Cluster',
            y='Gene_Gene_Cluster',
            hue='Mean Marker Expression',
            size='Percent Nonzero Count',
            sizes=(0, max_point_size),  # control size range of bubbles
            palette=palette,  # color palette
            edgecolor='w',  # white edge for visibility
            hue_norm=(0, max_expression),
            ax=ax,
            zorder=3
        )

        if fig is not None:
            # Adding colorbar and labels
            ax.set_ylabel(None)
            ax.set_xlabel('Cell Cluster')
            plt.grid(True, zorder=1)

        return fig

    def visualize_marker_gene_bubbleplot_per_gene_cluster(self, adata,
                                                          gene_cluster_id='gene_clusters',
                                                          gene_embedding_id="gene_embedding",
                                                          prefix=None,
                                                          top_k=5,
                                                          max_point_size=300,
                                                          palette='viridis',
                                                          ax=None):
        """
        Visualize marker gene expression as a bubble plot, where the size of the bubbles represents the
        percent of non-zero counts and the color represents the mean marker expression. Top_k are selected for each
        gene cluster. Genes names are follows by the gene cluster that each gene corresponds to.

        :param self: The instance of the class containing the data.
        :type self: object

        :param adata: An AnnData object containing single-cell gene expression data.
        :type adata: anndata.AnnData

        :param gene_cluster_id: The identifier for the gene cluster used for selecting top markers.
        :type gene_cluster_id: str, optional

        :param gene_embedding_id: The identifier for the gene embedding used for selecting top markers.
        :type gene_embedding_id: str, optional

        :param prefix: The prefix string used when adding SCOTCH data to anndata object.
        :type prefix: str, optional

        :param top_k: The number of top markers to consider per gene cluster (default is 5).
        :type top_k: int, optional

        :param max_point_size: The maximum size of the bubbles in the plot (default is 300).
        :type max_point_size: int, optional

        :param palette: The color palette used for the plot (default is 'viridis').
        :type palette: str, optional

        :param ax: A `matplotlib.axes.Axes` object to plot the bubble plot. If not provided a new figures is generated
        :type ax: matplotlib.axes.Axes, optional

        :returns: A `matplotlib.figure.Figure` object containing the bubble plot.
        :rtype: matplotlib.figure.Figure
        """

        if prefix is not None:
            if not isinstance(prefix, str):
                raise TypeError("`prefix` must be a string if provided.")
            if len(prefix) > 0 and prefix[-1] != '_':
                prefix = prefix + '_'
            gene_cluster_id = prefix + gene_cluster_id
            gene_embedding_id = prefix + gene_embedding_id

        # Safety checks for `adata`
        if not isinstance(adata, anndata.AnnData):
            raise TypeError("`adata` must be an AnnData object.")

        # Safety checks for `gene_cluster_id`
        if not isinstance(gene_cluster_id, str):
            raise TypeError("`gene_cluster_id` must be a string.")
        if gene_cluster_id not in adata.var.columns:
            raise ValueError(f"`gene_cluster_id` '{gene_cluster_id}' was not found in `adata.var.columns`.")

        # Safety checks for `gene_embedding_id`
        if not isinstance(gene_embedding_id, str):
            raise TypeError("`gene_embedding_id` must be a string.")
        if gene_embedding_id not in adata.varm.keys():
            raise ValueError(f"`gene_embedding_id` '{gene_embedding_id}' was not found in `adata.varm` keys.")

        # Safety checks for `top_k`
        if not isinstance(top_k, int):
            raise TypeError("`top_k` must be an integer.")
        if top_k <= 0:
            raise ValueError("`top_k` must be greater than zero.")

        # Safety checks for `max_point_size`
        if not isinstance(max_point_size, (int, float)):
            raise TypeError("`max_point_size` must be a number (int or float).")
        if max_point_size <= 0:
            raise ValueError("`max_point_size` must be greater than zero.")

        # Safety checks for `palette`
        if not isinstance(palette, str):
            raise TypeError("`palette` must be a string.")
        try:
            # Check if palette is a valid matplotlib colormap
            _ = plt.get_cmap(palette)
        except ValueError:
            raise ValueError(f"'{palette}' is not a valid color palette. Please choose a valid matplotlib colormap.")

        # Safety checks for `ax`
        if ax is not None and not isinstance(ax, plt.Axes):
            raise TypeError("`ax` must be a matplotlib.axes.Axes object or None.")

        markers = self.make_top_regulators_list(adata=adata, gene_cluster_id=gene_cluster_id,
                                                gene_embedding_id=gene_embedding_id, top_k=top_k)

        marker_list = pd.Series([gene for top_gene in markers for gene in top_gene[1]])
        marker_list_idx = adata.var_names.isin(marker_list)

        marker_data = adata[:, marker_list_idx]
        marker_V_matrix = pd.DataFrame(marker_data.varm[gene_embedding_id])
        marker_V_matrix.index = marker_data.var_names
        marker_V_matrix = marker_V_matrix.reset_index()
        marker_V_matrix = marker_V_matrix.melt(id_vars=['index'], var_name="gene cluster", value_name='embedding')
        marker_V_order = [str(i) for i in marker_V_matrix['gene cluster'].unique()]
        marker_V_matrix['gene cluster'] = pd.Categorical([str(i) for i in marker_V_matrix['gene cluster']],
                                                         categories=marker_V_order, ordered=True)
        marker_V_matrix['index'] = pd.Categorical(marker_V_matrix['index'], categories=pd.unique(marker_list),
                                                  ordered=True)

        max_embedding = np.ceil(max(marker_V_matrix['embedding']))

        categories = marker_V_matrix['gene cluster'].unique()

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 15))

        sns.scatterplot(
            data=marker_V_matrix,
            x='gene cluster',
            y='index',
            hue='embedding',
            size='embedding',
            sizes=(0, max_point_size),  # control size range of bubbles
            palette=palette,  # color palette
            edgecolor='w',  # white edge for visibility
            hue_norm=(0, max_embedding),
            ax=ax,
            zorder=3
        )

        plt.grid(True, zorder=1)
        plt.xlim(-0.5, len(categories) - 0.5)
        return fig

    def plot_cooccurrence_proportions(self, adata, field_1="cell_clusters", field_2="sample", cmap="Reds", ax=None):
        """
        Generate a heatmap of co-occurrence proportions between two categorical variables.

        This function creates a heatmap to visualize the co-occurrence proportions of two categorical variables
        stored in the `adata.obs` dataframe. The rows correspond to values from `field_1` and the
        columns correspond to values from `field_2`. The heatmap displays normalized proportions per row.

        :param adata: An AnnData object containing the single-cell data.
            It must include `adata.obs` with `field_1` and `field_2` as categorical variables.
        :type adata: anndata.AnnData

        :param field_1: The name of the first categorical variable in `adata.obs` (used for heatmap rows).
            Its values will define the rows in the heatmap. (default is 'cell_clusters')
        :type field_1: str, optional

        :param field_2: The name of the second categorical variable in `adata.obs` (used for heatmap columns).
            Its values will define the columns in the heatmap. (Default is 'sample'.)
        :type field_2: str, optional

        :param cmap: The color map used for the heatmap. It must be a valid Matplotlib colormap, with "Reds"
            as the default. This determines the gradient colors representing value intensity in the heatmap.
        :type cmap: str, optional

        :param ax: A Matplotlib `Axes` object on which the heatmap will be plotted. If `None`, a new figure
            and axis are created, and the function returns the generated figure. If provided, the heatmap
            is plotted on the existing axis, and no figure is returned.
        :type ax: matplotlib.axes.Axes or None, optional

        :returns: A Matplotlib `Figure` object containing the heatmap of co-occurrence proportions,
            if a new figure is generated. If `ax` is provided, the function returns None.
        :rtype: matplotlib.figure.Figure or None

        :raises TypeError: If `adata` is not an `AnnData` object, or `field_1` and `field_2` are not strings.
        :raises ValueError: If `field_1` or `field_2` are not found in `adata.obs.columns`.
        """

        if not isinstance(adata, anndata.AnnData):
            raise TypeError('adata must be an Adata object')
        if not isinstance(field_1, str):
            raise TypeError('variable_1 must be a string')
        if not isinstance(field_2, str):
            raise TypeError('variable_2 must be a string')

        if field_1 not in adata.obs.columns:
            raise ValueError(f"'{field_1}' not found in adata.obs.")
        if field_2 not in adata.obs.columns:
            raise ValueError(f"'{field_2}' not found in adata.obs.")

        if not isinstance(cmap, str):
            raise TypeError("`cmap` must be a string representing a valid Matplotlib colormap.")
        try:
            # Check if the colormap is valid
            plt.get_cmap(cmap)
        except ValueError:
            raise ValueError(f"'{cmap}' is not a valid Matplotlib colormap.")

        if ax is not None and not isinstance(ax, plt.Axes):
            raise TypeError("`ax` must be a `matplotlib.axes.Axes` object or `None`.")

        # Extract the categorical variables from adata.obs
        var_1 = adata.obs[field_1]
        var_2 = adata.obs[field_2]

        # Create a contingency table of co-occurrence counts
        co_occurrence_counts = pd.crosstab(var_1, var_2)

        # Normalize by rows to get proportions
        co_occurrence_proportions = co_occurrence_counts.div(co_occurrence_counts.sum(axis=1), axis=0)

        # Plot the heatmap
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 15))

        sns.heatmap(co_occurrence_proportions, cmap=cmap, annot=True, fmt='.2f', ax=ax, vmin=0, vmax=1, cbar=False)

        if fig is not None:
            plt.tight_layout()
            plt.show()

        return fig

    def plot_element_count_heatmap(self, adata, field='cell_clusters', orientation='vertical', cmap='Blues', v_min=0,
                                   ax=None):
        """
        This function produces a heatmap displaying the count of unique elements in a specific column of an `AnnData` object.
        The orientation of the heatmap can be controlled, along with customization options like the color map and axis.

        :param adata: An AnnData object containing the single-cell data.
            This parameter stores observations and variables, including metadata used for the analysis.
        :type adata: anndata.AnnData

        :param field: The column in `adata.obs` for which the counts should be calculated.
            The unique values in this column are counted and visualized in the heatmap. Default is cell_clusters
        :type field: str optional.

        :param orientation: The orientation of the heatmap (either rows or columns represent the elements being counted).
            Acceptable values are 'vertical' (default) or 'horizontal'.
        :type orientation: str, optional

        :param cmap: The color map used to style the heatmap. For example, use 'Blues' for a blue shade gradient.
            Defaults to 'Blues'.
        :type cmap: str, optional

        :param v_min: The minimum value for the heatmap color scale.
            This is useful to set a threshold for visualization. Default is 0.
        :type v_min: int, optional

        :param ax: A matplotlib `Axes` object onto which the heatmap will be drawn.
            If not provided, a new figure and axes will be created.
        :type ax: matplotlib.axes.Axes, optional

        :return: A matplotlib `Figure` object containing the heatmap.
            If `ax` is provided, then the returned `Figure` will be `None`, as the plot will be drawn on the given `Axes`.
        :rtype: matplotlib.figure.Figure or None
        """
        if not isinstance(adata, anndata.AnnData):
            raise TypeError("`adata` must be an instance of `anndata.AnnData`.")

        if not isinstance(field, str):
            raise TypeError("`field` must be a string.")

        if field not in adata.obs.columns:
            raise ValueError(f"The field `{field}` is not found in `adata.obs` columns.")

        if orientation not in ["vertical", "horizontal"]:
            raise ValueError("`orientation` must be either 'vertical' or 'horizontal'.")

        try:
            plt.get_cmap(cmap)  # Access colormap to verify it's valid
        except ValueError:
            raise ValueError(f"The colormap `{cmap}` is not valid. Check available Matplotlib colormaps.")

        if not (isinstance(v_min, (int, float)) and v_min >= 0):
            raise ValueError("`v_min` must be a non-negative number (int or float).")

        if ax is not None and not isinstance(ax, plt.Axes):
            raise TypeError(f"`ax` must be an instance of matplotlib `Axes` or None. Received type: {type(ax)}.")

        # Count the occurrences of each unique element in the field
        element_counts = adata.obs[field].value_counts(sort=False)
        # min_counts = min(element_counts)
        max_counts = max(element_counts)

        count_df = pd.DataFrame(element_counts)

        # Create a heatmap based on the element counts
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 15))

        if orientation == 'vertical':
            sns.heatmap(count_df, annot=True, fmt='d', cmap=cmap, cbar=False, ax=ax, vmin=v_min, vmax=max_counts)
            ax.set_xlabel(field)
        elif orientation == 'horizontal':
            sns.heatmap(count_df.T, annot=True, fmt='d', cmap=cmap, cbar=False, ax=ax, vmin=v_min, vmax=max_counts)
            ax.set_ylabel(field)
        else:
            raise ValueError("Orientation must be either 'vertical' or 'horizontal'.")

        if fig is not None:
            # Show the plot
            plt.tight_layout()
            plt.show()

        return fig


    def plot_S_matrix(self, adata, S_matrix_id="S_matrix", palette='viridis', ax=None):
        """
        Plot a heatmap of the S matrix stored in `adata.uns` under the given key (`S_matrix_id`).

        :param adata: An AnnData object containing the single-cell data.
        :type adata: anndata.AnnData

        :param S_matrix_id: The key in `adata.uns` where the S matrix is stored. Default is 'S_matrix'.
        :type S_matrix_id: str

        :param palette: The color palette to use for the heatmap. Default is 'viridis'.
        :type palette: str

        :param ax: A matplotlib Axes object to plot the heatmap on. If not provided, a new figure and Axes will be created.
        :type ax: matplotlib.axes.Axes, optional

        :return: A matplotlib Figure object containing the heatmap, or None if `ax` is provided.
        :rtype: matplotlib.figure.Figure or None
        """
        if not isinstance(adata, anndata.AnnData):
            raise TypeError("`adata` must be an instance of `anndata.AnnData`.")

        if not isinstance(S_matrix_id, str):
            raise TypeError("`S_matrix_id` must be a string.")

        if S_matrix_id not in adata.uns:
            raise ValueError(f"`S_matrix_id` '{S_matrix_id}' not found in `adata.uns`.")

        S = adata.uns[S_matrix_id]
        if not (isinstance(S, (np.ndarray, pd.DataFrame)) and S.ndim == 2):
            raise ValueError(f"The object linked to `S_matrix_id` '{S_matrix_id}' must be a 2D array or DataFrame.")

        try:
            plt.get_cmap(palette)  # Will raise a ValueError if the colormap is invalid
        except ValueError:
            raise ValueError(f"The colormap '{palette}' is not a valid matplotlib colormap.")

        if ax is not None and not isinstance(ax, plt.Axes):
            raise TypeError(f"If provided, `ax` must be a matplotlib `Axes` object. Received: {type(ax)}.")

        fig = None
        if ax is None:
            fix, ax = plt.subplots()
        if S_matrix_id in adata.uns:
            S = adata.uns[S_matrix_id]
            sns.heatmap(S, cmap='viridis', ax=ax, annot=False, cbar=False)
        else:
            raise ValueError(f"S_matrix_id '{S_matrix_id}' not found in adata.uns.")
        return fig

    def combined_enrichment_visualization(self, adata, enrich_object_id, top_k=5, max_point_size=100,
                                          palette='viridis', var1="cell_clusters", var2="sample", S_matrix_id=None):
        """
        Create a 2x3 subplot visualization combining enrichment bubble plots, element count heatmaps,
        co-occurrence proportions, and S matrix heatmaps.

        This visualization provides insights into cellular data enrichment and relationships between variables.

        Parameters
        ----------
        adata : anndata.AnnData
            An AnnData object containing single-cell data. Must contain `obs`, `uns`, and required data matrices.
        enrich_object_id : str
            The identifier for enrichment data in `adata.uns`.
        top_k : int, optional (default=5)
            The number of top enrichment terms to display in visualizations.
            Must be a positive integer.
        max_point_size : int, optional (default=100)
            The maximum size for the points in the bubble plot. Determines the largest bubble size.
        palette : str, optional (default='viridis')
            The color palette used for the bubble plot.
        var1 : str, optional
            The first variable for co-occurrence proportions. Must exist in `adata.obs.columns`.
        var2 : str, optional
            The second variable for co-occurrence proportions. Must exist in `adata.obs.columns`.
        S_matrix_id : str, optional
            The key in `adata.uns` for the additional heatmap matrix. The associated value must be a 2D matrix.

        Returns
        -------
        matplotlib.figure.Figure
            A matplotlib figure object containing the generated subplots.
        """

        # Validate 'top_k'
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"'top_k' must be a positive integer, but got {top_k}.")

        # Validate 'max_point_size'
        if not isinstance(max_point_size, int) or max_point_size <= 0:
            raise ValueError(f"'max_point_size' must be a positive integer, but got {max_point_size}.")

        # Validate 'adata'
        if not hasattr(adata, "obs") or not hasattr(adata, "uns"):
            raise ValueError("The input `adata` object must have both `obs` and `uns` attributes.")

        # Validate 'var1' and 'var2'
        if var1 is not None:
            if var1 not in adata.obs.columns:
                raise ValueError(f"The specified 'var1' ('{var1}') is not a valid column in 'adata.obs'.")
        if var2 is not None:
            if var2 not in adata.obs.columns:
                raise ValueError(f"The specified 'var2' ('{var2}') is not a valid column in 'adata.obs'.")

        # Validate 'S_matrix_id'
        if S_matrix_id is not None:
            if S_matrix_id not in adata.uns:
                raise ValueError(f"The specified 'S_matrix_id' ('{S_matrix_id}') is not found in 'adata.uns'.")
            S = adata.uns[S_matrix_id]
            if not hasattr(S, "shape") or len(S.shape) != 2:
                raise ValueError(
                    f"The 'S_matrix_id' value must be a 2D matrix, but got an invalid object of type: {type(S)}.")

        # Setup plot grid:
        if var2 is not None and var2 in adata.obs.columns:
            n_set_2 = adata.obs[var2].unique().shape[0]
        else:
            raise ValueError(f"var2 '{var2}' not found in adata.obs.")
        if S_matrix_id in adata.uns:
            S = adata.uns[S_matrix_id]
            gene_factors = S.shape[1]
            cell_factors = S.shape[0]
        else:
            raise ValueError(f"S_matrix_id '{S_matrix_id}' not found in adata.uns.")

        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(nrows=2, ncols=3, figure=fig,
                               width_ratios=[1, n_set_2, gene_factors],
                               height_ratios=[cell_factors, top_k * 0.5 * gene_factors])

        ax_A = fig.add_subplot(gs[0, 0])  # Panel A
        ax_B = fig.add_subplot(gs[0, 1])  # Panel B
        ax_C = fig.add_subplot(gs[0, 2])  # Panel C
        ax_D = fig.add_subplot(gs[1, 2])  # Panel D

        # Plot 1: Element Count Heatmap
        if var1 is not None:
            self.plot_element_count_heatmap(adata, var1, "vertical", ax=ax_A)
        else:
            raise ValueError("var1 parameter is required for the element count heatmap.")
        ax_A.set_xlabel(None)
        ax_A.set_ylabel(None)
        ax_A.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=True)
        ax_A.tick_params(axis='y', which='both', left=False, right=False, labelleft=True, labelright=False)

        # Plot 2: Co-occurrence Proportions
        if var1 is not None and var2 is not None:
            self.plot_cooccurrence_proportions(adata, var1, var2, ax=ax_B)
        else:
            raise ValueError("Both var1 and var2 are required for co-occurrence proportions.")
        ax_B.set_xlabel(None)
        ax_B.set_ylabel(None)
        ax_B.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=True)
        ax_B.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)

        # Plot S matrix
        self.plot_S_matrix(adata=adata, S_matrix_id=S_matrix_id, palette=palette, ax=ax_C)
        ax_C.set_xlabel(None)
        ax_C.set_ylabel(None)
        ax_C.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
        ax_C.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)

        # Plot 4: Enrichment Bubble Plot
        gene_subset = [str(i) for i in range(S.shape[1])]
        self.visualize_enrichment_bubbleplots(adata, enrich_object_id, top_k=5, max_point_size=300,
                                              gene_cluster_set=gene_subset, ax=ax_D)
        ax_D.set_ylabel(None)
        ax_D.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, labeltop=False)
        ax_D.tick_params(axis='y', which='both', left=False, right=False, labelleft=True, labelright=False)
        ax_D.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        # plt.show()
        return fig

    def combined_embedding_visualization(self, adata,
                                         gene_cluster_id = 'gene_clusters',
                                         gene_embedding_id='gene_embedding',
                                         top_k=5, max_point_size=100,
                                         palette='viridis',
                                         var1='cell_clusters', 
                                         var2='sample', 
                                         S_matrix_id="S_matrix",
                                         prefix = None):
        """
        Generate a combined visualization of embeddings from the data, with options to color by metadata.

        This function is designed to visualize embeddings (e.g., UMAP, PCA, t-SNE) stored in `adata`, optionally
        allowing users to color the points by specific metadata columns (like cell type or condition). It arranges
        one or more plots in a grid layout.

        Parameters
        ----------
        adata : anndata.AnnData
            An AnnData object containing single-cell data. Must contain embeddings in `adata.obsm`.
        gene_embedding_id : str, optional
            The key in `adata.obsm` where the embedding is stored (e.g., `'X_umap'` for UMAP). Default is `'gene_embedding'`.
        top_k : int, optional (default=5)
            the number of top features to display per gene cluster. Must be a positive integer.
        max_point_size : int, optional (default=2)
            Max point size for V bubble plot.
        palette : str, optional (default='viridis')
            The color palette to use for the scatter plot when coloring points.
        var1 : str, optional
            The primary variable (from `adata.obs`) to use for heatmap plotting (e.g., 'cell_clusters').
        var2 : str, optional
            The secondary variable (from `adata.obs`) for co-occurrence and proportions (e.g., 'sample').
        S_matrix_id : str, optional
            Key in `adata.uns` for an externally referenced matrix (e.g., factor matrix or count matrix).

         Returns
        -------
        matplotlib.figure.Figure
            A matplotlib figure containing the generated visualizations.

        """
        # Setup plot grid:
        if var2 is not None and var2 in adata.obs.columns:
            n_set_2 = adata.obs[var2].unique().shape[0]
        else:
            raise ValueError(f"var2 '{var2}' not found in adata.obs.")
        if S_matrix_id in adata.uns:
            S = adata.uns[S_matrix_id]
            gene_factors = S.shape[1]
            cell_factors = S.shape[0]
        else:
            raise ValueError(f"S_matrix_id '{S_matrix_id}' not found in adata.uns.")

        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(nrows=2, ncols=3, figure=fig,
                               width_ratios=[1, n_set_2, gene_factors],
                               height_ratios=[cell_factors, top_k * 0.5 * gene_factors])

        ax_A = fig.add_subplot(gs[0, 0])  # Panel A
        ax_B = fig.add_subplot(gs[0, 1])  # Panel B
        ax_C = fig.add_subplot(gs[0, 2])  # Panel C
        ax_D = fig.add_subplot(gs[1, 2])  # Panel D

        # Plot 1: Element Count Heatmap
        if var1 is not None:
            self.plot_element_count_heatmap(adata, var1, "vertical", ax=ax_A)
        else:
            raise ValueError("var1 parameter is required for the element count heatmap.")
        ax_A.set_xlabel(None)
        ax_A.set_ylabel(None)
        ax_A.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=True)
        ax_A.tick_params(axis='y', which='both', left=False, right=False, labelleft=True, labelright=False)

        # Plot 2: Co-occurrence Proportions
        if var1 is not None and var2 is not None:
            self.plot_cooccurrence_proportions(adata, var1, var2, ax=ax_B)
        else:
            raise ValueError("Both var1 and var2 are required for co-occurrence proportions.")
        ax_B.set_xlabel(None)
        ax_B.set_ylabel(None)
        ax_B.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=True)
        ax_B.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)

        # Plot S matrix
        self.plot_S_matrix(adata=adata, S_matrix_id=S_matrix_id, palette=palette, ax=ax_C)
        ax_C.set_xlabel(None)
        ax_C.set_ylabel(None)
        ax_C.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
        ax_C.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)

        # Plot 4: Enrichment Bubble Plot
        gene_subset = [str(i) for i in range(S.shape[1])]
        self.visualize_marker_gene_bubbleplot_per_gene_cluster(adata, gene_cluster_id=gene_cluster_id,
                                                               gene_embedding_id=gene_embedding_id,
                                                               top_k=5, max_point_size=max_point_size, ax=ax_D)
        ax_D.set_ylabel(None)
        ax_D.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, labeltop=False)
        ax_D.tick_params(axis='y', which='both', left=False, right=False, labelleft=True, labelright=False)
        ax_D.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        # plt.show()
        return fig

    def _fill_missing_categories(self, df, categorical_columns, fill_value=np.nan):
        """
        Fills missing combinations of categories in categorical columns with the specified fill_value.

        Parameters:
        - df: pandas DataFrame
        - categorical_columns: list of columns to consider for filling missing categories
        - fill_value: value to fill missing data with (default is NaN)

        Returns:
        - A DataFrame with all combinations of categories filled.
        """

        # Generate all possible combinations of the categorical columns
        all_combinations = pd.MultiIndex.from_product(
            [df[col].cat.categories for col in categorical_columns],
            names=categorical_columns
        )

        # Reindex the DataFrame to ensure all combinations are included
        df_filled = df.set_index(categorical_columns).reindex(all_combinations, fill_value=fill_value).reset_index()

        return df_filled

    def plot_reconstruction_error(self, adata):

        # Extract keys matching the pattern '*_reconstruction_error'
        reconstruction_error_keys = [key for key in adata.uns if key.endswith('_reconstruction_error')]

        # Create line plot for each key
        plt.figure(figsize=(12, 8))
        for key in reconstruction_error_keys:
            prefix = key.split('_reconstruction_error')[0]
            reconstruction_error = adata.uns[key].flatten()
            sns.lineplot(x=range(len(reconstruction_error)), y=reconstruction_error, label=prefix)

        plt.yscale('log')
        #plt.title('Reconstruction Errors (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.legend(title='Regularization', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def visualize_adata_factors(self,
                          adata,
                          U_factor_id = "cell_embedding",
                          V_factor_id = "gene_embedding",
                          S_matrix_id = "S_matrix",
                          prefix = None,
                          cmap='viridis', interp='nearest', max_u=1, max_v=1, max_x=1):
        """
        This function generates a visual representation of the NMTF factors, allowing users to specify
        the colormap and interpolation method used for image display.

        :param cmap: The colormap to be used for visualization. Default is 'viridis'.
        :type cmap: str, optional

        :param interp: The interpolation method to be used for image display. Default is 'nearest'.
        :type interp: str, optional

        :param max_u: The maximum for color scale. Value between [0, 1] where 1 represents the max value in U.
            Default is 1.
        :type max_u: float, optional

        :param max_v: The maximum for color scale. Value between [0, 1] where 1 represents the max value in V.
            Default is 1.
        :type max_v: float, optional

        :param max_x: The maximum for color scale. Value between [0, 1] where 1 represents the max value in X.
            Default is 1.
        :type max_x: float, optional

        :return: U, S, V  matrix heatmaps with X and product.
        :rtype: matplotlib.figure.Figure

        """
        fig = plt.figure(figsize=(16, 6))
        grids = gridspec.GridSpec(2, 3, wspace=0.1, width_ratios=(0.2, 0.4, 0.4), height_ratios=(0.3, 0.7))

        if prefix is not None and prefix[-1] != '_':
            prefix = prefix + '_'

        if prefix is not None:
            U_factor_id = prefix + U_factor_id
            V_factor_id = prefix + V_factor_id
            S_matrix_id = prefix + S_matrix_id

        U = torch.tensor(adata.obsm[U_factor_id])
        V = torch.tensor(adata.varm[V_factor_id]).t()
        S = torch.tensor(adata.uns[S_matrix_id])

        fig = plt.figure(figsize=(16, 6))
        grids = gridspec.GridSpec(2, 3, wspace=0.1, width_ratios=(0.2, 0.4, 0.4), height_ratios=(0.3, 0.7))

        U_viz = U.detach().numpy()
        U_viz = (U_viz - U_viz.min()) / (U_viz.max() - U_viz.min())
        ax1 = fig.add_subplot(grids[1, 0])
        ax1.imshow(U_viz, aspect="auto", cmap=cmap, interpolation=interp,
                  vmin=0, vmax=max_u)
        ax1.set_axis_off()
        # ax1.set_title("U Matrix")

        # Visualize S matrix
        ax2 = fig.add_subplot(grids[0, 0])
        ax2.imshow(S.t().detach().numpy(), aspect="auto", cmap=cmap, interpolation=interp)
        ax2.set_axis_off()
        # ax2.set_title("S Matrix")

        # Visualize V matrix
        V_viz = V.detach().numpy()
        V_viz = (V_viz - V_viz.min()) / (V_viz.max() - V_viz.min())
        ax3 = fig.add_subplot(grids[0, 1])
        ax3.imshow(V_viz, aspect="auto", cmap=cmap, interpolation=interp,
                   vmin=0, vmax=max_v)
        ax3.set_axis_off()
        # ax3.set_title("V Matrix")

        # Visualize X matrix
        X_est_viz = (U @ S @ V).detach().numpy()
        X_est_viz = (X_est_viz - X_est_viz.min())/(X_est_viz.max() - X_est_viz.min())
        ax4 = fig.add_subplot(grids[1, 1])
        ax4.imshow(X_est_viz, aspect="auto", cmap=cmap,
                   interpolation=interp, vmin=0, vmax=max_x)
        # ax4.set_title("X Matrix")
        ax4.set_axis_off()

        X_viz = adata.X.toarray()
        X_viz = (X_viz - X_viz.min()) / (X_viz.max() - X_viz.min())
        ax5 = fig.add_subplot(grids[1, 2])
        ax5.imshow(X_viz, aspect="auto", cmap=cmap, interpolation=interp,
                   vmin=0, vmax=max_x)
        ax5.set_axis_off()
        plt.close(fig)
        return fig

    def visualize_adata_factors_sorted(self, adata,
                                 U_factor_id='cell_embedding',
                                 V_factor_id='gene_embedding',
                                 S_matrix_id="S_matrix",
                                 prefix = None,
                                 cmap='viridis', interp='nearest', max_u=1, max_v=1, max_x=1):
        """
        This function generates a visual representation of the NMTF factors, allowing users to specify
        the colormap and interpolation method used for image display.

        :param cmap: Colormap for the visualization. Default is 'viridis'.
        :type cmap: str, optional

        :param interp: Interpolation method for image display. Default is 'nearest'.
        :type interp: str, optional

        :param max_u: The maximum for color scale. Value between [0, 1] where 1 represents the max value in U. Default is 1.
        :type max_u: float, optional

        :param max_v: The maximum for color scale. Value between [0, 1] where 1 represents the max value in V. Default is 1.
        :type max_v: float, optional

        :param max_x: The maximum for color scale. Value between [0, 1] where 1 represents the max value in X. Default is 1.
        :type max_x: float, optional

        :return: U, S, V  matrix heatmaps with X and product.
        :rtype: matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=(16, 6))
        grids = gridspec.GridSpec(2, 3, wspace=0.1, width_ratios=(0.2, 0.4, 0.4), height_ratios=(0.3, 0.7))

        if prefix is not None and prefix[-1] != '_':
            prefix = prefix + '_'

        if prefix is not None:
            U_factor_id = prefix + U_factor_id
            V_factor_id = prefix + V_factor_id
            S_matrix_id = prefix + S_matrix_id

        U = torch.tensor(adata.obsm[U_factor_id])
        V = torch.tensor(adata.varm[V_factor_id]).t()
        S = torch.tensor(adata.uns[S_matrix_id])

        # Generate Sorting for U
        max_U, max_U_idx = U.max(dim=1)
        sorting_criteria = torch.stack([max_U_idx, max_U], dim=1)
        sorted_U_indices = torch.argsort(sorting_criteria, dim=0, stable=True)[:, 0]

        # Generate Sorting for V
        max_V, max_V_idx = V.max(dim=0)
        sorting_criteria = torch.stack([max_V_idx, max_V], dim=1)
        sorted_V_indices = torch.argsort(sorting_criteria, dim=0, stable=True)[:, 0]

        U_viz = U[sorted_U_indices, :].detach().numpy()
        U_viz = (U_viz - U_viz.min()) / (U_viz.max() - U_viz.min())
        ax1 = fig.add_subplot(grids[1, 0])
        ax1.imshow(U_viz, aspect="auto", cmap=cmap, interpolation=interp,
                   vmin=0, vmax=max_u) # set color scale
        ax1.set_axis_off()
        # ax1.set_title("U Matrix")

        # Visualize S matrix
        ax2 = fig.add_subplot(grids[0, 0])
        ax2.imshow(S.t().detach().numpy(), aspect="auto", cmap=cmap, interpolation=interp)
        ax2.set_axis_off()
        # ax2.set_title("S Matrix")

        # Visualize V matrix
        V_viz = V[:, sorted_V_indices].detach().numpy()
        V_viz = (V_viz - V_viz.min())/(V_viz.max() - V_viz.min())
        ax3 = fig.add_subplot(grids[0, 1])
        ax3.imshow(V_viz, aspect="auto", cmap=cmap, interpolation=interp,
                   vmin=0, vmax=max_v) # set color scale
        ax3.set_axis_off()
        # ax3.set_title("V Matrix")

        # Visualize X matrix
        X_est = U @ S @ V
        X_est = X_est[sorted_U_indices, :]
        X_est = X_est[:, sorted_V_indices]
        X_est = (X_est - X_est.min()) / (X_est.max() - X_est.min())
        ax4 = fig.add_subplot(grids[1, 1])
        ax4.imshow(X_est, aspect="auto", cmap=cmap,
                   interpolation=interp, vmin=0, vmax=max_x) # set color scale
        ax4.set_axis_off()

        # ax4.set_title("X Matrix")
        X_temp = adata.X.toarray()
        X_temp = X_temp[sorted_U_indices, :]
        X_temp = X_temp[:, sorted_V_indices]
        X_temp = (X_temp - X_temp.min()) / (X_temp.max() - X_temp.min())
        ax5 = fig.add_subplot(grids[1, 2])
        ax5.imshow(X_temp, aspect="auto", cmap=cmap, interpolation=interp,
                   vmin=0, vmax=max_x) # set color scale
        ax5.set_axis_off()
        plt.close(fig)
        return fig

    def visualize_adata_clusters(self, adata,
                                 U_factor_id='cell_embedding',
                                 V_factor_id='gene_embedding',
                                 prefix = None,
                           cmap='viridis', interp='nearest', max_x=1):
        """
        Visualizes the factors from the NMTF model.

        This function generates a visualization of the factors resulting from the NMTF model. It supports customizing the
        color scheme, interpolation method, and the scaling of the visualization.

        :param factor_name: The name of the factor to visualize (e.g., 'U', 'V').
        :type factor_name: str

        :param cmap: The colormap to use for the visualization. Default is 'viridis'.
        :type cmap: str, optional

        :param interp: The interpolation method for rendering. Default is 'nearest'.
        :type interp: str, optional

        :param max_val: The maximum value for scaling the color map. Default is 1.
        :type max_val: float, optional

        :return: The matplotlib figure object representing the factor visualization.
        :rtype: matplotlib.figure.Figure
        """

        if prefix is not None and prefix[-1] != '_':
            prefix = prefix + '_'

        if prefix is not None:
            U_factor_id = prefix + U_factor_id
            V_factor_id = prefix + V_factor_id

        U = torch.tensor(adata.obsm[U_factor_id])
        V = torch.tensor(adata.varm[V_factor_id]).t()

        U_assign = torch.argmax(U, dim=1)
        V_assign = torch.argmax(V, dim=0)

        fig = plt.figure(figsize=(8, 6))
        grids = gridspec.GridSpec(2, 2, hspace=0.1, wspace=0.1, width_ratios=(0.05, 0.95), height_ratios=(0.05, 0.95))

        # Setup safe color palette for U
        n_u_clusters = max(U_assign)
        tab_20 = plt.get_cmap('tab20')
        if n_u_clusters > 20:
            raise ValueWarning('Number of U clusters exceeds maximum number of supported by palette (tab20). Repeat '
                               'colors will be used.')
        colors = [tab_20(i % 20) for i in range(n_u_clusters + 1)]
        u_cmap = ListedColormap(colors)

        # Visualize U matrix
        ax1 = fig.add_subplot(grids[1, 0])
        ax1.imshow(U_assign.view(-1, 1).detach().numpy(), norm='linear', aspect="auto", cmap=u_cmap,
                   interpolation=interp)
        ax1.set_axis_off()

        n_v_clusters = max(V_assign)
        if n_v_clusters > 20:
            raise ValueWarning('Number of V clusters exceeds maximum of supported by palette (tab20). Repeat '
                               "colors will be used.")
        colors = [tab_20(i % 20) for i in range(n_v_clusters + 1)]
        v_cmap = ListedColormap(colors)

        # Visualize V matrix
        ax3 = fig.add_subplot(grids[0, 1])
        ax3.imshow(V_assign.view(1, -1).detach().numpy(), norm='linear', aspect="auto", cmap=v_cmap,
                   interpolation=interp)
        ax3.set_axis_off()

        ax4 = fig.add_subplot(grids[1, 1])
        X_viz = adata.X.toarray()
        X_viz = (X_viz - X_viz.min()) / (X_viz.max() - X_viz.min())
        ax4.imshow(X_viz, norm='linear', aspect="auto", cmap=cmap, interpolation=interp,
                   vmin=0, vmax=max_x)
        ax4.set_axis_off()
        plt.close(fig)
        return fig

    def visualize_adata_clusters_sorted(self, adata,
                                 U_factor_id='cell_embedding',
                                 V_factor_id='gene_embedding',
                                 prefix = None,
                                 cmap='viridis', interp='nearest', max_x=1):
        """
            Visualizes the clusters by ordering elements of the matrix based on their cluster assignments.

            The function sorts the elements of the matrix by their cluster order and alternates the color of each
                cluster between grey and black. This approach avoids potential issues with limited color palettes, ensuring
                better visual distinction between clusters.

            :param cmap: The colormap to be used for visualization. Defaults to 'viridis'.
            :type cmap: str, optional
            :param interp: The interpolation method for rendering the image. Defaults to 'nearest'.
            :type interp: str, optional
            :param max_x: The maximum for color scale. Value between [0, 1] where 1 represents the max value in X.  Default is 1.
            :type max_x: int, optional
            :return: Sorted clusters heatmap representation.
            :rtype: matplotlib.figure.Figure
        """

        if prefix is not None and prefix[-1] != '_':
            prefix = prefix + '_'

        if prefix is not None:
            U_factor_id = prefix + U_factor_id
            V_factor_id = prefix + V_factor_id

        U = torch.tensor(adata.obsm[U_factor_id])
        V = torch.tensor(adata.varm[V_factor_id]).t()

        U_assign = torch.argmax(U, dim=1)
        V_assign = torch.argmax(V, dim=0)

        fig = plt.figure(figsize=(8, 6))
        grids = gridspec.GridSpec(2, 2, hspace=0.1, wspace=0.1, width_ratios=(0.05, 0.95), height_ratios=(0.05, 0.95))

        # Generate Sorting for U
        max_U, max_U_idx = U.max(dim=1)
        sorting_criteria = torch.stack([max_U_idx, max_U], dim=1)
        sorted_U_indices = torch.argsort(sorting_criteria, dim=0, stable=True)[:, 0]

        # Generate Sorting for V
        max_V, max_V_idx = V.max(dim=0)
        sorting_criteria = torch.stack([max_V_idx, max_V], dim=1)
        sorted_V_indices = torch.argsort(sorting_criteria, dim=0, stable=True)[:, 0]

        barcode_U = torch.zeros_like(U_assign)
        for i, class_value in enumerate(torch.unique(self.U_assign)):
            barcode_U[self.U_assign == class_value] = 0 if i % 2 == 0 else 1.0

        barcode_V = torch.zeros_like(V_assign)
        for i, class_value in enumerate(torch.unique(self.V_assign)):
            barcode_V[self.V_assign == class_value] = 0 if i % 2 == 0 else 1.0

        ax1 = fig.add_subplot(grids[1, 0])
        ax1.imshow(barcode_U[sorted_U_indices].view(-1, 1).detach().numpy(), aspect="auto", cmap='gray', vmin=0,
                   vmax=2, interpolation=interp)
        ax1.set_axis_off()

        # Visualize V matrix
        ax3 = fig.add_subplot(grids[0, 1])
        ax3.imshow(barcode_V[sorted_V_indices].view(1, -1).detach().numpy(), aspect="auto", cmap='gray',
                   vmin=0, vmax=2, interpolation=interp)
        ax3.set_axis_off()
        # ax3.set_title("V Matrix")

        X_temp = adata.X[sorted_U_indices, :][:, sorted_V_indices].toarray()
        X_temp = (X_temp - X_temp.min()) / (X_temp.max() - X_temp.min())
        ax5 = fig.add_subplot(grids[1, 1])
        ax5.imshow(X_temp, aspect="auto", cmap=cmap, interpolation=interp,
                   vmin=0, vmax=max_x)
        ax5.set_axis_off()
        plt.close(fig)
        return fig

