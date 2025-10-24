import networkx as nx
import numpy as np
import torch 
import math 
from scipy.sparse import random, identity 
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def make_contractive(weights):
    s = np.linalg.svd(weights, compute_uv=False)
    scale=1.1
    if s[0] >= 1.0:
        scale = 1.1 * s[0]
    
    return weights/scale

def make_non_cotractive(weights):
    s = np.linalg.svd(weights, compute_uv=False)
    scale = 1.0
    # if s[0] <= 1.0:
    scale = 1.5/s[0]
    
    return scale * weights 

def generate_sparse_positive_def_cov_matrix(size, off_diag_nonzeros=None, seed=None):

    if seed is not None:
        np.random.seed(seed)

    max_pairs = size * (size - 1) // 2
    if off_diag_nonzeros > max_pairs:
        raise ValueError(
            f"Requested {off_diag_nonzeros} off-diagonal non-zeros but only {max_pairs} are possible"
        )

    cov_matrix = np.zeros((size, size), dtype=float)

    if off_diag_nonzeros > 0:
        upper_idx = np.triu_indices(size, k=1)
        chosen = np.random.choice(len(upper_idx[0]), size=off_diag_nonzeros, replace=False)
        values = np.random.uniform(0.05, 1.0, size=off_diag_nonzeros)
        signs = np.random.choice([-1.0, 1.0], size=off_diag_nonzeros)
        off_diag_values = values * signs

        rows = upper_idx[0][chosen]
        cols = upper_idx[1][chosen]
        cov_matrix[rows, cols] = off_diag_values
        cov_matrix[cols, rows] = off_diag_values

    row_abs_sum = np.sum(np.abs(cov_matrix), axis=1)
    eps = 1e-1
    np.fill_diagonal(cov_matrix, row_abs_sum + eps)

    return cov_matrix

class SEM:

    """
    -------------------------------------------------------------------
    This class models a Linear Structural Equation Model (Linear SEM)
    -------------------------------------------------------------------
    The model is initialized with the number of nodes in the graph and
    the absolute minimum and maximum weights for the edges. 
    """
    def __init__(
            self, 
            graph, 
            abs_weight_low=0.2, 
            abs_weight_high=0.9, 
            noise_scale=0.5, 
            contractive=True, 
            beta=1.0, 
            confounders=False, 
            off_diag_nonzeros=3,
            set_bias=True,
            seed=None
        ):
        self.graph = graph
        self.abs_weight_low = abs_weight_low 
        self.abs_weight_high = abs_weight_high
        self.contractive = contractive
        self.off_diag_nonzeros = off_diag_nonzeros

        self.n_nodes = len(graph.nodes)
        
        self.weights = np.random.uniform(self.abs_weight_low, self.abs_weight_high, size=(self.n_nodes, self.n_nodes))
        self.weights *= 2 * np.random.binomial(1, 0.5, size=self.weights.shape) - 1
        self.weights *= nx.to_numpy_array(self.graph)
        self.bias = 1.5 * np.random.randn(self.n_nodes) if set_bias else np.zeros(self.n_nodes)

        self.noise_scale = noise_scale
        self.beta = beta # linear-nonlinear mixing factor 
        self.confounders = confounders

        if self.confounders: 
            self.confounder_sigma = generate_sparse_positive_def_cov_matrix(
                size=self.n_nodes,
                off_diag_nonzeros=self.off_diag_nonzeros,
                seed=seed
            )
            self.confounder_sigma = self.noise_scale**2 * self.confounder_sigma/self.confounder_sigma.max()
            self.confounder_inverse_cov = np.linalg.inv(self.confounder_sigma)

        if not self.contractive:
            self.weights = make_non_cotractive(self.weights)
        else:
            self.weights = make_contractive(self.weights)

    def generateData(self, n_samples, intervention_set=[None], lat_provided=False, latent_vec=None, fixed_intervention=False, return_latents=False, n_iter=30, beta_given=False, beta=1.0):
        # set intervention_set = [None] for purely observational data.
        
        observed_set = np.setdiff1d(np.arange(self.n_nodes), intervention_set)
        U = np.zeros((self.n_nodes, self.n_nodes))
        U[observed_set, observed_set] = 1

        C = np.zeros((self.n_nodes, n_samples))
        if intervention_set[0] != None:
            if fixed_intervention:
                C[intervention_set, :] = np.random.randn(len(intervention_set), 1)
            else:
                C[intervention_set, :] = 2.0 * np.random.randn(len(intervention_set), n_samples)

        I = np.eye(self.n_nodes)
        if lat_provided:
            E = latent_vec.T
        else:
            E = self.noise_scale * np.random.randn(self.n_nodes, n_samples)
        
        if self.confounders: 
            N = np.random.multivariate_normal(mean=np.zeros(self.n_nodes), cov=self.confounder_sigma, size=n_samples).T
            E = N
        
        wtx = lambda x: self.weights.T @ x
        lin_func = lambda x: wtx(x) + E

        beta_ = beta if beta_given else self.beta

        X = np.random.randn(self.n_nodes, n_samples)
        for _ in range(n_iter):
            X = U @ ( (1 - beta_) * lin_func(X) + beta_ * np.tanh(lin_func(X)) ) + C

        # The final data matrix has dimensions - n_samples X self.nodes
        if return_latents:
            return X.T, E.T
            
        return X.T