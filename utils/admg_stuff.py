import numpy as np
import scipy.optimize as sopt
import pandas as pd
from ananke.graphs import ADMG


def least_squares_loss(params, X, Z, var_index):

    n, d = X.shape
    return 0.5 / n * np.linalg.norm(X[:, var_index] - np.dot(X, params[0:d]) - np.dot(Z, params[d:])) ** 2


def bic(data, G, tol=1e-4, max_iters=100):
    """
    Perform RICF given data and a graph.

    :param data: pandas data frame
    :param G: Ananke ADMG
    :return: W1, W2 corresponding to directed edge and bidirected edge coefficients
    """

    X = data.values.copy()
    n, d = X.shape
    S = np.cov(X.T)
    W1 = np.zeros((d, d))
    W2 = S.copy()
    Wii = np.eye(d)

    # just code to ensure zeros appear in the correct places when optimizing
    # i.e. we only have parents for parents and siblings of each variable
    idx_var_map = {i: v for i, v in enumerate(data.columns)}
    var_idx_map = {v: i for i, v in enumerate(data.columns)}
    idxs = set(range(d))
    zero_param_directed = {}
    zero_param_bidirected = {}
    for var_index in range(d):

        # get indices of non parents
        parent_idxs = set(var_idx_map[p] for p in G.parents([idx_var_map[var_index]]))
        zero_param_directed[var_index] = idxs - parent_idxs

        # get indices of non siblings
        sibling_idxs = set(var_idx_map[s] for s in G.siblings([idx_var_map[var_index]]))
        zero_param_bidirected[var_index] = idxs - sibling_idxs

    # keep going until desired convergence
    for iter in range(max_iters):

        W1_old = W1.copy()
        W2_old = W2.copy()
        Wii_old = Wii.copy()

        # iterate over each vertex
        for var_index in range(d):

            # get omega and calculate pseudo variables Zi
            omega = W2 + Wii
            omega_minusi = np.delete(omega, var_index, axis=0)
            omega_minusii = np.delete(omega_minusi, var_index, axis=1)
            omega_minusii_inv = np.linalg.inv(omega_minusii)

            # get epsilon_minusi
            epsilon = X - np.matmul(X, W1)
            epsilon_minusi = np.delete(epsilon, var_index, axis=1)

            # calculate Z_minusi
            Z_minusi = (omega_minusii_inv @ epsilon_minusi.T).T

            # insert a column of zeros to maintain the shape
            Z = np.insert(Z_minusi, var_index, 0, axis=1)

            # set bounds on possible values, get values of the current estimates and solve
            bounds = [(0, 0) if i in zero_param_directed[var_index] else (None, None) for i in range(d)]
            bounds += [(0, 0) if i in zero_param_bidirected[var_index] else (None, None) for i in range(d)]
            sol = sopt.minimize(least_squares_loss, np.zeros(2*d),
                                args=(X, Z, var_index),
                                method='L-BFGS-B',
                                options={'disp': False}, bounds=bounds)

            # update W1, W2 with the new solution
            W1[:, var_index] = sol.x[0:d]
            W2[var_index, :] = sol.x[d:]
            W2[:, var_index] = sol.x[d:]
            Wii[var_index, var_index] = np.var(X[:, var_index] - np.dot(X, W1[:, var_index]))

        if np.sum(np.abs(W1_old - W1)) + np.sum(np.abs((W2_old + Wii_old) - (W2 + Wii))) < tol:
            break

    sigma = np.linalg.inv(np.eye(d) - W1.T) @ (W2+Wii) @ np.linalg.inv((np.eye(d) - W1.T).T)
    negll_true = (n / 2) * (np.log(np.linalg.det(sigma)) + np.trace(np.dot(np.linalg.inv(sigma), S)))
    bic = 2*negll_true + np.log(n)*(len(G.di_edges) + len(G.bi_edges))
    return bic


def params(data, G, tol=1e-4, max_iters=100):
    """
    Perform RICF given data and a graph.

    :param data: pandas data frame
    :param G: Ananke ADMG
    :return: W1, W2 corresponding to directed edge and bidirected edge coefficients
    """

    X = data.values.copy()
    n, d = X.shape
    S = np.cov(X.T)
    W1 = np.zeros((d, d))
    W2 = S.copy()
    Wii = np.eye(d)

    # just code to ensure zeros appear in the correct places when optimizing
    # i.e. we only have parents for parents and siblings of each variable
    idx_var_map = {i: v for i, v in enumerate(data.columns)}
    var_idx_map = {v: i for i, v in enumerate(data.columns)}
    idxs = set(range(d))
    zero_param_directed = {}
    zero_param_bidirected = {}
    for var_index in range(d):

        # get indices of non parents
        parent_idxs = set(var_idx_map[p] for p in G.parents([idx_var_map[var_index]]))
        zero_param_directed[var_index] = idxs - parent_idxs

        # get indices of non siblings
        sibling_idxs = set(var_idx_map[s] for s in G.siblings([idx_var_map[var_index]]))
        zero_param_bidirected[var_index] = idxs - sibling_idxs

    # keep going until desired convergence
    for iter in range(max_iters):

        W1_old = W1.copy()
        W2_old = W2.copy()
        Wii_old = Wii.copy()

        # iterate over each vertex
        for var_index in range(d):

            # get omega and calculate pseudo variables Zi
            omega = W2 + Wii
            omega_minusi = np.delete(omega, var_index, axis=0)
            omega_minusii = np.delete(omega_minusi, var_index, axis=1)
            omega_minusii_inv = np.linalg.inv(omega_minusii)

            # get epsilon_minusi
            epsilon = X - np.matmul(X, W1)
            epsilon_minusi = np.delete(epsilon, var_index, axis=1)

            # calculate Z_minusi
            Z_minusi = (omega_minusii_inv @ epsilon_minusi.T).T

            # insert a column of zeros to maintain the shape
            Z = np.insert(Z_minusi, var_index, 0, axis=1)

            # set bounds on possible values, get values of the current estimates and solve
            bounds = [(0, 0) if i in zero_param_directed[var_index] else (None, None) for i in range(d)]
            bounds += [(0, 0) if i in zero_param_bidirected[var_index] else (None, None) for i in range(d)]
            sol = sopt.minimize(least_squares_loss, np.zeros(2*d),
                                args=(X, Z, var_index),
                                method='L-BFGS-B',
                                options={'disp': False}, bounds=bounds)

            # update W1, W2 with the new solution
            W1[:, var_index] = sol.x[0:d]
            W2[var_index, :] = sol.x[d:]
            W2[:, var_index] = sol.x[d:]
            Wii[var_index, var_index] = np.var(X[:, var_index] - np.dot(X, W1[:, var_index]))

        if np.sum(np.abs(W1_old - W1)) + np.sum(np.abs((W2_old + Wii_old) - (W2 + Wii))) < tol:
            break

    sigma = np.linalg.inv(np.eye(d) - W1.T) @ (W2+Wii) @ np.linalg.inv((np.eye(d) - W1.T).T)
    negll_true = (n / 2) * (np.log(np.linalg.det(sigma)) + np.trace(np.dot(np.linalg.inv(sigma), S)))
    bic = 2*negll_true + np.log(n)*(len(G.di_edges) + len(G.bi_edges))
    return W1, W2 + Wii

def main():

    np.random.seed(0)
    size = 5000
    dim = 4
    # generate data from A->B->C->D, A<->D, A<->C
    beta = np.array([[0, 1.2, 0, 0],
                     [0, 0, -1.5, 0],
                     [0, 0, 0, 1.0],
                     [0, 0, 0, 0]]).T

    omega = np.array([[1.2, 0, 0.5, 0.6],
                      [0, 1, 0, 0.],
                      [0.5, 0, 1, 0],
                      [0.6, 0., 0, 1]])

    true_sigma = np.linalg.inv(np.eye(dim) - beta) @ omega @ np.linalg.inv((np.eye(dim) - beta).T)
    X = np.random.multivariate_normal([0] * dim, true_sigma, size=size)
    X = X - np.mean(X, axis=0)  # centre the data
    data = pd.DataFrame({'A': X[:, 0], 'B': X[:, 1], 'C': X[:, 2], 'D': X[:, 3]})

    # make Ananke ADMG and call RICF
    G = ADMG(vertices=['A', 'B', 'C', 'D'],
             di_edges=[('A', 'B'), ('B', 'C'), ('C', 'D')],
             bi_edges=[('A', 'D'), ('A', 'C')])
    print(bic(data, G))


if __name__ == '__main__':
    main()