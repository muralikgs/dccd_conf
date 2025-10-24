import numpy as np 
import networkx as nx 

from itertools import permutations
from sklearn.linear_model import LinearRegression 
from copent import copent 
from tqdm import tqdm

# Residual calculation
def get_residuals(y, X):
    model = LinearRegression().fit(X, y)
    return y - model.predict(X)

# Mutual information estimation from copula entropy
def compute_mi(x, y):
    data = np.vstack([x, y]).T
    return -copent(data)  # MI = - copula entropy

# Total MI for a given ordering
def total_mi_for_ordering(data, ordering):
    residuals = []
    for i, var in enumerate(ordering):
        if i == 0:
            residuals.append(data[:, var])
        else:
            parents = ordering[:i]
            res = get_residuals(data[:, var], data[:, parents])
            residuals.append(res)

    total = 0
    for j in range(len(residuals) - 1):
        for k in range(j+1, len(residuals)):
            total += compute_mi(residuals[j], residuals[k])
    return total

# Find best ordering
def find_best_ordering(data):
    p = data.shape[1]
    best_order, min_mi = None, np.inf
    perms = list(permutations(range(p)))
    
    for perm in tqdm(perms, desc="Searching orderings"):
        mi = total_mi_for_ordering(data, perm)
        if mi < min_mi:
            min_mi, best_order = mi, perm
    return best_order[::-1], min_mi

def estimate_graph(data, ordering, threshold=1e-1):
    p = data.shape[1]
    G = nx.DiGraph()
    G.add_nodes_from(ordering)

    for j in range(1, p):
        y = data[:, ordering[j]]
        X = data[:, ordering[:j]]
        model = LinearRegression().fit(X, y)
        coefs = model.coef_
        for i, coef in enumerate(coefs):
            if abs(coef) > threshold:
                G.add_edge(ordering[i], ordering[j])

    return G

class lingamMMI:
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes 

    def fit(self, X):
        best_order, _ = find_best_ordering(X)
        G = estimate_graph(X, best_order)
        return nx.to_numpy_array(G)