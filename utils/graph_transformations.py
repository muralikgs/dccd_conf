import numpy as np 
import networkx as nx 

from itertools import compress
from typing import Tuple

from models.nodags.resblock import iResBlock

def convert_model_to_dmg(
        model: iResBlock, 
        w_threshold:float=0.95, 
        cov_threshold:float=1e-2
    ) -> Tuple[np.ndarray, np.ndarray]:

    # extract the adjacency corresponding to directional edges
    di_edge_mat = (model.get_w_adj() > w_threshold)*1.0

    # extract the adjacency corresponding to undirected edges
    cov_est = model.cov_mat.detach().cpu().numpy()
    
    diagonal_mask = np.ones_like(cov_est)
    np.fill_diagonal(diagonal_mask, 0)

    bi_di_edge_mat = (np.abs(cov_est * diagonal_mask) > cov_threshold)*1.0

    return di_edge_mat, bi_di_edge_mat

# acyclification procedure to convert a DMG G to acy(G)
def convert_dmg_to_admg(
        dmg: Tuple[np.ndarray, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:

    E, B = dmg

    # Create a directed graph from the adjacency matrix E
    G = nx.DiGraph(E)

    # Compute the strongly connected components
    scc = list(nx.strongly_connected_components(G))
 
    # Create a mapping of each node to its strongly connected component
    node_to_scc = {node: idx for idx, component in enumerate(scc) for node in component}

    E_acy = np.zeros_like(E)
    B_acy = np.zeros_like(B)
    for parent in range(len(E)):
        for child in range(len(E)):
            # get the parents of the SCC of the child
            par_scc = set(pred for node in scc[node_to_scc[child]] for pred in G.predecessors(node))
            # add the edge if the parent is in the parents of SCC
            # but not in the SCC of the child
            if parent in par_scc.difference(set(scc[node_to_scc[child]])):
                E_acy[parent, child] = 1

            # add a bidirectional edge if the SCC of the parent and child have a common node
            child_scc = scc[node_to_scc[child]]
            parent_scc = scc[node_to_scc[parent]]
            if len(set(child_scc).intersection(set(parent_scc))) > 0 and parent != child:
                B_acy[parent, child] = 1
                B_acy[child, parent] = 1
            
            # add a bidirectional edge if the a node from parent SCC 
            # and a node from child SCC have a bidirectional edge
            bidirectional_edges = [B[node1, node2] for node1 in child_scc for node2 in parent_scc]
            if sum(bidirectional_edges) > 0 and parent != child:
                B_acy[parent, child] = 1
                B_acy[child, parent] = 1

    return E_acy, B_acy

# function to generate an augmented DMG given a DMG and a family of interventions
def generate_augmented_dmg(
        dmg: Tuple[np.ndarray, np.ndarray],
        intervention_sets: list[list[int]],
    ) -> Tuple[np.ndarray, np.ndarray]:

    E, B = dmg
    d = E.shape[0]
    k = len(intervention_sets)

    E_aug = np.zeros((d+k, d+k))
    B_aug = np.zeros_like(E_aug)

    E_aug[:d, :d] = E
    B_aug[:d, :d] = B

    for i, targets in enumerate(intervention_sets):
        for node in targets:
            E_aug[i+d, node] = 1

    return E_aug, B_aug

# function to check if a node is a collider on a path
def is_collider(G: nx.DiGraph, path: list, node: int) -> bool:
    """
    Check if node at position `idx` in the path is a collider.
    """
    idx = path.index(node)
    if idx <= 0 or idx >= len(path) - 1:
        return False  # Can't be a collider at endpoints

    prev_node = path[idx - 1]
    curr_node = path[idx]
    next_node = path[idx + 1]

    # Check if both prev and next nodes have edges INTO the current node
    return G.has_edge(prev_node, curr_node) and G.has_edge(next_node, curr_node)

# function to check if a path is an inducing path
def is_inducing_path(path: list[int], L: list[int], G: nx.DiGraph) -> bool:

    nodes_outside_L = list(set(path[1:-1]).difference(set(L)))
    
    # All single edge paths are inducing paths
    if len(nodes_outside_L) == 0:
        return True 
    
    else:
        # Every node not in L (except for endpoints) is a collider?
        colliders_mask = list(map(lambda node: is_collider(G, path, node), nodes_outside_L))

        # condition 1
        if all(colliders_mask):
            # Ever collider is an ancester of an endpoint
            endpoint_checker = lambda node: nx.has_path(G, node, path[0]) or nx.has_path(G, node, path[-1])
            condition_2 = list(map(endpoint_checker, nodes_outside_L))

            if all(condition_2):
                return True
            
    return False

# function to convert ADMG to MAG
def convert_admg_to_mag(
        dmg: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
    
    E, B = dmg
    d = E.shape[0]
    n_confounders = B.sum()//2

    # Generate the outer projection matrix
    E_out = np.zeros((d + int(n_confounders), d + int(n_confounders)))
    E_out[:d, :d] = E

    conf_counter = 0
    for node_i in range(d):
        for node_j in range(node_i+1, d):
            if B[node_i, node_j] == 1:
                E_out[d + conf_counter, node_i] = 1
                E_out[d + conf_counter, node_j] = 1
                conf_counter += 1

    G = nx.DiGraph(E_out)
    G_undirected = G.to_undirected()
    L = [d+i for i in range(int(n_confounders))]

    # Rule one to construct the skeleton of the MAG
    undirected_edges = list()
    for node_i in range(d):
        for node_j in range(d):
            if node_i != node_j:
                all_paths = list(nx.all_simple_paths(G_undirected, node_i, node_j))
                inducing_path_mask = list(map(
                    lambda path: is_inducing_path(path, L, G), all_paths
                ))
                if any(inducing_path_mask):
                    undirected_edges.append([node_i, node_j])
    
    E_mag, B_mag = np.zeros_like(E), np.zeros_like(B)
    
    # Rule two to orient the edges
    for node_i, node_j in undirected_edges:
        if nx.has_path(G, node_i, node_j):
            E_mag[node_i, node_j] = 1
        elif nx.has_path(G, node_j, node_i):
            E_mag[node_j, node_i] = 1
        else:
            B_mag[node_i, node_j] = 1
            B_mag[node_j, node_i] = 1

    return E_mag, B_mag

def is_unshielded_collider(G: nx.DiGraph, node: int) -> bool:
    """
    Check if a specific node in a directed graph is an unshielded collider.

    A node is an unshielded collider if:
    1. It has two or more distinct parents (nodes with edges pointing to it).
    2. There is no edge (in either direction) between any pair of its parents.

    Parameters:
    - G: A directed graph (DiGraph)
    - node: The node to check

    Returns:
    - True if the node is an unshielded collider, False otherwise
    """
    # Get the predecessors (parents) of the node
    parents = list(G.predecessors(node))
    # A node is a collider if it has at least two distinct parents
    if len(parents) < 2:
        return False

    # Check if the collider is unshielded
    for i in range(len(parents)):
        for j in range(i + 1, len(parents)):
            parent1, parent2 = parents[i], parents[j]
            # If there is an edge between any pair of parents, it's not unshielded
            if G.has_edge(parent1, parent2) or G.has_edge(parent2, parent1):
                return False

    return True

def find_unshielded_colliders(G: nx.DiGraph) -> list[int]:
    """
    Identify all unshielded colliders in a directed graph.

    A node is an unshielded collider if:
    1. It has two or more distinct parents (nodes with edges pointing to it).
    2. There is no edge (in either direction) between any pair of its parents.

    Parameters:
    - G: A directed graph (DiGraph)

    Returns:
    - unshielded_colliders: A list of nodes that are unshielded colliders
    """
    unshielded_colliders = []
    for node in G.nodes:
        if is_unshielded_collider(G, node):
            unshielded_colliders.append(node)

    return unshielded_colliders

def is_discriminating_path(G: nx.DiGraph, path: list[int]) -> bool: 

    # For illustration let the path be X-...-W-Z-Y

    skeleton = nx.Graph(G)

    # path has at least three edges
    if not len(path) > 3: 
        return False
    
    # X should not be adjacent to Y
    if G.has_edge(path[0], path[-1]) or G.has_edge(path[-1], path[0]):
        return False

    # every except X, Y, and Z should be a collider and a parent of Y
    nonendpoint_nodes = path[1:-2]
    is_parent = lambda node: G.has_edge(node, path[-1]) and not G.has_edge(path[-1], node)
    check_condition_3 = lambda node: is_collider(G, path, node) and is_parent(node)

    condition_3_mask = map(check_condition_3, nonendpoint_nodes)

    if not all(condition_3_mask):
        return False

    return True

def find_discriminating_paths(G: nx.DiGraph) -> list[list[int]]:
    
    discriminating_paths = list()
    for source in G.nodes:
        for target in G.nodes:
            if source != target:
                for path in nx.all_simple_paths(nx.Graph(G), source, target):
                    if is_discriminating_path(G, path):
                        discriminating_paths.append(path)

    return discriminating_paths

# function to check equivalence of two MAGs M_1, and M_2
def mags_equivalence_check(mag_1: Tuple[np.ndarray, np.ndarray], mag_2: Tuple[np.ndarray, np.ndarray]) -> Tuple[int, tuple[int]]:
    # Here mag_1 is taken to be the ground truth
    
    shd = 0

    # convert the MAGs to networkx DiGraphs
    m_1 = nx.DiGraph(mag_1[0] + mag_1[1])
    m_2 = nx.DiGraph(mag_2[0] + mag_2[1])

    # 1. check for condition one
    # Both the mags should have the same skeleton
    skeleton1 = nx.Graph(m_1)
    skeleton2 = nx.Graph(m_2)

    edges_1 = set(skeleton1.edges())
    edges_2 = set(skeleton2.edges())
    num_extra_edges = len(edges_1.symmetric_difference(edges_2))
    
    # Missing edges + extra edges
    shd += num_extra_edges

    # 2. check for condition two
    # Both the mags should have the same unshielded colliders
    m1_unshielded_colliders = set(find_unshielded_colliders(m_1))
    m2_unshielded_colliders = set(find_unshielded_colliders(m_2))
    num_extra_unshielded_colliders = len(m1_unshielded_colliders.symmetric_difference(m2_unshielded_colliders))

    shd += num_extra_unshielded_colliders

    # 3. check for condition three
    # If p is a discriminating path for a node Y in both m1 and m2 then it should either be a collider in 
    # both or a non-collider in both
    m1_discriminating_paths = find_discriminating_paths(m_1)
    m2_discriminating_paths = find_discriminating_paths(m_2)
    # Extract common discriminating paths between m1 and m2
    common_discriminating_paths = [
        path for path in m1_discriminating_paths if path in m2_discriminating_paths
    ]

    discriminating_path_mismatch = 0
    for path in common_discriminating_paths:
        Z = path[-2]
        is_Z_collider_m1 = is_collider(m_1, path, Z)
        is_Z_collider_m2 = is_collider(m_2, path, Z)

        # A XOR B
        if is_Z_collider_m1 ^ is_Z_collider_m2:
            discriminating_path_mismatch += 1
    
    shd += discriminating_path_mismatch

    return shd, (num_extra_edges, num_extra_unshielded_colliders, discriminating_path_mismatch)

def compute_equivalance_score_mag_pag(mag: Tuple[np.ndarray, np.ndarray], pag: np.ndarray) -> Tuple[int, tuple[int]]:

    shd = 0
    mag_graph = nx.DiGraph(mag[0] + mag[1])
    
    # check if the skeletons match
    skeleton_mag = nx.Graph(mag_graph)
    skeleton_pag = nx.Graph(np.abs(pag) > 0)

    edges_1 = set(skeleton_mag.edges())
    edges_2 = set(skeleton_pag.edges())
    num_extra_edges = len(edges_1.symmetric_difference(edges_2))

    shd += num_extra_edges

    num_edge_reversals = 0
    # check if the correct edge marks are there: 
    for edge in edges_1.intersection(edges_2):
        # Let the edge be A - B
        a = edge[0]
        b = edge[1]

        # case 1: A -> B
        if pag[a,b] == -1 and pag[b,a] == 1:
            if not mag[0][a,b] == 1:
                num_edge_reversals += 1
        # case 2: A <- B
        if pag[a,b] == 1 and pag[b,a] == -1:
            if not mag[0][b,a] == 1:
                num_edge_reversals += 1
        
        # case 3: A o-> B
        if pag[a,b] == 2 and pag[b,a] == 1:
            if mag[0][a,b] == 1 or mag[1][a,b] == 1:
                pass
            else:
                num_edge_reversals += 1        
        # case 4: A <-o B
        if pag[a,b] == 1 and pag[b,a] == 2:
            if mag[0][b,a] == 1 or mag[1][a,b] == 1:
                pass
            else:
                num_edge_reversals += 1

        # case 5: A <-> B
        if pag[a,b] == 1 and pag[b,a] == 1:
            if not mag[1][a,b] == 1:
                num_edge_reversals += 1

    shd += num_edge_reversals

    # mismatch in unshielded collider
    
    # convert PAG to DiGraph
    # leave directional_edges and bidirectional edges
    dir_edge_mask = (np.abs(pag) == 1).T
    # convert o-o to -> 
    empty_mask = (pag == 2) * (pag == 2).T

    pag_graph = nx.DiGraph(dir_edge_mask*1.0 + empty_mask*1.0)

    m1_unshielded_colliders = set(find_unshielded_colliders(mag_graph))
    m2_unshielded_colliders = set(find_unshielded_colliders(pag_graph))
    num_extra_unshielded_colliders = len(m1_unshielded_colliders.symmetric_difference(m2_unshielded_colliders))

    shd += num_extra_unshielded_colliders 

    return shd, (num_extra_edges, num_edge_reversals, num_extra_unshielded_colliders)  



# function to compute the SHD between two the corresponding MAGs of two ADMGs
def compute_shd_from_est(
        dmg_1: Tuple[np.ndarray, np.ndarray],
        dmg_2: Tuple[np.ndarray, np.ndarray],
        intervention_sets: list[list[int]]
) -> Tuple[int, tuple[int]]:
    
    # convert DMG to augmented DMG
    print("Converting DMG to Aug(DMG)")
    aug_dmg_1 = generate_augmented_dmg(dmg_1, intervention_sets)
    aug_dmg_2 = generate_augmented_dmg(dmg_2, intervention_sets)

    # convert augmented DMG to ADMGs
    print("Converting Aug(DMG) to ADMG")
    admg_1 = convert_dmg_to_admg(aug_dmg_1)
    admg_2 = convert_dmg_to_admg(aug_dmg_2)

    # convert ADMG to MAG
    print("Converting ADMG to MAG")
    mag_1 = convert_admg_to_mag(admg_1)
    mag_2 = convert_admg_to_mag(admg_2)

    # compute SHD
    print("Computing SHD")
    shd, indiv_nums = mags_equivalence_check(mag_1, mag_2)

    return shd, indiv_nums


if __name__ == "__main__":
    E = np.array(
        [[0,1,0,0,0],
         [0,0,1,0,1],
         [0,0,0,0,0],
         [0,1,1,0,0],
         [0,0,0,0,0]]
    )
    B = np.array(
        [[0,0,0,0,0],
         [0,0,1,0,0],
         [0,1,0,0,1],
         [0,0,0,0,0],
         [0,0,1,0,0]]
    )

    E_mag, B_mag = convert_admg_to_mag((E, B))
    print(E_mag)
    print(B_mag)
