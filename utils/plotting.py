import networkx as nx

from matplotlib.patches import FancyArrowPatch
from networkx.drawing.nx_pylab import draw_networkx_nodes, draw_networkx_labels

# Custom function to draw curved edges without cutting into nodes
def draw_curved_edges(G, pos, ax, curve_scale=0.2, node_size=300):
    node_radius= (node_size)**0.5 * 0.5
    for u, v in G.edges():
        rad = curve_scale if (u, v) in G.edges() and (v, u) in G.edges() else 0
        arrow = FancyArrowPatch(pos[u], pos[v], connectionstyle=f"arc3,rad={rad}",
                                arrowstyle="->", color="blue", mutation_scale=15, lw=1.5,
                                shrinkA=node_radius, shrinkB=node_radius)
        ax.add_patch(arrow)
