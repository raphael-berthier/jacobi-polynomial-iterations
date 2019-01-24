import numpy as np
import matplotlib.pyplot as plt
from polynomial_gossip import *

# 2D RANDOM GEOMETRIC GRAPH

# Parameters

n = 1600
d = 2
r = 1.5/n**(1/d) 

methods = [("jacobi",2),"simple","shift-register","local averaging"]

n_graphs = 10
T = 400

# Run the simulation

results_averaged = {}
for method in methods:
    results_averaged[method] = np.zeros(T)
    
for graph in range(n_graphs):
    A, positions = adjacency_and_positions_rdm_geom_graph(n,d,r)
    W = gossip_matrix_from_adjacency_matrix(A)
    W = largest_component(W)
    n_nodes = W.shape[0]
    initial_values = np.random.randn(n_nodes)
    results = do_the_gossip(methods,T,W,initial_values)
    for method in methods:
        results_averaged[method] += results[method]

for method in methods:
    results_averaged[method] /= n_graphs
    
# Create and save Figure 2E

fig = plot_curves(results_averaged, T, 
                  legend=False, logscale=False, 
                  xmin=0, xmax=200, ymin=0, ymax=1.05, 
                  figsize=(6,5))

fig.tight_layout()
fig.savefig("rgg_curve.eps", format='eps')

# 3D RANDOM GEOMETRIC GRAPH

# Parameters

n = 1728
d = 3
r = 1.5/n**(1/d)

methods = [("jacobi",3),"simple","shift-register","local averaging"]

n_graphs = 10
T = 40

# Run the simulation 

results_averaged = {}
for method in methods:
    results_averaged[method] = np.zeros(T)
    
for graph in range(n_graphs):
    A, positions = adjacency_and_positions_rdm_geom_graph(n,d,r)
    W = gossip_matrix_from_adjacency_matrix(A)
    W = largest_component(W)
    n_nodes = W.shape[0]
    initial_values = np.random.randn(n_nodes)
    results = do_the_gossip(methods,T,W,initial_values)
    for method in methods:
        results_averaged[method] += results[method]

for method in methods:
    results_averaged[method] /= n_graphs
    
# Create and save Figure 2F

fig = plot_curves(results_averaged, T, 
                  legend=False, logscale=False, 
                  xmin=0, xmax=T-1, ymin=0, ymax=1.05, 
                  figsize=(6,5))

fig.tight_layout()
fig.savefig("3D_rgg_curve.eps", format='eps')