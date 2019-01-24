import numpy as np
import matplotlib.pyplot as plt
from polynomial_gossip import *

# GENERATION OF FIGURE 6

# Parameters

l = 40
n = l**2

# Run the simulation 

W = build_2D_graph(n)

methods = ["simple","shift-register",("message-passing",4),"local averaging"]

n_graphs = 1
T = 500

results_averaged = {}
for method in methods:
    results_averaged[method] = np.zeros(T)
    
for graph in range(n_graphs):
    initial_values = np.random.randn(n)
    results = do_the_gossip(methods,T,W,initial_values)
    for method in methods:
        results_averaged[method] += results[method]

for method in methods:
    results_averaged[method] /= n_graphs
    
# Create the figure and save it 

fig = plot_curves(results_averaged, T, 
                  legend=True, logscale=False, 
                  xmin=0, xmax=300, ymin=0, ymax=1.05, 
                  figsize=(6,5))
fig.tight_layout()
fig.savefig("grid_curve_long_mp.eps", format='eps')

fig = plot_curves(results_averaged, T, 
                  legend=False, logscale=True, 
                  xmin=0, xmax=300, ymin=10**(-6), ymax=1.05, 
                  figsize=(6,5))
fig.tight_layout()
fig.savefig("grid_curve_log_mp.eps", format='eps')


# GENERATION OF FIGURE 7 

# Parameters 

n = 2000
d=3

methods = ["simple","shift-register","local averaging",("message-passing",d)]

n_graphs = 1
T = 500

# Run the simulation 

results_averaged = {}
for method in methods:
    results_averaged[method] = np.zeros(T)
    
for graph in range(n_graphs):
    A = adjacency_random_regular_graph(n,d)
    W = csr_matrix(A/d)
    initial_values = np.random.randn(n)
    results = do_the_gossip(methods,T,W,initial_values)
    for method in methods:
        results_averaged[method] += results[method]

for method in methods:
    results_averaged[method] /= n_graphs
    
# Create and save figure 

fig = plot_curves(results_averaged, T, 
                  legend=True, logscale=False, 
                  xmin=0, xmax=20, ymin=0, ymax=1.05, 
                  figsize=(6,5))

fig.tight_layout()
fig.savefig("rrg_curve_mp.eps", format='eps')

fig = plot_curves(results_averaged, T, 
                  legend=False, logscale=True, 
                  xmin=0, xmax=20, ymin=10**(-3), ymax=1.05, 
                  figsize=(6,5))

fig.tight_layout()
fig.savefig("rrg_curve_log_mp.eps", format='eps')