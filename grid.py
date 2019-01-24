import numpy as np
import matplotlib.pyplot as plt
from polynomial_gossip import *

# 2D GRID

# Run the simulation

l = 40
n = l**2
W = build_2D_graph(n)

methods = [("jacobi",2),"simple",
           "shift-register","local averaging",
           "best polynomial gossip",("jacobi-gap",2)]

n_graphs = 10
T = 300

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
    
# Generation of Figure 2A
    
fig = plot_curves(results_averaged, T, 
                  legend=True, logscale=False, 
                  xmin=0, xmax=120, ymin=0, ymax=1.05, 
                  figsize=(6,5), methods = [("jacobi",2),"simple",
           "shift-register","local averaging"])

fig.tight_layout()
fig.savefig("grid_curve.eps", format='eps')

# Generation of Figure 3A 

fig = plot_curves(results_averaged, T, 
                  legend=True, logscale=False, 
                  xmin=0, xmax=300, ymin=0, ymax=1.05, 
                  figsize=(6,5), methods = [("jacobi",2),"simple",
           "shift-register","local averaging"])

fig.tight_layout()
fig.savefig("grid_curve_long.eps", format='eps')

# Generation of Figure 3B 

fig = plot_curves(results_averaged, T, 
                  legend=False, logscale=True, 
                  xmin=0, xmax=300, ymin=10**(-6), ymax=1.05, 
                  figsize=(6,5), methods = [("jacobi",2),"simple",
           "shift-register","local averaging"])

fig.tight_layout()
fig.savefig("grid_curve_log.eps", format='eps')

# Generation of Figure 4A 

fig = plot_curves(results_averaged, T, 
                  legend=True, logscale=False, 
                  xmin=0, xmax=300, ymin=0, ymax=1.05, 
                  figsize=(6,5))

fig.tight_layout()
fig.savefig("grid_curve_long_all.eps", format='eps')

# Generation of Figure 4B

fig = plot_curves(results_averaged, T, 
                  legend=False, logscale=True, 
                  xmin=0, xmax=300, ymin=10**(-6), ymax=1.05, 
                  figsize=(6,5))

fig.tight_layout()
fig.savefig("grid_curve_log_all.eps", format='eps')


# 3D GRID

# Run the simulation

l = 12
n = l**3
W = build_3D_graph(n)

methods = [("jacobi",3),"simple","shift-register","local averaging"]

n_graphs = 10
T = 50

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
    
# Generation of Figure 2B

fig = plot_curves(results_averaged, T, legend=False, logscale=False, 
         xmin=0, xmax=40, ymin=0, ymax=1.05, figsize=(6,5))

fig.tight_layout()
fig.savefig("3D_grid_curve.eps", format='eps')

