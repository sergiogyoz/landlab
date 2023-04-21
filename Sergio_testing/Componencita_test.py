#%%
import matplotlib.pyplot as plt
import numpy as np
import copy

from landlab import RasterModelGrid, imshow_grid
from landlab import NetworkModelGrid
from landlab.components import FlowDirectorSteepest, FlowDirectorD8
import landlab.plot.graph as graph
from landlab.grid.create_network import network_grid_from_raster

# import my DumbComponent
from landlab.components import Componentcita as comp

#%%
shape = (7,6)
# landlab grid
rastergrid = RasterModelGrid(shape=shape, xy_spacing=1)
rastergrid.add_field("topographic__elevation", [
    20.0, 20.0, 0.0, 20.0, 20.0, 20.0,
    20.0, 11.0, 3.0, 2.0, 11.0, 20.0,
    20.0, 12.0, 3.0, 4.0, 12.0, 20.0,
    20.0, 13.0, 5.0, 4.0, 13.0, 20.0,
    20.0, 14.0, 5.0, 6.0, 14.0, 20.0,
    20.0, 15.0, 7.0, 6.0, 15.0, 20.0,
    20.0, 20.0, 20.0, 20.0, 20.0, 20.0])
imshow_grid(
    rastergrid,
    rastergrid.at_node["topographic__elevation"],
    cmap='inferno_r')

# flow director
"""
flowdirector = FlowDirectorSteepest(rastergrid, "topographic__elevation")
rastergrid.add_ones("reach_length", at="link", units="m")
"""

# should I use the network to grid to create the simplified network? mmm
network_grid = network_grid_from_raster(rastergrid)
print(rastergrid.fields())

#%%
graph.plot_graph(
    network_grid,
    at="node,link",
    with_id=True
)

#%%
flow_director = FlowDirectorSteepest(network_grid)
flow_director.run_one_step()
nety = comp.Componentcita(network_grid, flow_director)
nety._add_upstream__downstream_nodes()
network_grid["link"].keys()
# %%


