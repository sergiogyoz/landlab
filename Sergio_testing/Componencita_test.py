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
flowdirector = FlowDirectorD8(rastergrid)
flowdirector.run_one_step()
# should I use the network to grid to create the simplified network? mmm
network_grid = network_grid_from_raster(rastergrid)
"""graph.plot_graph(
    network_grid,
    at="node,link",
    with_id=True
)"""

print(rastergrid.fields())
# rastergrid.add_field("upstream_node", values = rastergrid.at_node["flow__receiver_node"], at="link")

# rastergrid.at_node["flow__receiver_node"]


#%%
# Here I create an instance of my component
mydummy = comp.Componentcita(rastergrid, flowdirector, sm=0.5)
mydummy.grid.fields()
# plot the component topographic elevation field
plt.figure()
imshow_grid(
    mydummy.grid,
    mydummy.grid.at_node["topographic__elevation"],
    cmap='inferno_r')
#%%
mydummy._update_channel_slopes()
plt.figure()
imshow_grid(
    mydummy.grid,
    mydummy.grid.at_node["topographic__elevation"],
    cmap='inferno_r')
mydummy.grid.fields()
# %%
mydummy.grid.fields()
# %%
mydummy.grid.at_link["channel_slope"]
# %%
mydummy.grid.at_node["topographic__steepest_slope"]
# %%

# %%

down_links = copy.copy(rastergrid.at_node["flow__link_to_receiver_node"])
active_links_index = (down_links != -1)
nodes = np.array(range(rastergrid.number_of_nodes))
upstream_nodes = copy.copy(nodes[active_links_index])
downstream_nodes = copy.copy(rastergrid.at_node["flow__receiver_node"][active_links_index])
# %%
