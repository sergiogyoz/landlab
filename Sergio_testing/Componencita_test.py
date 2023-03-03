#%%
from landlab import RasterModelGrid, imshow_grid
from landlab.components import FlowDirectorSteepest
import landlab.plot.graph as graph
from landlab.grid.create_network import network_grid_from_raster

import matplotlib.pyplot as plt
# import my DumbComponent
from landlab.components import Componentcita as comp
#%%
# landlab grid
rastergrid = RasterModelGrid(shape=(5, 4), xy_spacing=1)
rastergrid.add_field("topographic__elevation", [
    11.0, 3.0, 2.0, 11.0,
    12.0, 3.0, 4.0, 12.0,
    13.0, 5.0, 4.0, 13.0,
    14.0, 5.0, 6.0, 14.0,
    15.0, 7.0, 6.0, 15.0])
imshow_grid(
    rastergrid,
    rastergrid.at_node["topographic__elevation"],
    cmap='inferno_r')

# flow director
flowdirector = FlowDirectorSteepest(rastergrid, "topographic__elevation")
rastergrid.add_ones("reach_length", at="link", units="m")
imshow_grid(
    flowdirector.grid,
    flowdirector.grid.at_node["topographic__elevation"],
    cmap='inferno_r')

flowdirector.run_one_step()
print(rastergrid.fields())
print(flowdirector.grid.fields())


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
network_grid = network_grid_from_raster(rastergrid)
# %%
graph.plot_graph(
    network_grid,
    at="node,link",
    with_id=True
)

# %%
