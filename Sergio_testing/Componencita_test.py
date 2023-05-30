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

# grid setup

class setups:

    def __init__(self, shape, steepness=0.1):
        self.n = shape[0]
        self.m = shape[1]
        self.steepness = steepness
        self.grid = [0] * (self.n * self.m) 

    def line(self, horizontal=False):
        if not horizontal:
            self.grid = [0] * (3 * self.n)
            max = self.steepness * self.n * 1.2
            for i in range(self.n):
                self.grid[3 * i] = max
                self.grid[3 * i + 1] = self.steepness * (self.n - i)
                self.grid[3 * i + 2] = max
        return self.grid

    @staticmethod
    def custom(id=1):
        match id:
            case 1:
                return [20.0, 20.0, 0.0, 20.0, 20.0, 20.0,
                        20.0, 11.0, 3.0, 2.0, 11.0, 20.0,
                        20.0, 12.0, 3.0, 4.0, 12.0, 20.0,
                        20.0, 13.0, 5.0, 4.0, 13.0, 20.0,
                        20.0, 14.0, 5.0, 6.0, 14.0, 20.0,
                        20.0, 15.0, 7.0, 6.0, 15.0, 20.0,
                        20.0, 20.0, 20.0, 20.0, 20.0, 20.0]


#%%
shape = (7, 3)
s = setups(shape)
# landlab grid
rastergrid = RasterModelGrid(shape=shape, xy_spacing=1)
rastergrid.add_field("topographic__elevation", s.line())
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
    at="link",
    with_id=True
)

#%%
discharge = 10  # m^3/s
nodes1 = np.ones(network_grid.at_node.size)
links1 = np.ones(network_grid.at_link.size)
network_grid.add_field("reach_length", 1000*links1, at="link", clobber=True)  # 1km
network_grid.add_field("flood_discharge", discharge*10*links1, at="link", clobber=True)  # floods are ten times as big
network_grid.add_field("flood_intermittency", 0.05*links1, at="link", clobber=True)  # flood intermittency 5%
network_grid.add_field("channel_width", 3*links1, at="link", clobber=True)  # channel_width 3m
network_grid.add_field("sediment_grain_size", 2*(10**-3)*links1,at="link", clobber=True)  # sediment sizes 1mm
network_grid.add_field("macroroughness", 1*links1, at="link", clobber=True)  # 1m macroroughness
flow_director = FlowDirectorSteepest(network_grid)
flow_director.run_one_step()
nety = comp.Componentcita(network_grid, flow_director, clobber=True)
network_grid["link"].keys()
network_grid["node"].keys()

# %%
nety.run_one_step(dt = 1)

# %%
graph.plot_graph(
    network_grid,
    at="link,node",
    with_id=False
)
# %%
