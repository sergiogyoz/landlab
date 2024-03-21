# %%
from landlab import NetworkModelGrid
import landlab.plot.graph as graph
from landlab import imshow_grid
from landlab.io import read_esri_ascii
from landlab.grid.create_network import network_grid_from_raster
from landlab.components import Componentcita as comp
from landlab.components import FlowDirectorSteepest
import matplotlib.pyplot as plt
import numpy as np

x_of_nodes = [1, 1, 2, 3]
y_of_nodes = [3, 1, 2, 2]
links = [(0, 2) , (1, 2), (2, 3)]

ngrid = NetworkModelGrid((y_of_nodes, x_of_nodes), links)
graph.plot_graph(ngrid, at="node,link", with_id=True)

# %%
# store it as a raster file
rastergrid, topography = read_esri_ascii("short_reach.asc")
rastergrid.add_field("topographic__elevation", topography)
# plot spatially the topography
imshow_grid(rastergrid, rastergrid.at_node["topographic__elevation"], cmap='inferno_r')

# %%
ngrid = network_grid_from_raster(rastergrid)
graph.plot_graph(ngrid, at="node,link", with_id=True)

# %%
comp.Componentcita._preset_fields(ngrid=ngrid)
flow_director = FlowDirectorSteepest(ngrid)
flow_director.run_one_step()

# %%
nety = comp.Componentcita(ngrid, flow_director)

# %%
YEAR = 365.25 * 24 * 60 * 60

# x values are distance downstream (going from left to right)
xs = np.zeros_like(nety._downstream_distance)
xs[1:] = np.cumsum(nety._downstream_distance[:-1])
# y can be any grid landlab fields
ys = ngrid.at_node["topographic__elevation"]
plt.plot(xs, ys, label="initial")
plt.title("mean alluvium thickness")

# %%
# setup
dt = YEAR / 1000
one_year = round(1 * YEAR / dt)
total_time = round(0.5 * one_year)
record_time = round(0.1 * one_year)
# plot every
for t in range(total_time + 1):
    if t % record_time == 0:
        # plot the new alluvium
        print(t / one_year)
        ys = ngrid.at_node["mean_alluvium_thickness"]
        plt.plot(xs, ys, label=f"{t / one_year :.2f} year")
    nety.run_one_step(dt=dt, q_in=0.000834)
plt.legend()
plt.title("mean alluvium cover")

# %%
