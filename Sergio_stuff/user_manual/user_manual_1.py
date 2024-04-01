# %%
from landlab import NetworkModelGrid
import landlab.plot.graph as graph
from landlab import imshow_grid
from landlab.io import read_esri_ascii
from landlab.grid.create_network import network_grid_from_raster
from landlab.components import BedRockAbrasionCoverEroder as BRACE
from landlab.components import FlowDirectorSteepest
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# %%
x_of_nodes = [1, 1, 2, 3]
y_of_nodes = [3, 1, 2, 2]
links = [(0,2) ,(1,2), (2,3)]

ngrid = NetworkModelGrid((y_of_nodes, x_of_nodes), links)
graph.plot_graph(ngrid, at="node,link", with_id=True)

# %%
# store it as a raster file
dir = Path(__file__).resolve().parent
file = dir / "short_reach.asc"
rastergrid, topography = read_esri_ascii(file)
rastergrid.add_field("topographic__elevation", topography)
# plot spatially the topography
imshow_grid(rastergrid, rastergrid.at_node["topographic__elevation"], cmap='inferno_r')

# %%
ngrid = network_grid_from_raster(rastergrid)
graph.plot_graph(ngrid, at="node,link", with_id=True)

# %%
flow_director = FlowDirectorSteepest(ngrid)
flow_director.run_one_step()

# %%
nety = BRACE(ngrid, flow_director,
             discharge=600,
             mean_alluvium_thickness=1)

# %%
YEAR = 365.25 * 24 * 60 * 60

# x values are distance downstream (going from left to right)
xs = np.zeros_like(nety.downstream_distance)
xs[1:] = np.cumsum(nety.downstream_distance[:-1])
# y can be any grid landlab fields
ys = ngrid.at_node["bedrock"]
plt.plot(xs, ys)
plt.title("Bedrock")

# %%
# times of the run
dt = YEAR / 1000
one_year = round(1 * YEAR / dt)
total_time = round(0.3 * one_year)
record_time = round(0.02 * one_year)
# plot every record time
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
