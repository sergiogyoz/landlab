# %%
import matplotlib.pyplot as plt
import numpy as np
import copy
import math

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

    def line(self, Vertical=False):
        if not Vertical:
            max = self.steepness * self.m * 1.2 + 0.01
            self.grid = [max] * (3 * self.m)
            for i in range(self.m):
                self.grid[self.m + i] = self.steepness * (self.m - i)
        else:
            max = self.steepness * self.n * 1.2 + 0.01
            self.grid = [max] * (3 * self.n)
            for i in range(self.n):
                self.grid[3 * i + 1] = self.steepness * i
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


# %%
shape = (3, 20)
reach_lenght = 1
slope = 0.004  # 0.004
s = setups(shape, steepness=slope * reach_lenght)
# landlab grid
rastergrid = RasterModelGrid(shape=shape, xy_spacing=1)
rastergrid.add_field("topographic__elevation", s.line())
imshow_grid(
    rastergrid,
    rastergrid.at_node["topographic__elevation"],
    cmap='inferno_r')

# should I use the network to grid to create the simplified network? mmm
ngrid = network_grid_from_raster(rastergrid)
print(rastergrid.fields())

# %%
flow_director = FlowDirectorSteepest(ngrid)
flow_director.run_one_step()
comp.Componentcita._preset_fields(ngrid, all_ones=True)
nety = comp.Componentcita(ngrid, flow_director, clobber=True)

# %%
graph.plot_graph(ngrid, at="node,link", with_id=True)
print(ngrid["link"].keys())
print(ngrid["node"].keys())


# %%
fig1 = plt.figure(1)
fig2 = plt.figure(2)
fig3 = plt.figure(3)

n = len(ngrid.at_node["sed_capacity"])
xs = list(range(n))
fraction = 0.001
year = 365.25
dt = fraction * year
total_time = 0.001 * year  # how long to simulate
record_t = 1 * fraction  # how often to record
sed_source = np.array([0])
i = 0
# downstream distance for plots
distance = np.zeros_like(ngrid.at_node["sed_capacity"])
distance[1:] = np.cumsum(ngrid.at_link["reach_length"])
# sedimentograph initial pulse
qt = 0.04342996
ngrid.at_node["sed_capacity"][sed_source] = qt

# %%
for time in np.arange(0, total_time, dt):
    # print(ngrid.at_node["sed_capacity"])
    if math.isclose(time % record_t, 0, abs_tol=dt / 3):
        plt.figure(fig1);
        bed = ngrid.at_node["bedrock"][xs]
        plt.plot(distance, bed, label=f"t= {time:.3f} and i ={i}");

        plt.figure(fig2);
        alluvium = ngrid.at_node["mean_alluvium_thickness"][xs]
        plt.plot(distance, alluvium, label=f"t= {time:.3f} and i ={i}");

        plt.figure(fig3);
        plt.plot(distance, np.log10(alluvium), label=f"t= {time:.3f} and i ={i}");

    nety.run_one_step(dt=dt)  # ,omit=sed_source)
    i = i + 1

# %%
plt.figure(fig1)
plt.xlabel("downstream distance")
plt.ylabel("bedrock elevation (m)")
plt.legend()
plt.show()

plt.figure(fig2)
plt.xlabel("downstream distance")
plt.ylabel("alluvium thickness (m)")
plt.legend()
plt.show()

plt.figure(fig3)
plt.xlabel("downstream distance")
plt.ylabel("log alluvium thickness (log m)")
plt.legend()
plt.show()

# %%
print(ngrid["node"]["bedrock"])
print(ngrid["node"]["mean_alluvium_thickness"])
print(ngrid["node"]["sed_capacity"])

# %%
