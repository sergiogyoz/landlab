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
shape = (3, 3)
reach_lenght = 1
slope = 1  # 0.004
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
print(ngrid["node"]["bedrock"])
print(ngrid["node"]["mean_alluvium_thickness"])
print(ngrid["node"]["sed_capacity"])
# %%
t = 0
# %%
dt = 1
t += dt
# nety._set_boundary_conditions(0,q_up=0.01, t=t, )
nety.run_one_step(dt)


# %%













# %%
# trying out the model over 100 years
fig1 = plt.figure(1)
fig2 = plt.figure(2)

xs = list(range(n - 1))
dt = 0.001 * 365.25
year = 100
sed_source = np.array([0])
total_time = year * 1000  # in years
# sedimentograph

qt = 1
ngrid.at_node["sed_capacity"][sed_source] = qt

for time in range(total_time):
    if (time % (10 * year)) == 0:
        plt.figure(fig1)
        bed = ngrid.at_node["bedrock"][xs]
        plt.plot(xs, bed, label=f"iter {time}")

        plt.figure(fig2)
        alluvium = ngrid.at_node["mean_alluvium_thickness"][xs]
        plt.plot(xs, alluvium, label=f"iter {time}")
    nety.run_one_step(dt=dt, urate=0, omit=sed_source)

# %%
plt.figure(fig1)
plt.legend()
plt.show()

plt.figure(fig2)
plt.legend()
plt.show()

# %%
