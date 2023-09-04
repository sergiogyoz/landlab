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
# network from raster
shape = (3, 6)
reach_lenght = 200  # dummy for the raster
slope = 0.004  # 0.004
s = setups(shape, steepness=slope * reach_lenght)
# landlab grid
rastergrid = RasterModelGrid(shape=shape, xy_spacing=reach_lenght)
rastergrid.add_field("topographic__elevation", s.line())
imshow_grid(
    rastergrid,
    rastergrid.at_node["topographic__elevation"],
    cmap='inferno_r')

ngrid = network_grid_from_raster(rastergrid)
print(rastergrid.fields())

# %%
flow_director = FlowDirectorSteepest(ngrid)
flow_director.run_one_step()
# %%
# initial values and parameters of the network
nodes1 = np.ones(ngrid.at_node.size)
links1 = np.ones(ngrid.at_link.size)
reach_lenght = 200
discharge = 300
intermittency = 0.05
channel_width = 100
D = 0.02
sed_capacity = 0
macroroughness = 1
alluvium = 0.5
comp.Componentcita._preset_fields(
    ngrid=ngrid,
    channel_width=channel_width,
    flood_discharge=discharge,
    flood_intermittency=intermittency,
    sediment_grain_size=D,
    sed_capacity=sed_capacity,
    macroroughness=macroroughness,
    mean_alluvium_thickness=alluvium)

nety = comp.Componentcita(ngrid, flow_director, clobber=True)

# %%
# fields and network plot
graph.plot_graph(ngrid, at="node,link", with_id=True)
print(ngrid["link"].keys())
print(ngrid["node"].keys())

# %%
# prep for model
fig1 = plt.figure(1)
fig2 = plt.figure(2)
fig3 = plt.figure(3)
fig4 = plt.figure(4)
figsed = plt.figure(0)

n = len(ngrid.at_node["sed_capacity"])
xs = list(range(n))
year = 365.25 * 24 * 60 * 60  # in seconds
dt = 0.001 * year
total_time = 0.02 * year  # how long to simulate in years
record_t = 0.001 * year  # how often to record in years
sed_source = np.array([0])
# downstream distance for plots
distance = np.zeros_like(ngrid.at_node["sed_capacity"])
distance[1:] = np.cumsum(ngrid.at_link["reach_length"])
times = np.arange(0, total_time, dt)

# %%
# sedimentograph at the source nodes
fraction_at_high_feed = 2.5 / 40
scale_of_high_feed = 9
cycle_period = 40 * year
random_seed = 2
sedgraph = comp.Componentcita.sedimentograph(
    time=times, dt=dt,
    Tc=cycle_period,
    rh=fraction_at_high_feed,
    rqh=scale_of_high_feed,
    random=False,
    random_seed=random_seed)
plt.figure(figsed)
plt.plot(times / year, sedgraph,
         label=f"rh ={fraction_at_high_feed}, rqh ={scale_of_high_feed}")
plt.title("Sedimentograph")
plt.legend()


# %%
i = 0
ngrid.at_node["sed_capacity"][:] = sedgraph[0]
for time in times:
    ngrid.at_node["sed_capacity"][sed_source] = sedgraph[i]
    # print(ngrid.at_node["sed_capacity"])
    if math.isclose(time % record_t, 0, abs_tol=dt / 3):
        plt.figure(fig1)
        bed = ngrid.at_node["bedrock"][xs]
        plt.plot(distance, bed, label=f"t= {time/(year):.3f}")

        plt.figure(fig2)
        alluvium = ngrid.at_node["mean_alluvium_thickness"][xs]
        plt.plot(distance, alluvium, label=f"t= {time/(year):.3f}")

        plt.figure(fig3)
        plt.plot(distance, bed + alluvium, label=f"t= {time/(year):.3f}")

        plt.figure(fig4)
        sediment = ngrid.at_node["sed_capacity"][xs]
        plt.plot(distance, sediment, label=f"t= {time/(year):.3f}")

        print(f"{time / (year):.3f}") #,end=' ', flush=True)

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
plt.ylabel("bed + alluvium (m)")
plt.legend()
plt.show()

plt.figure(fig4)
plt.xlabel("downstream distance")
plt.ylabel("sediment capacity (m^2/s)")
plt.legend()
plt.show()

# %%
print(ngrid["node"]["bedrock"])
print(ngrid["node"]["mean_alluvium_thickness"])
print(ngrid["node"]["sed_capacity"])

# %%
