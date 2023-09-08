# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colorsmaps
import numpy as np
import math
import time as pytimer

from landlab import RasterModelGrid, imshow_grid
# from landlab import NetworkModelGrid
from landlab.components import FlowDirectorSteepest
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
        self.timer = 0.0

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

    def start_timer(self):
        self.timer = pytimer.perf_counter()

    def clock_timer(self, show=True):
        current_time = pytimer.perf_counter() - self.timer
        if show:
            print(f"iteration time: {current_time : .2f}")
        self.start_timer()
        return current_time


# %%
# network from raster
shape = (3, 100 + 2)  # n+2
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
# prep for model
figsed = plt.figure(0)

n = ngrid.at_node.size
xs = np.arange(0, n, 1)
year = 365.25 * 24 * 60 * 60  # in seconds
dt = 0.001 * year
total_time = 10 * year  # how long to simulate in years
record_t = 1 * year  # how often to record in years
uplift = 0.005 / year  # m/s
# downstream distance for plots
distance = np.copy(ngrid.at_node["reach_length"])
distance = np.cumsum(distance) - distance[0]
times = np.arange(0, total_time + dt, dt)

# %%
# sedimentograph at the source nodes
fraction_at_high_feed = 0.25  # rh
scale_of_high_feed = 1  # rqh
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
# plot prep
fields = ["bedrock",
          "mean_alluvium_thickness",
          "sed_capacity",
          "channel_slope"]
extras = ["bed+alluvium",
          "bed_slope",
          "alluvium_slope"]

figs = [plt.figure(i) for i in range(len(fields + extras))]
custom_legend = False
if custom_legend:
    legend_times = [100 * year * i for i in range(13)]
else:
    legend_times = list(np.arange(0, total_time + dt, record_t))
norm = colorsmaps.Normalize(0, total_time)
colormap = mpl.colormaps["plasma_r"]

# %%
# run and record model for time in times
baselevel = ngrid["node"]["bedrock"][nety.outlets][0]
s.start_timer()
for time, sed in zip(times, sedgraph):
    record = math.isclose(time % record_t, 0, abs_tol=dt / 2)
    add_to_legend = False
    y_time = time / year
    if legend_times:
        add_to_legend = math.isclose(time, legend_times[0], abs_tol=dt / 2)
    if record:
        label = ""
        print(f"year: {y_time: .3f}")
        s.clock_timer(show=True)
    if add_to_legend:
        label = f"t= {y_time:.3f}"
        legend_times.pop(0)

    if record or add_to_legend:
        i = 0
        for field in fields:
            plt.figure(figs[i])
            y = ngrid.at_node[field]
            plt.plot(distance, y,
                     color=colormap(norm(time)),
                     label=label)
            i = i + 1
        # extras
        # alluvium + bed
        plt.figure(figs[i])
        y = ngrid.at_node[fields[0]] + ngrid.at_node[fields[1]]
        plt.plot(distance, y,
                 color=colormap(norm(time)),
                 label=label)
        i = i + 1
        # bed slope
        plt.figure(figs[i])
        y = ((ngrid.at_node[fields[0]][nety._unode]
              - ngrid.at_node[fields[0]][nety._dnode])
             / ngrid.at_node["reach_length"])
        y[1:-1] = y[1:-1] / 2
        plt.plot(distance, y,
                 color=colormap(norm(time)),
                 label=label)
        i = i + 1
        # bed slope
        plt.figure(figs[i]) 
        y = ((ngrid.at_node[fields[1]][nety._unode]
              - ngrid.at_node[fields[1]][nety._dnode])
             / ngrid.at_node["reach_length"])
        y[1:-1] = y[1:-1] / 2
        plt.plot(distance, y,
                 color=colormap(norm(time)),
                 label=label)
        i = i + 1

    # uplift
    ngrid.at_node["bedrock"] = ngrid.at_node["bedrock"] + uplift * dt
    # fix outlet condition
    ngrid.at_node["bedrock"][nety.outlets] = baselevel
    # run one step
    nety.run_one_step(dt=dt, q_in=sed)

# %%
# plots
titles = fields + extras
xlabel = "downstream distance (m)"
ylabels = ["bedrock elevation (m)",
           "alluvium thickness (m)",
           "sediment capacity (m^2/s)",
           "channel slope (m/m)",
           "bed + alluvium (m)",
           "bed slope (m/m)",
           "alluvium slope (m/m)"]

for fig, title, ylabel in zip(figs, titles, ylabels):
    plt.figure(fig)
    fig.set_dpi(75)
    fig.set_size_inches(18, 14)
    axes = fig.axes[0]
    for line in axes.get_lines():
        line.set_linewidth(2.)
    axes.set_title(title, fontsize=30)
    axes.set_xlabel(xlabel, fontsize=20)
    axes.set_ylabel(ylabel, fontsize=20)
    axes.tick_params(labelsize=24)
    axes.legend(fontsize=14)
    axes.set_label
    save_folder = "C:/Users/Paquito/Desktop/MRSAAc/plot2show/My_1200years/"
    fname = title + ".jpg"
    #plt.savefig(save_folder + fname, bbox_inches='tight')
    plt.show()

# %%
print(ngrid["node"]["bedrock"])
print(ngrid["node"]["mean_alluvium_thickness"])
print(ngrid["node"]["sed_capacity"])

# %%
# fields and network plot
graph.plot_graph(ngrid, at="node,link", with_id=True)
print(ngrid["link"].keys())
print(ngrid["node"].keys())

# %%
