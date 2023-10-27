# %%
# imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colorsmaps
import numpy as np
import pandas as pd
import math
import time as pytimer
import pathlib as path

from landlab import RasterModelGrid, imshow_grid
# from landlab import NetworkModelGrid
from landlab.components import FlowDirectorSteepest
import landlab.plot.graph as graph
from landlab.grid.create_network import network_grid_from_raster

# import my DumbComponent
from landlab.components import Componentcita as comp
from setup import setup
YEAR = 365.25 * 24 * 60 * 60


# 1D model functions
def model1D(total_length=3000,
            reach_length=200,
            initial_slope=0.004,
            discharge=300,
            intermittency=0.05,
            channel_width=100,
            sediment_size=0.02,
            initial_sed_capacity=0,
            macroroughness=1,
            initial_allu_thickness=0.5,
            allu_smooth=0.8,
            slope_smooth=0.2,
            dt=0.001 * YEAR,
            total_time=50 * YEAR,
            record_time=1 * YEAR,
            uplift=0.005 / YEAR,
            fraction_at_high_feed=0.2,  # rh
            scale_of_high_feed=1,  # rqh
            cycle_period=20 * YEAR,  # Tc
            random_seed_sed=2,
            plot_fields=False,
            extra_fields=False,
            ):
    """
    This method depends on the global variable YEARS
    """
    n = round(total_length / reach_length)
    # network from raster
    shape = (3, n + 2)
    reach_lenght = reach_length  # dummy for the raster
    slope = initial_slope
    s = setup(shape, steepness=slope * reach_lenght)
    # landlab grid
    rastergrid = RasterModelGrid(shape=shape, xy_spacing=reach_lenght)
    rastergrid.add_field("topographic__elevation", s.line())

    ngrid = network_grid_from_raster(rastergrid)

    # flow director
    flow_director = FlowDirectorSteepest(ngrid)
    flow_director.run_one_step()

    # initial values and parameters of the network
    comp.Componentcita._preset_fields(
        ngrid=ngrid,
        channel_width=channel_width,
        flood_discharge=discharge,
        flood_intermittency=intermittency,
        sediment_grain_size=sediment_size,
        sed_capacity=initial_sed_capacity,
        macroroughness=macroroughness,
        mean_alluvium_thickness=initial_allu_thickness)
    nety = comp.Componentcita(ngrid, flow_director,
                              au=allu_smooth, su=slope_smooth)

    # downstream distance for plots
    xs = np.copy(ngrid.at_node["reach_length"])
    xs = np.cumsum(xs) - xs[0]
    times = np.arange(0, total_time + dt, dt)

    # sedimentograph at the source nodes
    sed_data = comp.Componentcita.sedimentograph(
        time=times, dt=dt,
        Tc=cycle_period,
        rh=fraction_at_high_feed,
        rqh=scale_of_high_feed,
        random=False,
        random_seed=random_seed_sed)
    sedgraph = sed_data["sedgraph"]

    # plot prep
    if plot_fields:
        fields = plot_fields
    else:
        fields = ["bedrock",
                  "mean_alluvium_thickness",
                  "sed_capacity",
                  "channel_slope"]
    if extra_fields:
        extras = []
    else:
        extras = ["bed+alluvium",
                  "bed_slope",
                  "alluvium_slope"]

    record_times = list(np.arange(0, total_time + dt, record_time))
    nt = len(record_times)
    mx = ngrid.at_node.size
    records = {}
    for field in fields + extras:
        records[field] = np.zeros((nt, mx))

    # run and record model for time in times
    baselevel = ngrid["node"]["bedrock"][nety.outlets][0]
    s.start_timer()
    r_ind = 0
    for time, sed in zip(times, sedgraph):
        record = math.isclose(time, record_times[r_ind], abs_tol=dt / 2)
        if record:
            # time record
            print(f"year: {time / YEAR: .3f}")
            s.clock_timer(show=True)
            # store records
            for field in fields:
                records[field][r_ind] = ngrid.at_node[field]
            if not extra_fields:
                # alluvium + bed
                y = ngrid.at_node[fields[0]] + ngrid.at_node[fields[1]]
                records[extras[0]][r_ind] = y
                # bed slope
                y = ((ngrid.at_node[fields[0]][nety._unode]
                      - ngrid.at_node[fields[0]][nety._dnode])
                     / ngrid.at_node["reach_length"])
                y[1:-1] = y[1:-1] / 2
                records[extras[1]][r_ind] = y
                # alluvium slope
                y = ((ngrid.at_node[fields[1]][nety._unode]
                      - ngrid.at_node[fields[1]][nety._dnode])
                     / ngrid.at_node["reach_length"])
                y[1:-1] = y[1:-1] / 2
                records[extras[2]][r_ind] = y
            r_ind = r_ind + 1

        # uplift
        ngrid.at_node["bedrock"] = ngrid.at_node["bedrock"] + uplift * dt
        # fix outlet condition
        ngrid.at_node["bedrock"][nety.outlets] = baselevel
        # run one step
        nety.run_one_step(dt=dt, q_in=sed)

    context = {}
    context["x"] = xs
    context["record_times"] = record_times
    context["fields"] = fields + extras
    return context, records, sed_data


def plot_sed_graph(sed_data, name, savedir=False):
    figsed = plt.figure(0)
    times = sed_data["t"]
    sedgraph = sed_data["sedgraph"]
    plt.figure(figsed)
    sed_label = (f"fraction at high_feed = {sed_data['data']['rh']} \n"
                 f"scale_of_high_feed ={sed_data['data']['rqh']}")
    plt.plot(times / YEAR, sedgraph, label=sed_label)
    plt.title("Sedimentograph")
    plt.xlabel("time (years)")
    plt.ylabel("width average sediment flux $(m^2/s)$")
    plt.legend()
    plt.show()
    if savedir:
        fpath = path.Path(savedir)
        fname = name + "_" + "sedgraph" + ".jpg"
        plt.savefig(fpath / fname, bbox_inches='tight')


def plot_1D_fields(context, records, name, savedir=False):
    # prep
    xs = context["x"]
    r_times = context["record_times"]
    fields = context["fields"]
    figs = [plt.figure(i + 1) for i in range(len(fields))]
    norm = colorsmaps.Normalize(r_times[0], r_times[-1])
    colormap = mpl.colormaps["plasma_r"]
    r_inds = range(len(r_times))
    # plot
    for time, r in zip(r_times, r_inds):
        i = 0
        label = f"t= {time / YEAR :.3f}"
        for field in fields:
            plt.figure(figs[i])
            y = records[field][r]
            plt.plot(xs, y,
                     color=colormap(norm(time)),
                     label=label)
            i = i + 1
            # extras
        # alluvium + bed
        plt.close("all")

    # labels and titles
    xlabel = "downstream distance (m)"
    ylabels = {"bedrock": "bedrock elevation (m)",
               "mean_alluvium_thickness": "alluvium thickness (m)",
               "sed_capacity": "sediment capacity (m^2/s)",
               "channel_slope": "channel slope (m/m)",
               "bed+alluvium": "bed + alluvium (m)",
               "bed_slope": "bed slope (m/m)",
               "alluvium_slope": "alluvium slope (m/m)"}

    for fig, field in zip(figs, fields):
        plt.figure(fig)
        fig.set_dpi(75)
        fig.set_size_inches(18, 14)
        axes = fig.axes[0]
        for line in axes.get_lines():
            line.set_linewidth(2.)
        axes.set_title(field, fontsize=30)
        axes.set_xlabel(xlabel, fontsize=20)
        axes.set_ylabel(ylabels[field], fontsize=20)
        axes.tick_params(labelsize=24)
        axes.legend(fontsize=14)
        axes.set_label
        if savedir:
            fpath = path.Path(savedir)
            fname = name + "_" + field + ".jpg"
            plt.savefig(fpath / fname, bbox_inches='tight')
        plt.show()


def save_records_csv(records, savedir, fname):
    folder = path.Path(savedir)
    for field in records:
        df = pd.DataFrame(records[field])
        filename = fname + "_" + field + ".csv"
        fpath = folder / filename
        df.to_csv(fpath, header=False, index=False)

# %%
context, records, sed_data = model1D(total_time=100 * YEAR,
                                     record_time=5*YEAR)
# %%
savedir = path.Path("C:/Users/Sergio/Documents/GitHub/Sharing/Nicole/runs")
plotsdir = savedir / "plots"
datadir = savedir / "data"
name = "100y_record_5y"

# %%
plot_sed_graph(sed_data, name, plotsdir)
# %%
plot_1D_fields(context, records, name, plotsdir)
# %%
save_records_csv(records, datadir, name)
# %%
