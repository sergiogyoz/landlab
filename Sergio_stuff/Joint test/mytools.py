import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colorsmaps
import numpy as np
import pandas as pd
import math
import pathlib as path
import time as pytimer


from landlab import RasterModelGrid
# from landlab import NetworkModelGrid
from landlab.components import FlowDirectorSteepest
from landlab.grid.create_network import network_grid_from_raster
from landlab import NetworkModelGrid

# import my DumbComponent
from landlab.components import Componentcita as comp
YEAR = 365.25 * 24 * 60 * 60


# grid setup
class Grid_geometry:

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

    def Yfork(self):
        n = self.n
        m = self.m
        steepness = self.steepness
        midn = n // 2

        x_of_node = [i for i in range(n)]  # 0,1,...,n-1
        y_of_node = [0 for i in range(n)]
        steep = [(n - 1 - i) * steepness for i in range (n)]

        x_vert = [midn for i in range(1, m)]  # n,..., n+m-1
        y_vert = [i for i in range(1, m)]
        steep_vert = [steep[midn] + i * steepness for i in range (1,m)]

        links = [(i, i + 1) for i in range(n - 1)]
        links_vert = [(i, i + 1) for i in range(n, n + m - 2)]
        joint = [(n, midn)]

        x_of_node = x_of_node + x_vert
        y_of_node = y_of_node + y_vert
        links = links + links_vert + joint
        steep = steep + steep_vert
        return x_of_node, y_of_node, links, steep

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
            case 2:
                return [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0,
                        20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0,
                        20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0,
                        15.0, 14.0, 13.0, 12.0, 10.0, 8.0, 6.0,
                        20.0, 20.0, 20.0, 13.0, 20.0, 20.0, 20.0,
                        20.0, 20.0, 20.0, 14.0, 20.0, 20.0, 20.0,
                        20.0, 20.0, 20.0, 15.0, 20.0, 20.0, 20.0]


class Mytimer:

    def __init__(self):
        self.timer = 0.0

    def start(self):
        self.timer = pytimer.perf_counter()

    def clock(self, show=True):
        current_time = pytimer.perf_counter() - self.timer
        if show:
            print(f"iteration time: {current_time : .2f}")
        self.start()
        return current_time


class Sedgraph:

    @staticmethod
    def Zhang(time, dt, Tc, rh=0.25, qm=0.000834, rqh=1, random=False, **kwargs):
        sed_graph = comp.Componentcita.sedimentograph(time, dt, Tc, rh, qm,
                                                      rqh, random, **kwargs)
        return sed_graph

    @staticmethod
    def plot_sedgraph(sed_data, fprefix, savedir):
        figsed = plt.figure(0)
        times = sed_data["t"]
        sed_graph = sed_data["sedgraph"]
        plt.figure(figsed)
        sed_label = (f"fraction at high_feed = {sed_data['data']['rh']} \n"
                     f"scale_of_high_feed ={sed_data['data']['rqh']}")
        plt.plot(times / YEAR, sed_graph, label=sed_label)
        plt.title("Sedimentograph")
        plt.xlabel("time (years)")
        plt.ylabel("width average sediment flux $(m^2/s)$")
        plt.legend()
        if savedir:
            fpath = path.Path(savedir)
            fname = fprefix + "_" + "sedgraph" + ".png"
            plt.savefig(fpath / fname, bbox_inches='tight')
        plt.show()


class Model1D:
    @staticmethod
    def basegrid_1D(total_length=2000,
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
                    slope_smooth=0.2):
        """
        creates and returns a grid with all the provided parameters
        """
        n = round(total_length / reach_length)
        # network from raster
        shape = (3, n + 2)
        reach_lenght = reach_length  # dummy for the raster th ht
        slope = initial_slope
        s = Grid_geometry(shape, steepness=slope * reach_lenght)
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
            discharge=discharge,
            channel_width=channel_width,
            flood_intermittency=intermittency,
            sediment_grain_size=sediment_size,
            sed_capacity=initial_sed_capacity,
            macroroughness=macroroughness,
            mean_alluvium_thickness=initial_allu_thickness)
        nety = comp.Componentcita(ngrid, flow_director,
                                  au=allu_smooth, su=slope_smooth)

        return ngrid, nety

    @staticmethod
    def run_and_record_1D(ngrid, nety, sed_data,
                          dt=0.001 * YEAR,
                          total_time=50 * YEAR,
                          record_time=1 * YEAR,
                          uplift=0.005 / YEAR,
                          plot_fields=False,
                          extra_fields=False,
                          show_timer=True
                          ):
        # downstream distance for plots
        xs = np.copy(ngrid.at_node["reach_length"])
        xs = np.cumsum(xs) - xs[0]
        times = np.arange(0, total_time + dt, dt)
        sed_graph = sed_data["sedgraph"]

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
        timer = Mytimer()
        timer.start()
        r_ind = 0
        for time, sed in zip(times, sed_graph):
            record = math.isclose(time, record_times[r_ind], abs_tol=dt / 2)
            if record:
                # time record
                if show_timer:
                    print(f"year: {time / YEAR: .2f}")
                timer.clock(show=show_timer)
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
        return context, records

    @staticmethod  # deprecated, left for backwards compatibility
    def model1D(total_length=2000,
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
                mean_sediment_feed=0.000834,  # m^2/s
                fraction_at_high_feed=0.25,  # rh
                scale_of_high_feed=1,  # rqh
                cycle_period=40 * YEAR,  # Tc
                random_seed_sed=2,
                plot_fields=False,
                extra_fields=False,
                show_timer=True
                ):
        """
        This method depends on the global variable YEARS
        """
        n = round(total_length / reach_length)
        # network from raster
        shape = (3, n + 2)
        reach_lenght = reach_length  # dummy for the raster
        slope = initial_slope
        s = Grid_geometry(shape, steepness=slope * reach_lenght)
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
            discharge=discharge,
            channel_width=channel_width,
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
        sed_data = Sedgraph.Zhang(
            time=times, dt=dt,
            Tc=cycle_period,
            qm=mean_sediment_feed,
            rh=fraction_at_high_feed,
            rqh=scale_of_high_feed,
            random=False,
            random_seed=random_seed_sed)
        sed_graph = sed_data["sedgraph"]

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
        timer = Mytimer()
        timer.start()
        r_ind = 0
        for time, sed in zip(times, sed_graph):
            record = math.isclose(time, record_times[r_ind], abs_tol=dt / 2)
            if record:
                # time record
                if show_timer:
                    print(f"year: {time / YEAR: .2f}")
                timer.clock(show=show_timer)
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

    @staticmethod  # deprecated, moved to it's own class
    def plot_sed_graph(sed_data, fprefix, savedir):
        Sedgraph.plot_sedgraph(sed_data, fprefix, savedir)

    @staticmethod
    def plot_1D_fields(context, records, prefix, savedir,
                       from_time=False, to_time=False, suptitle=False):
        # prep
        xs = context["x"]
        r_times = context["record_times"]
        from_t = r_times[0] if not from_time else from_time
        to_t = r_times[-1] if not to_time else to_time
        fields = context["fields"]
        figs = [plt.figure(i + 1) for i in range(len(fields))]
        # only plot range from from_t to to_t
        ind_df = pd.DataFrame(r_times)
        ind_df = ind_df[ind_df[0] <= to_t]
        ind_df = ind_df[ind_df[0] >= from_t]
        # colors
        norm = colorsmaps.Normalize(ind_df.iloc[0][0], ind_df.iloc[-1][0])
        colormap = mpl.colormaps["plasma_r"]
        # plot
        for ind in ind_df.itertuples(index=True):
            time, r = ind[1], ind[0]
            label = f"t= {time / YEAR :.3f}"
            for field, fig in zip(fields, figs):
                plt.figure(fig)
                y = records[field][r]
                plt.plot(xs, y,
                         color=colormap(norm(time)),
                         label=label)
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
            if suptitle:
                fig.suptitle(suptitle, fontsize=40)
            axes.set_xlabel(xlabel, fontsize=20)
            axes.set_ylabel(ylabels[field], fontsize=20)
            axes.tick_params(labelsize=24)
            axes.legend(fontsize=14)
            if savedir:
                fpath = path.Path(savedir)
                fname = prefix + "_" + field + ".png"
                plt.savefig(fpath / fname, bbox_inches='tight')
            plt.show()

    @staticmethod
    def plot_1D_field_file(readdir, fname, ylabel, title,
                           savedir=False, suptitle=False,
                           from_time=False, to_time=False, dt_in_sec=-1):
        """
        returns the figure plotted. If savedir is provided it is saved on the
        provided directory.
        """
        # prep
        folder = path.Path(readdir)
        file = folder / fname
        tfile = folder / "time.csv"
        xfile = folder / "space.csv"
        df_data = pd.read_csv(file, header=None)
        df_t = pd.read_csv(tfile)
        df_x = pd.read_csv(xfile)

        xs = df_x["distance"]
        ts = df_t["time"]
        from_t = ts[0] if not from_time else from_time
        to_t = ts.iloc[-1] if not to_time else to_time

        fig = plt.figure(1)

        # only plot range from from_t to to_t
        onestep = (ts[1] - ts[0])
        dt_in_steps = round(dt_in_sec / onestep)
        dt_in_steps = dt_in_steps if dt_in_steps > 1 else 1

        ind_df = df_t[df_t["time"] <= to_t]
        ind_df = ind_df[ind_df["time"] >= from_t]
        ind_df = ind_df[::dt_in_steps]

        # colors
        norm = colorsmaps.Normalize(ind_df.iloc[0][0], ind_df.iloc[-1][0])
        colormap = mpl.colormaps["plasma_r"]

        # plot
        for ind in ind_df.itertuples(index=True):
            time, r = ind[1], ind[0]
            label = f"t= {time / YEAR :.3f}"
            plt.figure(fig)
            y = df_data.iloc[:, r]
            plt.plot(xs, y,
                     color=colormap(norm(time)),
                     label=label)
            plt.close("all")

        # labels and titles
        field = fname[:-4]
        xlabel = "downstream distance (m)"

        fig.set_dpi(75)
        fig.set_size_inches(18, 14)
        axes = fig.axes[0]
        for line in axes.get_lines():
            line.set_linewidth(2.)
        axes.set_title(title, fontsize=30)
        if suptitle:
            fig.suptitle(suptitle, fontsize=40)
        axes.set_xlabel(xlabel, fontsize=20)
        axes.set_ylabel(ylabel, fontsize=20)
        axes.tick_params(labelsize=24)
        axes.legend(fontsize=14)
        if savedir:
            fpath = path.Path(savedir)
            fname = field + ".png"
            plt.savefig(fpath / fname, bbox_inches='tight')
        plt.show()
        return fig

    @staticmethod
    def save_records_csv(records, savedir, fprefix="", context=False):
        folder = path.Path(savedir)
        for field in records:
            df = pd.DataFrame(records[field])
            df = df.T
            fname = fprefix + field + ".csv"
            fpath = folder / fname
            df.to_csv(fpath, header=False, index=False)
            if context:
                df2 = pd.DataFrame()
                df2["time"] = context["record_times"]
                df2.to_csv(folder / "time.csv", header=True, index=False)
                df3 = pd.DataFrame()
                df3["distance"] = context["x"]
                df3.to_csv(folder / "space.csv", header=True, index=False)

    @staticmethod
    def read_records_csv(readdir, fname):
        """
        filename with extension. headers is a bool that should be
        true if the file has headers.

        returns dfs for data, time and x
        """
        folder = path.Path(readdir)
        file = folder / fname
        tfile = folder / "time.csv"
        xfile = folder / "space.csv"
        df_data = pd.read_csv(file, header=None)
        df_t = pd.read_csv(tfile)
        df_x = pd.read_csv(xfile)
        return df_data, df_t, df_x

    @staticmethod
    def discharge_calc(downQ, upQ, dx, n, pup=0.5, pdown=0.2, hacks=0.56):
        """
        It creates discharge curves as an array of size n. This curve joins the
        downQ and upQ using dx with two different approaches.
        One uses a quadratic->linear->quadratic relationship to join them.
        The other returns Hack's law relationship.
        """
        n = n - 1
        Q = pd.DataFrame()
        Q["x"] = np.arange(0, n * dx + dx, dx)
        L = n * dx
        # hack's law applied as D = C (l + a)^h with upQ=D(0) and downQ=D(L)
        if upQ > 0:
            oneoverh = 1 / hacks
            a = L / ((downQ / upQ) ** oneoverh - 1)
            C = upQ / (a ** hacks)
        elif upQ == 0:
            a = 0
            C = downQ / (L ** hacks)
        else:
            raise (ValueError("negative discharge"))

        Q["y_hack"] = C * np.power(Q["x"] + a, hacks)

        # qlq curve always starts at Qup = 0 to make sense
        dq = [0] * (n + 1)
        li = round(n * pup)
        ri = round(n * (1 - pdown))

        long = n * dx
        mid = (1 - pdown - pup) * long
        h = 2 * downQ / (long + mid)

        step = h * dx / len(range(1, li + 1))
        for i in range(1, li + 1):
            dq[i] = dq[i - 1] + step
        step = 0
        for i in range(li, ri + 1):
            dq[i] = dq[i - 1] + step
        step = -h * dx / len(range(ri, n + 1))
        for i in range(ri, n + 1):
            dq[i] = dq[i - 1] + step
        Q["dy"] = np.array(dq)
        Q["y_qlq"] = Q["dy"].cumsum()

        Q["logx"] = np.log(Q["x"])
        Q["logyqlq"] = np.log(Q["y_qlq"])
        Q["logy_hack"] = np.log(Q["y_hack"])

        Q.plot(x="x", y=["y_qlq", "y_hack"],
               xlabel="downstream distance", ylabel="Discharge")
        Q.plot(x="x", y=["y_qlq", "y_hack"], loglog=True,
               xlabel="downstream distance", ylabel="Discharge")
        return Q
