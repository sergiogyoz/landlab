# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colorsmaps
from mytools import Model1D as m1d
import pathlib as path
import os

# animation
from matplotlib.animation import FuncAnimation

YEAR = 365.25 * 24 * 60 * 60
# %%
# read file
folder = path.Path("C:/Users/Sergio/Documents/"
                   + "GitHub/Sharing/Nicole/runs/discharge/"
                   + "2000_D_doublehack")
read_folder = folder / "data"

files = ["bedrock.csv",
         "sed_capacity.csv",
         "mean_alluvium_thickness.csv",
         "bed+alluvium.csv"]
fields = ["bedrock",
          "sed capacity",
          "alluvium",
          "bed+alluvium"]
units = [" (m)", " (m^2/s)", " (m)", " (m)"]
xlabel = "downstream distance"
ylabels = [fields[i] + units[i] for i in range(4)]
alldata = []
for i in range(4):
    data, time, space = m1d.read_records_csv(read_folder, files[i])
    alldata.append(data)

fromtime = 0 * YEAR
totime = 1500 * YEAR

x = space["distance"].values
ind = (fromtime <= time["time"]) & (time["time"] <= totime)
t = time["time"].values / YEAR


# %%
# extra stuff

"""
m1d.plot_1D_field_file(read_folder,
                       files[1],
                       ylabel=fields[1] + units[1],
                       title=fields[1],
                       from_time=10 * YEAR,
                       to_time=90 * YEAR,
                       dt_in_sec=10 * YEAR)
"""
# %%

# %%
# plot style
total_frames = t[ind].size
frames = list(ind[ind].index)
base_frame = 30  # for axis lims
n_stop_frames = 15  # = n - 1
stop_frames = [0] * (n_stop_frames + 1)
for i in range(n_stop_frames + 1):
    stop_frames[i] = frames[0] + (total_frames * i // n_stop_frames)

fig, axs = plt.subplots(2, 2, sharex=True)
fig.set_dpi(75)
fig.set_size_inches(24, 13.5)

norm = colorsmaps.Normalize(vmin=t[ind][0], vmax=t[ind][-1])
colormap = mpl.colormaps["plasma_r"]
mappable_object = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
cbar = fig.colorbar(mappable_object, ax=axs)
cbar.set_label("time (yrs)", fontsize=15)
cbar.ax.tick_params(labelsize=15)

fig.subplots_adjust(right=0.75, wspace=0.2, hspace=0.2)

lines = []
for i in range(4):
    index = (i // 2, i % 2)
    axs[index].set_title(fields[i], fontsize=20)
    axs[index].set_xlabel(xlabel, fontsize=15)
    axs[index].set_ylabel(ylabels[i], fontsize=15)
    axs[index].tick_params(labelsize=15)
    line, = axs[index].plot([], [])
    lines.append(line)
    lines[i].set_linewidth(2.)

stoplines = [[], [], [], []]
for i in range(4):
    index = (i // 2, i % 2)
    for _j in range(n_stop_frames + 1):
        line, = axs[index].plot([], [])
        stoplines[i].append(line)

# set xlim and ylim
for i in range(4):
    index = (i // 2, i % 2)

    ymin = min(alldata[i][frames[0] + base_frame])
    ymax = max(alldata[i][frames[0] + base_frame])
    yrange = ymax - ymin
    axs[index].set_ylim((ymin - yrange * 0.1 , ymax + yrange * 0.1))

    xmin = min(x)
    xmax = max(x)
    xrange = xmax - xmin
    axs[index].set_xlim((xmin , xmax))


# %%
# animation
stop_index = []


def init():
    stop_index.clear()
    clear_stops = []
    for i in range(4):
        lines[i].set_data(x, alldata[i][0])
        for j in range(n_stop_frames + 1):
            stoplines[i][j].set_data([], [])
            clear_stops.append(stoplines[i][j])
    return lines + clear_stops


def animate(j):
    for i in range(4):
        lines[i].set_data(x, alldata[i][j])
        lines[i].set_color(colormap(norm(t[j])))

    if j in stop_frames:
        k = len(stop_index)
        addlines = []
        for i in range(4):
            stoplines[i][k].set_data(x, alldata[i][j])
            stoplines[i][k].set_color(colormap(norm(t[j])))
            addlines.append(stoplines[i][k])
        stop_index.append(j)
        return lines + addlines
    else:
        return lines


animation = FuncAnimation(fig, animate, frames=frames,
                          interval=50, blit=True, init_func=init)


plt.show()
# %%
# save animation
savedir = folder / "animations"
os.makedirs(savedir, exist_ok=True)
fname = "anim_x4" + ".mp4"
plt.show()
animation.save(savedir / fname, fps=30)
plt.close()
# %%
