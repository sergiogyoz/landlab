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
                   + "2000y_Q_range_hack")
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
datas = []
for i in range(4):
    data, time, space = m1d.read_records_csv(read_folder, files[i])
    datas.append(data)
x = space["distance"].values
t = time["time"].values
# %%
# plot style
total_frames = t.size
n_stop_frames = 10  # n - 1
stop_frames = [total_frames * i // n_stop_frames for i in range(n_stop_frames + 1)]
norm = colorsmaps.Normalize(vmin=t[0], vmax=t[-1])
colormap = mpl.colormaps["plasma_r"]
mappable_object = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)

fig, axs = plt.subplots(2, 2, sharex=True)
fig.set_dpi(75)
fig.set_size_inches(24, 13.5)
fig.colorbar(mappable_object, ax=axs)

base_frame = 20  # for axis lims
lines = []
for i in range(4):
    index = (i // 2, i % 2)
    axs[index].set_title(fields[i], fontsize=20)
    axs[index].set_xlabel(xlabel, fontsize=15)
    axs[index].set_ylabel(ylabels[i], fontsize=15)
    axs[index].tick_params(labelsize=24)
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

    ymin = min(datas[i][base_frame])
    ymax = max(datas[i][base_frame])
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
        lines[i].set_data(x, datas[i][0])
        for j in range(n_stop_frames + 1):
            stoplines[i][j].set_data([], [])
            clear_stops.append(stoplines[i][j])
    return lines + clear_stops


def animate(j):
    for i in range(4):
        lines[i].set_data(x, datas[i][j])
        lines[i].set_color(colormap(norm(t[j])))

    if j in stop_frames:
        k = len(stop_index)
        addlines = []
        for i in range(4):
            stoplines[i][k].set_data(x, datas[i][j])
            stoplines[i][k].set_color(colormap(norm(t[j])))
            addlines.append(stoplines[i][k])
        changes = lines + addlines
        stop_index.append(j)
        return changes
    else:
        return lines


animation = FuncAnimation(fig, animate, frames=total_frames,
                          interval=50, blit=True, init_func=init)


plt.show()
# %%
# save animation
savedir = folder / "animations"
os.makedirs(savedir, exist_ok=True)
fname = "anim_x4" + ".mp4"
plt.show()
animation.save(savedir / fname, fps=10)
plt.close()
# %%
