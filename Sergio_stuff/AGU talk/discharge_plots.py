# %%
# imports
import numpy as np
import pathlib as path
import os
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
# and some tools
from mytools import Model1D as m1d

sn.set_style("whitegrid")
YEAR = 365.25 * 24 * 60 * 60

# %%
# params
total_time = 2000 * YEAR
record_time = 5 * YEAR
sed_cycle = 40 * YEAR
total_length = 20000
reach_length = 200
dt = 0.001 * YEAR
initial_slope = 0.003
time = np.arange(0, total_time + dt, dt)
key = "y_hack"
ngrid, nety = m1d.basegrid_1D(total_length=total_length,
                              reach_length=reach_length,
                              initial_slope=initial_slope)
n = ngrid["node"].size
# %%
fig = plt.figure()
data = pd.DataFrame()

updischarge = 150
downdischarge = 400
Q = m1d.discharge_calc(downQ=downdischarge,
                       upQ=updischarge,
                       dx=reach_length, n=n)
data["x"] = Q["x"]
data["dis"] = Q[key]
# %%
updischarge = 150 * 2
downdischarge = 400 * 2
Q = m1d.discharge_calc(downQ=downdischarge,
                       upQ=updischarge,
                       dx=reach_length, n=n)

data["disx2"] = Q[key]
# %%
data.plot(x="x", y=["dis", "disx2"],
          legend=False,
          fontsize=12)
fig = plt.gcf()
fig.set_size_inches((14, 10))
ax = fig.gca()
ax.set_ylim(bottom=0)
ax.set_xlabel("donwstream distance (m)", fontsize=14)
ax.set_ylabel("discharge (m^3/s)", fontsize=14)
sn.despine(left=True, bottom=True)


# %%

# folder to save plot
# desktop
folder_name = "discharge_plots"
"""savedir = path.Path("C:/Users/Sergio/Documents/"
                    + "GitHub/Sharing/Nicole/runs/discharge/"
                    + folder_name)
"""
# laptop
savedir = path.Path("C:/Users/Paquito/Desktop/"
                    + "GitHub/Sharing/Nicole/runs/discharge/"
                    + folder_name)
os.makedirs(savedir, exist_ok=True)
fig.savefig(savedir / "discharge")

# %%
