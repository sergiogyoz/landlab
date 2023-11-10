# %%
# imports
import numpy as np
import pathlib as path
import os
# and some tools
from mytools import Sedgraph
from mytools import Model1D as m1d

YEAR = 365.25 * 24 * 60 * 60

# %%
# params
total_time = 1000 * YEAR
record_time = 100 * YEAR
sed_cycle = 40 * YEAR
total_length = 20000
reach_length = 200
dt = 0.001 * YEAR
time = np.arange(0, total_time + dt, dt)
discharge = 300

# %%
# setup
sed_data = Sedgraph.Zhang(time, dt, sed_cycle)
ngrid, nety = m1d.basegrid_1D(total_length=total_length,
                              reach_length=reach_length)
n = ngrid["node"].size
Q = m1d.discharge_calc(downQ=discharge, dx=reach_length, n=n)

ngrid["node"]["discharge"] = Q["y_hack"]

# %%
# run
context, records = m1d.run_and_record_1D(ngrid=ngrid,
                                         nety=nety,
                                         sed_data=sed_data,
                                         dt=dt,
                                         total_time=total_time,
                                         record_time=record_time)

# %%
# folder to save run results
folder_name = "1000y_qlq_hack"
filesname = ""
savedir = path.Path("C:/Users/Sergio/Documents/"
                    + "GitHub/Sharing/Nicole/runs/discharge/"
                    + folder_name)
# create folders
plotsdir = savedir / "plots"
datadir = savedir / "data"
os.makedirs(plotsdir, exist_ok=True)
os.makedirs(datadir, exist_ok=True)
# %%
m1d.plot_sed_graph(sed_data, filesname, plotsdir)
m1d.plot_1D_fields(context, records, filesname, savedir=plotsdir)
m1d.save_records_csv(records, datadir, filesname)
# %%
