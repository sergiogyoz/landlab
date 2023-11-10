# %%
# imports
import numpy as np
import pathlib as path
import os

# and some tools
from mytools import Model1D as m1d
YEAR = 365.25 * 24 * 60 * 60


# %%
n = 3 + 1
s_range = np.linspace(0, 1, n)
s_m, s_a = np.meshgrid(s_range, s_range)  # slope and alluvium

# %%
for i in range(n):
    for j in range(n):
        print(f"i = {i}, j = {j}")
        context, records, sed_data = 0, 0, 0
        context, records, sed_data = m1d.model1D(total_time=3 * YEAR,
                                                 record_time=1 * YEAR,
                                                 total_length=10000,
                                                 scale_of_high_feed=3,
                                                 fraction_at_high_feed=0.25,
                                                 cycle_period=40 * YEAR,
                                                 slope_smooth=s_m[i, j],
                                                 allu_smooth=s_a[i, j],
                                                 show_timer=False)
        # folder to save run results
        folder_name = "i_" + str(i) + "  j_" + str(j)
        filesname = ""
        savedir = path.Path("C:/Users/Sergio/Documents/"
                            + "GitHub/Sharing/Nicole/runs/param_analysis/"
                            + folder_name)
        # create folders
        plotsdir = savedir
        datadir = savedir
        os.makedirs(plotsdir, exist_ok=True)
        os.makedirs(datadir, exist_ok=True)

        # m1d.plot_sed_graph(sed_data, filesname, plotsdir)
        m1d.plot_1D_fields(context, records, filesname, savedir=plotsdir,
                           from_time=False, to_time=False,
                           suptitle=f"alluv s= {s_a[i,j]:.1f}     slope s = {s_m[i,j]:.1f}")
        m1d.save_records_csv(records, datadir, filesname, context)
# %%
