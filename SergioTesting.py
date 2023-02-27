#%%
from landlab import RasterModelGrid, imshow_grid
import matplotlib.pyplot as plt
# import my DumbComponent
from landlab.components import Componentcita as comp
#%%
# landlab grid
mygrid = RasterModelGrid(shape=(5, 5), xy_spacing=1)
mygrid.add_field("topographic__elevation",
                 [11.0, 3.0, 2.0, 11.0,
                  12.0, 3.0, 4.0, 12.0,
                  13.0, 5.0, 4.0, 13.0,
                  14.0, 5.0, 6.0, 14.0,
                  15.0, 7.0, 6.0, 15.0]
                 )

# plot the dumb_height values on the nodes
imshow_grid(
    mygrid,
    mygrid.at_node["topographic__elevation"],
    cmap='inferno_r')
#%%
# Here I create an instance of my component
mydummy = comp(mygrid, 0, 0.5)
print(mydummy.fields())
# plot the component topographic elevation field
plt.figure()
imshow_grid(
    mydummy.grid,
    mydummy.grid.at_node["topographic__elevation"],
    cmap='inferno_r')
#%%
# run one time
mydummy.run_one_step(dt=5)
plt.figure()
imshow_grid(
    mydummy.grid,
    mydummy.grid.at_node["topographic__elevation"],
    cmap='inferno_r')
