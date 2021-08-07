import numpy as np
from landlab import RasterModelGrid, imshow_grid
import matplotlib.pyplot as plt
#import my DumbComponent
from landlab.components import DumbC as dumb

mygrid = RasterModelGrid((3, 3),1);


#add a float field dumb_height to the grid at every node 
mygrid.add_field("dumb_height",[0.0,0.0,0.0,2.0,2.0,2.0,4.0,4.0,4.0]);

#plot the dumb_height values on the nodes
imshow_grid(mygrid,mygrid.at_node["dumb_height"],cmap='inferno_r');
plt.show();


#Here I create an instance of my DumbComponent component
mydummy=dumb(mygrid, bh=10, s=0.1);
#plot the component field
plt.figure();
imshow_grid(mygrid,mydummy.dhs,cmap='inferno_r');

#run one time 
mydummy.run_one_step(dt=2);
#plot again
plt.figure();
imshow_grid(mygrid,mydummy.dhs,cmap='inferno_r');


#I use the update instead of the time_step a few times
for i in range(5):
    mydummy.update_dumb_heights();
#plot again
plt.figure();
imshow_grid(mygrid,mydummy.dhs,cmap='inferno_r');

#I use the update instead of the time_step several times
for i in range(500):
    mydummy.update_dumb_heights();
#plot again
plt.figure();
imshow_grid(mygrid,mydummy.dhs,cmap='inferno_r');