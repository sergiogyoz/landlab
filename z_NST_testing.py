# %%%
import numpy as np
from landlab.components import FlowDirectorSteepest
from landlab.components import NetworkSedimentTransporter
from landlab import NetworkModelGrid
from landlab.data_record import DataRecord

# %%%

y_of_node = (0, 0, 0, 0)
x_of_node = (0, 100, 200, 300)
nodes_at_link = ((0, 1), (1, 2), (2, 3))
nmg = NetworkModelGrid((y_of_node, x_of_node), nodes_at_link)

_ = nmg.add_field("bedrock__elevation", [3., 2., 1., 0.], at="node")
_ = nmg.add_field("reach_length", [100., 100., 100.], at="link")
_ = nmg.add_field(
    "channel_width",
    15 * np.ones(nmg.size("link")),
    at="link")
_ = nmg.add_field(
    "flow_depth",
    (2 * np.ones(nmg.size("link"))),
    at="link")

_ = nmg.add_field(
    "topographic__elevation",
    np.copy(nmg.at_node["bedrock__elevation"]),
    at="node")

flow_director = FlowDirectorSteepest(nmg)
flow_director.run_one_step()

timesteps = 10
time = [0.0]

items = {"grid_element": "link",
         "element_id": np.array([[0]])}

variables = {
    "starting_link": (["item_id"], np.array([0])),
    "abrasion_rate": (["item_id"], np.array([0])),
    "density": (["item_id"], np.array([2650])),
    "time_arrival_in_link": (["item_id", "time"], np.array([[0]])),
    "active_layer": (["item_id", "time"], np.array([[1]])),
    "location_in_link": (["item_id", "time"], np.array([[0]])),
    "D": (["item_id", "time"], np.array([[0.05]])),
    "volume": (["item_id", "time"], np.array([[1]])),
}

one_parcel = DataRecord(
    nmg,
    items=items,
    time=time,
    data_vars=variables,
    dummy_elements={
        "link": [NetworkSedimentTransporter.OUT_OF_NETWORK]},
)

nst = NetworkSedimentTransporter(
    nmg,
    one_parcel,
    flow_director,
    bed_porosity=0.03,
    g=9.81,
    fluid_density=1000,
    transport_method="WilcockCrowe",
    active_layer_method="WongParker"
)

dt = 60  # (seconds) 1 min timestep

for t in range(0, (timesteps * dt), dt):
    nst.run_one_step(dt)

print(one_parcel.dataset.element_id.values)

# [[ 0.  0.  0.  0.  0.  1.  1.  1.  1.  1.  2.]]
# %%
