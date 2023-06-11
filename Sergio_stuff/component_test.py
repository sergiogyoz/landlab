#%%
import numpy as np

from landlab import NetworkModelGrid
from landlab.components import FlowDirectorSteepest
import landlab.plot.graph as graph

# import my DumbComponent
from landlab.components import Componentcita as comp


def test_preset_network():
    ngrid, flow_director = comp.Componentcita._preset_network()
    assert isinstance(ngrid, NetworkModelGrid)
    assert isinstance(flow_director, FlowDirectorSteepest)


def test_preset_fields():
    ngrid, flow_director = comp.Componentcita._preset_network()
    comp.Componentcita._preset_fields(ngrid)

    ngrid.at_node.keys()
    node_keys_output = ['topographic__elevation',
                        'flow__sink_flag',
                        'flow__link_to_receiver_node',
                        'flow__receiver_node',
                        'topographic__steepest_slope',
                        'sed_capacity',
                        'bedrock',
                        'mean_alluvium_thickness',
                        'fraction_alluvium_cover']
    for key in node_keys_output:
        assert ngrid.has_field( key, at="node")

    ngrid.at_link.keys()
    link_keys_output = ['flow__link_direction',
                        'reach_length',
                        'flood_discharge',
                        'flood_intermittency',
                        'channel_width',
                        'sediment_grain_size',
                        'macroroughness',
                        'upstream_node',
                        'downstream_node',
                        'channel_slope'] 
    for key in link_keys_output:
        assert ngrid.has_field( key, at="link")



    nety = comp.Componentcita(ngrid, flow_director)


def test_critical_shear_star():
    """Test dimentionless critical shear stress calculations"""
    ngrid, flow_director = comp.Componentcita._preset_network()
    comp.Componentcita._preset_fields(ngrid)
    eroder = comp.Componentcita(ngrid, flow_director)

    tau_crit = eroder._critical_shear_star()
    tau_crit_check = np.array([0.028508183, 0.028508183, 0.028508183])
    np.testing.assert_allclose(tau_crit, tau_crit_check, rtol=10**-4)


def test_calculate_shear_star():
    """Test dimentionless shear stress calculations for current flow conditions"""
    ngrid, flow_director = comp.Componentcita._preset_network()
    comp.Componentcita._preset_fields(ngrid)
    eroder = comp.Componentcita(ngrid, flow_director)

    tau = eroder._calculate_shear_star()
    tau_check = np.array([0.15986988613, 0.15986988613, 0.15986988613])
    np.testing.assert_allclose(tau, tau_check, rtol=10**-4)


def test_update_slopes():
    """Test function that updates slopes"""
    ngrid, flow_director = comp.Componentcita._preset_network()
    comp.Componentcita._preset_fields(ngrid)
    eroder = comp.Componentcita(ngrid, flow_director)

    S_alluv = 0.01  # not divided by dx yet so 0.0001 really
    alluvium = np.array([5, 4, 2, 1]) * S_alluv
    ngrid.at_node["mean_alluvium_thickness"] = alluvium

    slopes = eroder._update_channel_slopes()
    slopes_check = np.array([0.0041, 0.0042, 0.0041])
    np.testing.assert_allclose(slopes, slopes_check, rtol=10**-4)


def test_calculate_fraction_alluvium_cover():
    """test the calculation of the fraction of alluvium cover"""
    ngrid, flow_director = comp.Componentcita._preset_network()
    comp.Componentcita._preset_fields(ngrid)
    eroder = comp.Componentcita(ngrid, flow_director)

    alluvium = np.array([0.5, 0.2, 1.1, 0])
    ngrid.at_node["mean_alluvium_thickness"] = alluvium

    eroder._calculate_fraction_alluvium_cover()
    fraction = ngrid.at_node["fraction_alluvium_cover"]
    fraction_check = np.array([0.5, 0.23, 1, 0])
    np.testing.assert_allclose(fraction, fraction_check, rtol=10**-4)


def test_calculate_sed_capacity():
    """test the calculation of the sediment capacity"""
    ngrid, flow_director = comp.Componentcita._preset_network()
    comp.Componentcita._preset_fields(ngrid)
    eroder = comp.Componentcita(ngrid, flow_director)

    tau_check = np.array([0.15986988613, 0.15986988613, 0.15986988613])  # these values are based on the presets
    eroder._calculate_sed_capacity(tau_check)
    capa = ngrid.at_node["sed_capacity"]
    capa_check = np.array([0.001668719, 0.001668719, 0.001668719, 0])
    np.testing.assert_allclose(capa, capa_check, rtol=10**-4)


def test_calculate_flow_depths():
    """test function that calculates flow depth on the network"""
    ngrid, flow_director = comp.Componentcita._preset_network()
    comp.Componentcita._preset_fields(ngrid)
    eroder = comp.Componentcita(ngrid, flow_director)

    depth = eroder._calculate_flow_depths()
    depth_check = np.array([1.318926561, 1.318926561, 1.318926561])
    np.testing.assert_allclose(depth, depth_check, rtol=10**-4)


def test_bed_erosion():
    """test the calculation of the fraction of alluvium cover"""
    ngrid, flow_director = comp.Componentcita._preset_network()
    comp.Componentcita._preset_fields(ngrid)
    eroder = comp.Componentcita(ngrid, flow_director)

    alluvium = np.array([0.5, 0.2, 1.1, 0])
    ngrid.at_node["mean_alluvium_thickness"] = alluvium

    eroder._calculate_fraction_alluvium_cover()
    fraction = ngrid.at_node["fraction_alluvium_cover"]
    fraction_check = np.array([0.5, 0.23, 1, 0])
    np.testing.assert_allclose(fraction, fraction_check, rtol=10**-4)


# %%


#%%
import numpy as np

from landlab import NetworkModelGrid
from landlab.components import FlowDirectorSteepest
import landlab.plot.graph as graph

# import my DumbComponent
from landlab.components import Componentcita as comp


ngrid, flow_director = comp.Componentcita._preset_network()
comp.Componentcita._preset_fields(ngrid)
eroder = comp.Componentcita(ngrid, flow_director)



# %%
ngrid.at_node["fraction_alluvium_cover"][eroder._unode]
# %%
eroder._unode
# %%
ngrid.at_node["fraction_alluvium_cover"][eroder._unode] = eroder._unode
# %%
