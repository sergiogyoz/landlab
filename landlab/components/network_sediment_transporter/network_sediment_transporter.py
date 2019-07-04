#!/usr/env/python

"""Landlab component that simulates xxxxxx

info about the component here

Fixes that need to happen:

    -- Fix capacity calculation ~line 404. Still a placeholder. 
    
    -- Channel width-- why is this anything other than a link attribute??
    
    -- Need to find better way to define filterarrays in time and item_id
	current option is very clunky. The one Nathan Lyons suggested doesn't work.\

        KRB note: I've made some updates on this. Its not perfect, but better.
        I think I understand why this is difficult.
        1) Sometimes we just want the present timesteps data records, but the
            datarecord has everything in it.
        2) If you index the data record, it doesn't actually return a new TRUE data
            record that includes attributes like self._grid...
        3) Indexing is best supported for the COORDINATES (time and record ID)... we want
            typically, to first index on time, and then based on variables (e.g. where
            in the network, what size fraction). THis is hard to do without (2) being functional.

    -- What to do with parcels when they get to the last link?
            -KRB has dummy element_id in the works

    -- JC: I found two items that I think should be changed in the _calc_transport_wilcock_crowe and I made these changes
            - frac_parcels was the inverse of what it should be so instead of a fraction it was a number >1
            - in unpacking W* we had a (1-frac_sand) this was fine when we were treating sand and gravel separately,
              but now that we are treating all parcels together, I no longer think this should be there, because if we are trying
              to move a sand parcel then this (1-frac_sand) does not make sense. I think this is  now equivalent to the original WC2003.
              Before it was equivalent to WC2003 as implemented in Cui TUGS formulation.
            - finally, I added the calculation of a parcel velocity instead of the travel time. I think this is
              better suited to the parcel centric spirit of the code. It was also needed to simplify move_parcel_downstream
              Plus, I think one day we will have a better way to parameterize parcel virtual velocity and this will then be
              easy to incorporate/update.

    DONE -- Need to calculate distance a parcel travels in a timestep for abrasion
    JC: now complete with a rewrite of _move_parcel_downstream; this rewrite should be easier to understand
        I tried to test, but was having trouble putting in values for abrasion rate that would let the code run.

    DONE -- The abrasion exponent is applied to diameter, but doesn't impact parcel volume. Need to fix.
    JC: complete

    -- Fix inelegant time indexing

    -- Looks to me that as part of run-one-step the element_id variable in parcels is being changed from
       An int to a float. I haven't tracked down why... but I think it should stay as an int.
       ^ JC: This was happening for me for current_link[p] in move_parcel_downstream
           ... I fixed for now with int() but would be good to figure this out.
    
.. codeauthor:: Jon Allison Katy

Created on Tu May 8, 2018
Last edit ---
"""

import numpy as np

# %% Import Libraries
from landlab import BAD_INDEX_VALUE, Component
from landlab.data_record import DataRecord
from landlab.grid.network import NetworkModelGrid
from landlab.components import FlowDirectorSteepest
from landlab.utils.decorators import use_file_name_or_kwds

_SUPPORTED_TRANSPORT_METHODS = ["WilcockCrowe"]

_OUT_OF_NETWORK = BAD_INDEX_VALUE - 1


_ACTIVE = 1
_INACTIVE = 0


def _recalculate_channel_slope(z_up, z_down, dx, threshold=1e-4):
    """Recalculate channel slope based on elevation.

    Parameters
    ----------
    z_up : float
        Upstream elevation.
    z_down : float
        Downstream elevation.
    dz : float
        Distance.

    Examples
    --------
    >>> from landlab.components.network_sediment_transporter.network_sediment_transporter import _recalculate_channel_slope
    >>> import pytest
    >>> _recalculate_channel_slope(10., 0., 10.)
    1.0
    >>> _recalculate_channel_slope(0., 0., 10.)
    0.0001
    >>> with pytest.raises(ValueError):
    ...     _recalculate_channel_slope(0., 10., 10.)

    """
    chan_slope = (z_up - z_down) / dx

    if chan_slope < 0.0:
        raise ValueError("NST Channel Slope Negative")

    if chan_slope < threshold:
        chan_slope = threshold

    return chan_slope


class NetworkSedimentTransporter(Component):
    """Network bedload morphodynamic component.

    Landlab component designed to calculate _____.
    info info info

    **Usage:**
    Option 1 - Basic::
        NetworkSedimentTransporter(grid,
                             parcels,
                             transporter = asdfasdf,
                             discharge,
                             channel_geometry,
                             active_layer_thickness)

    Examples
    ----------
    >>> from landlab import RasterModelGrid
    >>> from landlab.components import FlowDirectorSteepest, FlowAccumulator, NetworkSedimentTransporter
    >>> from landlab.components.landslides import LandslideProbability
    >>> import numpy as np

    Do setup of various sorts (grid, width, parcels, discharge, etc)

    Set up NetworkSedimentTransporter

    >>> nst = NetworkSedimentTransporter(grid, stuff)

    Run NetworkSedimentTransporter forward 10 timesteps of size 10 time units.

    >>> for _ in range(10):
    ...     nst.run_one_step(10.)

    Now lets double check we got the right answer

    We'd put code here that showed some results. Our goal here is not so much
    to test the code but to show how it is used and what it does. We would make
    addtional tests in a folder called test that would completely test the code.

    """

    # component name
    _name = "NetworkSedimentTransporter"
    __version__ = "1.0"

    # component requires these values to do its calculation, get from driver
    _input_var_names = (
        "topographic__elevation",
        "channel_slope",
        "link_length",
        "channel_width",
        "flow_depth",
    )

    #  component creates these output values
    _output_var_names = ("thing", "thing")

    # units for each parameter and output
    _var_units = {
        "topographic__elevation": "m",
        "channel_slope": "m/m",
        "link_length": "m",
        "channel_width": "m",
        "flow_depth": "m",
    }

    # grid centering of each field and variable
    _var_mapping = {
        "topographic__elevation": "node",
        "channel_slope": "link",
        "link_length": "link",
        "channel_width": "link",
        "flow_depth": "link",
    }

    # short description of each field
    _var_doc = {
        "surface_water__depth": "Depth of streamflow on the surface",
        "topographic__elevation": "Topographic elevation at that node",
        "channel_slope": "Slope of the river channel through each reach",
        "link_length": "Length of each reach",
        "channel_width": "Flow width of the channel, assuming constant width",
        "flow_depth": "Depth of stream flow in each reach",
    }

    # Run Component
    #    @use_file_name_or_kwds
    #   Katy! We had to comment out ^ that line in order to get NST to instantiate. Back end changes needed.

    def __init__(
        self,
        grid,
        parcels,
        flow_director,
        flow_depth,
        active_layer_thickness,
        bed_porosity=0.3,
        g=9.81,
        fluid_density=1000.,
        channel_width="channel_width",
        transport_method="WilcockCrowe",
        **kwds
    ):
        """
        Parameters
        ----------
        grid: NetworkModelGrid
            A landlab network model grid in which links are stream channel 
            segments. 
        parcels: DataRecord
            A landlab DataRecord describing the characteristics and location of 
            sediment "parcels". 
        flow_director: FlowDirectorSteepest
            A landlab flow director. Currently, must be FlowDirectorSteepest. 
        flow_depth: float, numpy array of shape (timesteps,links)
            Flow depth of water in channel at each link at each timestep. (m)
        active_layer_thickness: float
            Depth of the sediment layer subject to fluvial transport (m)
            DANGER DANGER-- this is unused right now. check capacity calculation.
        bed_porosity: float, optional
            Proportion of void space between grains in the river channel bed. 
            Default value is 0.3.
        g: float, optional
            Acceleration due to gravity. Default value is 9.81 (m/s^2)
        fluid_density: float, optional
            Density of the fluid (generally, water) in which sediment is 
            moving. Default value is 1000 (kg/m^3)
        channel_width: float, optional 
            DANGER DANGER-- Why don't we have this attached to the grid?
        transport_method: string
            Sediment transport equation option. Default (and currently only) 
            option is "WilcockCrowe". 
        """
        super(NetworkSedimentTransporter, self).__init__(grid, **kwds)

        self._grid = grid

        if not isinstance(grid, NetworkModelGrid):
            msg = "NetworkSedimentTransporter: grid must be NetworkModelGrid"
            raise ValueError(msg)

        self._parcels = parcels

        if not isinstance(parcels, DataRecord):
            msg = (
                "NetworkSedimentTransporter: parcels must be an instance"
                "of DataRecord"
            )
            raise ValueError(msg)

        self._num_parcels = self._parcels.dataset.element_id.size

        self.parcel_attributes = [
            "time_arrival_in_link",
            "active_layer",
            "location_in_link",
            "D",
            "volume",
        ]

        # assert that the flow director is a component and is of type
        # FlowDirectorSteepest

        if not isinstance(flow_director, FlowDirectorSteepest):
            msg = (
                "NetworkSedimentTransporter: flow_director must be "
                "FlowDirectorSteepest."
            )
            raise ValueError(msg)

        # save reference to flow director
        self.fd = flow_director
        self.flow_depth = flow_depth
        self.bed_porosity = bed_porosity
        
        if not 0 <= self.bed_porosity < 1:
            msg = "NetworkSedimentTransporter: bed_porosity must be" "between 0 and 1"
            raise ValueError(msg)

        self.active_layer_thickness = active_layer_thickness

        # NOTE: variable active_layer_thickness Wong et al 2007
        # "a predictor for active layer thickness that increases with both grain size and Shields number, i.e., equations (50), (17), (18), and (23);

        self.g = g
        self.fluid_density = fluid_density
        self._time_idx = 0
        self._time = 0.0

        if transport_method in _SUPPORTED_TRANSPORT_METHODS:
            self.transport_method = transport_method
        else:
            msg = "Transport Method not supported"
            raise ValueError(msg)
        # self.transport_method makes it a class variable, that can be accessed within any method within this class
        if self.transport_method == "WilcockCrowe":
            self.update_transport_time = self._calc_transport_wilcock_crowe

        self._width = self._grid.at_link[channel_width]

        if "channel_width" not in self._grid.at_link:
            msg = (
                "NetworkSedimentTransporter: channel_width must be assigned"
                "to the grid links"
            )
            raise ValueError(msg)

        if "topographic__elevation" not in self._grid.at_node:
            msg = (
                "NetworkSedimentTransporter: topographic__elevation must be "
                "assigned to the grid nodes"
            )
            raise ValueError(msg)

        if "bedrock__elevation" not in self._grid.at_node:
            msg = (
                "NetworkSedimentTransporter: topographic__elevation must be "
                "assigned to the grid nodes"
            )
            raise ValueError(msg)

        if "link_length" not in self._grid.at_link:
            msg = (
                "NetworkSedimentTransporter: link_length must be assigned"
                "to the grid links"
            )
            raise ValueError(msg)

        if "drainage_area" not in self._grid.at_link:
            msg = (
                "NetworkSedimentTransporter: channel_width must be assigned"
                "to the grid links"
            )
            raise ValueError(msg)

        # create field for channel slope if it doesnt exist yet.
        if "channel_slope" not in self._grid.at_link:
            self._channel_slope = self._grid.zeros(at="node")
            self._update_channel_slopes()
        else:
            self._channel_slope = self._grid.at_link["channel_slope"]
        
        if "time_arrival_in_link" not in self._parcels.dataset:
            msg = ("NetworkSedimentTransporter: time_arrival_in_link must be" 
                   "assigned to the parcels"
                   )
            raise ValueError(msg)
            
        if "starting_link" not in self._parcels.dataset:
            msg = ("NetworkSedimentTransporter: starting_link must be" 
                   "assigned to the parcels"
                   )
            raise ValueError(msg)
        
        if "abrasion_rate" not in self._parcels.dataset:
            msg = ("NetworkSedimentTransporter: abrasion_rate must be" 
                   "assigned to the parcels"
                   )
            raise ValueError(msg)
            
        if "density" not in self._parcels.dataset:
            msg = ("NetworkSedimentTransporter: density must be" 
                   "assigned to the parcels"
                   )
            raise ValueError(msg)
            
        if "active_layer" not in self._parcels.dataset:
            msg = ("NetworkSedimentTransporter: active_layer must be" 
                   "assigned to the parcels"
                   )
            raise ValueError(msg)
        
        if "location_in_link" not in self._parcels.dataset:
            msg = ("NetworkSedimentTransporter: location_in_link must be" 
                   "assigned to the parcels"
                   )
            raise ValueError(msg)

        if "D" not in self._parcels.dataset:
            msg = ("NetworkSedimentTransporter: D (grain size) must be" 
                   "assigned to the parcels"
                   )
            raise ValueError(msg)
            
        if "volume" not in self._parcels.dataset:
            msg = ("NetworkSedimentTransporter: volume must be" 
                   "assigned to the parcels"
                   )
            raise ValueError(msg)
            
            
    @property
    def time(self):
        """Return current time."""
        return self._time

    def _create_new_parcel_time(self):
        """ If we are going to track parcels through time in DataRecord, we
        need to add a new time column to the parcels dataframe. This method simply
        copies over the attributes of the parcels from the former timestep.
        Attributes will be updated over the course of this step.
        """

        if self._time_idx != 0:

            self._parcels.add_record(time=[self._time])
            # ^ what's the best way to handle time?
            #            self._parcels.dataset['grid_element'].values[:,self._time_idx] = self._parcels.dataset[
            #                    'grid_element'].values[:,self._time_idx-1]
            #
            #            self._parcels.dataset['element_id'].values[:,self._time_idx] = self._parcels.dataset[
            #                    'element_id'].values[:,self._time_idx-1]

            self._parcels.ffill_grid_element_and_id()

            for at in self.parcel_attributes:
                self._parcels.dataset[at].values[
                    :, self._time_idx
                ] = self._parcels.dataset[at].values[:, self._time_idx - 1]

        self._find_now = self._parcels.dataset.time == self._time
        self._this_timesteps_parcels = np.zeros_like(
            self._parcels.dataset.element_id, dtype=bool
        )
        self._this_timesteps_parcels[:, -1] = True

        self._parcels_off_grid = (
            self._parcels.dataset.element_id[:, -1] == _OUT_OF_NETWORK
        )
        self._this_timesteps_parcels[self._parcels_off_grid, -1] = False

    def _update_channel_slopes(self):
        """text Can be simple-- this is what this does. 'private' functions can
        have very simple examples, explanations. Essentially note to yourself"""
        # Katy think this can be vectorized
        # Jon agrees, but is not sure yet how to do that
        for i in range(self._grid.number_of_links):

            upstream_node_id = self.fd.upstream_node_at_link()[i]
            downstream_node_id = self.fd.downstream_node_at_link()[i]

            self._channel_slope[i] = _recalculate_channel_slope(
                self._grid.at_node["topographic__elevation"][upstream_node_id],
                self._grid.at_node["topographic__elevation"][downstream_node_id],
                self._grid.length_of_link[i],
            )

    def _partition_active_and_storage_layers(self, **kwds):
        """For each parcel in the network, determines whether it is in the
        active or storage layer during this timestep, then updates node elevations
        """

        vol_tot = self._parcels.calc_aggregate_value(
            np.sum, "volume", at="link", filter_array=self._this_timesteps_parcels
        )
        vol_tot[np.isnan(vol_tot) == 1] = 0

        capacity = 2 * np.ones(
            self._num_parcels
        )  # REPLACE with real calculation for capacity

        for i in range(self._grid.number_of_links):

            if vol_tot[i] > 0:  # only do this check capacity if parcels are in link

                # First In Last Out.
                parcel_id_thislink = np.where(
                    self._parcels.dataset.element_id[:, self._time_idx] == i
                )[0]

                time_arrival_sort = np.flip(
                    np.argsort(
                        self._parcels.get_data(
                            time=[self._time],
                            item_id=parcel_id_thislink,
                            data_variable="time_arrival_in_link",
                        ),
                        0,
                    )
                )

                parcel_id_time_sorted = parcel_id_thislink[time_arrival_sort]

                cumvol = np.cumsum(
                    self._parcels.dataset.volume[parcel_id_time_sorted, self._time_idx]
                )

                idxinactive = np.where(cumvol > capacity[i])
                make_inactive = parcel_id_time_sorted[idxinactive]
                # idxbedabrade = np.where(cumvol < 2*capacity[i] and cumvol > capacity[i])
                # ^ syntax is wrong, but this is where we can identify the surface of the bed
                # for abrasion, I think we would abrade the particles in the active layer in active transport
                # and abrade the particles sitting on the bed. This line would identify those particles on
                # the bed that also need to abrade due to impacts from the sediment moving above.

                self._parcels.set_data(
                    time=[self._time],
                    item_id=parcel_id_thislink,
                    data_variable="active_layer",
                    new_value=_ACTIVE,
                )

                self._parcels.set_data(
                    time=[self._time],
                    item_id=make_inactive,
                    data_variable="active_layer",
                    new_value=_INACTIVE,
                )

        # Update Node Elevations

        # set active here. reference it below in wilcock crowe
        self._active_parcel_records = (
            self._parcels.dataset.active_layer == _ACTIVE
        ) * (self._this_timesteps_parcels)

        vol_act = self._parcels.calc_aggregate_value(
            np.sum, "volume", at="link", filter_array=self._active_parcel_records
        )
        vol_act[np.isnan(vol_act) == 1] = 0

        self.vol_stor = (vol_tot - vol_act) / (1 - self.bed_porosity)

    # %%
    def _adjust_node_elevation(self):
        """Adjusts slope for each link based on parcel motions from last
        timestep and additions from this timestep.
        """

        number_of_contributors = np.sum(
            self.fd.flow_link_incoming_at_node() == 1, axis=1
        )
        downstream_link_id = self.fd.link_to_flow_receiving_node[
            self.fd.downstream_node_at_link()
        ]
        upstream_contributing_links_at_node = np.where(
            self.fd.flow_link_incoming_at_node() == 1, self._grid.links_at_node, -1
        )

        # Update the node topographic elevations depending on the quantity of stored sediment
        for n in range(self._grid.number_of_nodes):

            if number_of_contributors[n] > 0:  # we don't update head node elevations

                upstream_links = upstream_contributing_links_at_node[n]
                real_upstream_links = upstream_links[upstream_links != BAD_INDEX_VALUE]
                width_of_upstream_links = self._grid.at_link["channel_width"][
                    real_upstream_links
                ]
                length_of_upstream_links = self._grid.length_of_link[
                    real_upstream_links
                ]

                width_of_downstream_link = self._grid.at_link["channel_width"][
                    downstream_link_id
                ][n]
                length_of_downstream_link = self._grid.length_of_link[
                    downstream_link_id
                ][n]

                if (
                    downstream_link_id[n] == BAD_INDEX_VALUE
                ):  # I'm sure there's a better way to do this, but...
                    length_of_downstream_link = 0

                alluvium__depth = (
                    2
                    * self.vol_stor[downstream_link_id][n]
                    / (
                        np.sum(width_of_upstream_links * length_of_upstream_links)
                        + width_of_downstream_link * length_of_downstream_link
                    )
                )

                #                print("alluvium depth = ",alluvium__depth)
                #                print("Volume stored at n = ",n,"=",self.vol_stor[downstream_link_id][n])
                #                print("Denomenator",np.sum(width_of_upstream_links * length_of_upstream_links) + width_of_downstream_link * length_of_downstream_link)
                #
                self._grid.at_node["topographic__elevation"][n] = (
                    self._grid.at_node["bedrock__elevation"][n] + alluvium__depth
                )

    def _calc_transport_wilcock_crowe(self):
        """Method to determine the transport time for each parcel in the active
        layer using a sediment transport equation.

        Note: could have options here (e.g. Wilcock and Crowe, FLVB, MPM, etc)
        """
        # parcel attribute arrays from DataRecord

        Darray = self._parcels.dataset.D[:, self._time_idx]
        Activearray = self._parcels.dataset.active_layer[:, self._time_idx].values
        Rhoarray = self._parcels.dataset.density.values
        Volarray = self._parcels.dataset.volume[:, self._time_idx].values
        Linkarray = self._parcels.dataset.element_id[
            :, self._time_idx
        ].values  # link that the parcel is currently in

        R = (Rhoarray - self.fluid_density) / self.fluid_density

        # parcel attribute arrays to populate below
        frac_sand_array = np.zeros(self._num_parcels)
        vol_act_array = np.zeros(self._num_parcels)
        Sarray = np.zeros(self._num_parcels)
        Harray = np.zeros(self._num_parcels)
        Larray = np.zeros(self._num_parcels)
        d_mean_active = np.zeros(self._num_parcels)
        d_mean_active.fill(np.nan)
        self.Ttimearray = np.zeros(self._num_parcels)
        # ^ Ttimearray is the time to move through the entire length of a link
        self.pvelocity = np.zeros(self._num_parcels)
        # ^ pvelocity is the parcel virtual velocity = link length / link travel time

        # Calculate bed statistics for all of the links
        vol_tot = self._parcels.calc_aggregate_value(
            np.sum, "volume", at="link", filter_array=self._find_now
        )
        vol_tot[np.isnan(vol_tot) == 1] = 0

        vol_act = self._parcels.calc_aggregate_value(
            np.sum, "volume", at="link", filter_array=self._active_parcel_records
        )
        vol_act[np.isnan(vol_act) == 1] = 0

        # find active sand.
        findactivesand = (
            self._parcels.dataset.D < 0.002
        ) * self._active_parcel_records  # since find active already sets all prior timesteps to False, we can use D for all timesteps here.

        if np.any(findactivesand):
            # print("there's active sand!")
            vol_act_sand = self._parcels.calc_aggregate_value(
                np.sum, "volume", at="link", filter_array=findactivesand
            )
            vol_act_sand[np.isnan(vol_act_sand) == True] = 0
        else:
            vol_act_sand = np.zeros(self._grid.number_of_links)

        frac_sand = np.zeros_like(vol_act)
        frac_sand[vol_act != 0] = vol_act_sand[vol_act != 0] / vol_act[vol_act != 0]
        frac_sand[np.isnan(frac_sand) == True] = 0

        # Calc attributes for each link, map to parcel arrays
        for i in range(self._grid.number_of_links):

            active_here = np.where(np.logical_and(Linkarray == i, Activearray == 1))[0]
            d_act_i = Darray[active_here]
            vol_act_i = Volarray[active_here]
            vol_act_tot_i = np.sum(vol_act_i)
            # ^ this behaves as expected. filterarray to create vol_tot above does not. --> FIXED?
            d_mean_active[Linkarray == i] = np.sum(d_act_i * vol_act_i) / (
                vol_act_tot_i
            )

            frac_sand_array[Linkarray == i] = frac_sand[i]
            vol_act_array[Linkarray == i] = vol_act[i]
            Sarray[Linkarray == i] = self._grid.at_link["channel_slope"][i]
            Harray[Linkarray == i] = self.flow_depth[self._time_idx, i]
            Larray[Linkarray == i] = self._grid.at_link["link_length"][i]

        Sarray = np.squeeze(Sarray)
        Harray = np.squeeze(Harray)
        Larray = np.squeeze(Larray)
        frac_sand_array = np.squeeze(frac_sand_array)

        # Wilcock and crowe claculate transport for all parcels (active and inactive)
        taursg = (
            self.fluid_density
            * R
            * self.g
            * d_mean_active
            * (0.021 + 0.015 * np.exp(-20.0 * frac_sand_array))
        )

        #        print("d_mean_active = ", d_mean_active)
        #        print("taursg = ", taursg)

        # frac_parcel should be the fraction of parcel volume in the active layer volume
        # frac_parcel = vol_act_array / Volarray
        # ^ This is not a fraction
        # Instead I think it should be this but CHECK CHECK
        frac_parcel = np.nan * np.zeros_like(Volarray)
        frac_parcel[vol_act_array != 0] = (
            Volarray[vol_act_array != 0] / vol_act_array[vol_act_array != 0]
        )
        # frac_parcel = Volarray / vol_act_array

        # print("frac_parcel = ", frac_parcel)

        b = 0.67 / (1 + np.exp(1.5 - Darray / d_mean_active))
        tau = self.fluid_density * self.g * Harray * Sarray
        taur = taursg * (Darray / d_mean_active) ** b

        tautaur = tau / taur
        tautaur_cplx = tautaur.astype(np.complex128)
        # ^ work around needed b/c np fails with non-integer powers of negative numbers
        W = 0.002 * np.power(tautaur_cplx.real, 7.5)
        W[tautaur >= 1.35] = 14 * np.power(
            (1 - (0.894 / np.sqrt(tautaur_cplx.real[tautaur >= 1.35]))), 4.5
        )
        W = W.real

        # compute parcel virtual velocity, m/s
        self.pvelocity[Activearray == 1] = (
            W[Activearray == 1]
            * (tau[Activearray == 1] ** (3 / 2))
            * frac_parcel[Activearray == 1]
            / (self.fluid_density ** (3 / 2))
            / self.g
            / R[Activearray == 1]
            / self.active_layer_thickness
        )

        # print("pvelocity = ", self.pvelocity)

        # Assign those things to the grid -- might be useful for plotting later...?
        self._grid.at_link["sediment_total_volume"] = vol_tot
        self._grid.at_link["sediment__active__volume"] = vol_act
        self._grid.at_link["sediment__active__sand_fraction"] = frac_sand

    def _move_parcel_downstream(self, dt):  # Jon
        """Method to update parcel location for each parcel in the active
        layer.
        """

        # we need to make sure we are pointing to the array rather than making copies
        current_link = self._parcels.dataset.element_id[
            :, self._time_idx
        ]  # same as Linkarray, this will be updated below
        location_in_link = self._parcels.dataset.location_in_link[
            :, self._time_idx
        ]  # updated below
        distance_to_travel_this_timestep = (
            self.pvelocity * dt
        )  # total distance traveled in dt at parcel virtual velocity
        # ^ movement in current and any DS links at this dt is at the same velocity as in the current link
        # ... perhaps modify in the future(?) or ensure this type of travel is kept to a minimum
        # ... or display warnings or create a log file when the parcel jumps far in the next DS link

        # print("distance traveled = ", distance_to_travel_this_timestep)

        #        if self._time_idx == 1:
        #            print("t", self._time_idx)

        for p in range(self._parcels.number_of_items):

            distance_to_exit_current_link = self._grid.at_link["link_length"][
                int(current_link[p])
            ] * (1 - location_in_link[p])

            # initial distance already within current link
            distance_within_current_link = self._grid.at_link["link_length"][
                int(current_link[p])
            ] * (location_in_link[p])

            running_travel_distance_in_dt = 0  # initialize to 0

            distance_left_to_travel = distance_to_travel_this_timestep[p]
            # if parcel in network at end of last timestep

            if self._parcels.dataset.element_id[p, self._time_idx] != _OUT_OF_NETWORK:
                # calc travel distances for all parcels on the network in this timestep

                # distance remaining before leaving current link

                while (
                    running_travel_distance_in_dt + distance_to_exit_current_link
                ) <= distance_to_travel_this_timestep[p]:
                    # distance_left_to_travel > 0:
                    # ^ loop through until you find the link the parcel will reside in after moving
                    # ... the total travel distance

                    # update running travel distance now that you know the parcel will move through the
                    # ... current link
                    running_travel_distance_in_dt = (
                        running_travel_distance_in_dt + distance_to_exit_current_link
                    )

                    # now in DS link so this is reset
                    distance_within_current_link = 0

                    # determine downstream link
                    downstream_link_id = self.fd.link_to_flow_receiving_node[
                        self.fd.downstream_node_at_link()[int(current_link[p])]
                    ]

                    # update current link to the next link DS
                    current_link[p] = downstream_link_id

                    if downstream_link_id == -1:  # parcel has exited the network
                        # (downstream_link_id == -1) and (distance_left_to_travel <= 0):  # parcel has exited the network
                        current_link[p] = _OUT_OF_NETWORK  # overwrite current link

                        # Keep parcel in data record but update its attributes so it is no longer accessed.
                        # Moving parcels into a separate exit array seems to be too computationally expensive.
                        # Probably worthwhile to update the following upon exit:
                        # parcels.dataset.element_id
                        # parcels.dataset.D
                        # parcels.dataset.volume
                        # and compute sub-dt time of exit

                        break  # break out of while loop

                    # ARRIVAL TIME in this link ("current_link") =
                    # (running_travel_distance_in_dt[p] / distance_to_travel_this_timestep[p]) * dt + "t" running time
                    # ^ DANGER DANGER ... if implemented make sure "t" running time + a fraction of dt
                    # ... correctly steps through time.

                    distance_to_exit_current_link = self._grid.at_link["link_length"][
                        int(current_link[p])
                    ]

                    distance_left_to_travel -= distance_to_exit_current_link

                # At this point, we have progressed to the link where the parcel will reside after dt
                distance_to_resting_in_link = (
                    distance_within_current_link  # zero if parcel in DS link
                    + distance_to_travel_this_timestep[p]
                    - running_travel_distance_in_dt  # zero if parcel in same link
                )

                # update location in current link
                if current_link[p] == _OUT_OF_NETWORK:
                    location_in_link[p] = np.nan

                else:
                    location_in_link[p] = (
                        distance_to_resting_in_link
                        / self._grid.at_link["link_length"][int(current_link[p])]
                    )

                # reduce D and volume due to abrasion
                vol = (self._parcels.dataset.volume[p, self._time_idx]) * (
                    np.exp(
                        distance_to_travel_this_timestep[p]
                        * (-self._parcels.dataset.abrasion_rate[p])
                    )
                )

                D = (self._parcels.dataset.D[p, self._time_idx]) * (
                    vol / self._parcels.dataset.volume[p, self._time_idx]
                ) ** (1 / 3)

                # update parcel attributes
                self._parcels.dataset.location_in_link[
                    p, self._time_idx
                ] = location_in_link[p]
                self._parcels.dataset.element_id[p, self._time_idx] = current_link[p]
                self._parcels.dataset.active_layer[p, self._time_idx] = 1
                # ^ reset to 1 (active) to be recomputed/determined at next timestep

                # Jon -- I suggest we do this after the fact when plotting to reduce model runtime:
                # calculate the x and y value of each parcel at this time (based on squiggly shape file)
                # could also create a function that calculates the x and y value for all parcels at all time
                # that is just called once at the end of running the model.

                # self._parcels.dataset["x"] = x_value
                # self._parcels.dataset["y"] = y_value

                self._parcels.dataset.D[p, self._time_idx] = D
                self._parcels.dataset.volume[p, self._time_idx] = vol

    # %%
    def run_one_step(self, dt):
        """stuff"""
        self._time += dt

        self._time_idx += 1
        self._create_new_parcel_time()

        if self._this_timesteps_parcels.any():
            self._partition_active_and_storage_layers()
            self._adjust_node_elevation()
            self._update_channel_slopes()  # I moved this down and commented out the second call to 'update channel slopes...'
            self._calc_transport_wilcock_crowe()
            self._move_parcel_downstream(dt)

        else:
            msg = "No more parcels on grid"
            raise RuntimeError(msg)
