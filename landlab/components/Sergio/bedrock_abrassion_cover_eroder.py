import numpy as np
from landlab import Component

from landlab.components import FlowDirectorSteepest
# from landlab.data_record import DataRecord
from landlab.grid.network import NetworkModelGrid


class BedRockAbrasionCoverEroder(Component):
    """
    BedRockAbrasionCoverEroder or BRACE is a landlab implementation of the
    MRSAA-c mode from Zhang's 2017 paper. BRACE models Saltation, Abrassion,
    and Alluvium cover on a river reach. It uses the scale of the macroroughness
    and sediment properties to model the change in the alluvium and erosion
    of the bed.

    Created by Sergio Villamarin
    """

    _name = "BedRockAbrasionCoverEroder"

    _unit_agnostic = False
    # all arguments are optional for now, yet to review
    _info = {
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "discharge": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m^3/s",
            "mapping": "node",
            "doc": "Unit width morphodynamically active water discharge.",
        },
        "channel_width": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Channel link average width",
        },
        "flood_intermittency": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Fraction of time when the river is morphodynamically active: former value used when bedrock morphodynamics is considered",
        },
        "sediment_grain_size": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "mm",
            "mapping": "node",
            "doc": "Sediment grain size on the node (single size sediment).",
        },
        "sed_capacity": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Sediment capacity of the link stored at the upstream node of each link",
        },
        "bedrock": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Bedrock elevation",
        },
        "macroroughness": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Thickness of macroroughness layer. See Zhang 2015 paper",
        },
        "mean_alluvium_thickness": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Mean alluvium thickness as described in L. Zhang 2015",
        },
        "fraction_alluvium_cover": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "%",
            "mapping": "node",
            "doc": "Fraction of alluvium cover of a link stored on its upstream node. Percentage of bed cover by alluvium protecting it from abrasion erosion as described in L. Zhang 2015",
        },
        "fraction_alluvium_avaliable": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "%",
            "mapping": "node",
            "doc": "Fraction of cover that is available for bedload transport as described in L. Zhang 2018",
        }

    }

    _cite_as = """@article{leMua2021DumbComponent,
      doi = {55.555/saywa555},
      url = {https://55.555/saywa555},
      year = {2021},
      publisher = {WeGetYourShitOut},
      volume = {1},
      number = {11},
      pages = {111},
      author = {Sergio Villamarin and Jane Gloriana Villanueva},
      title = {Components on landlab: Learn how the fuck to use them},
      journal = {The "I hope to get better" Journal of science}
    }"""

    def __init__(self, grid, flow_director, corrected=True, **kwargs):
        """
        Creates a BedRockAbrasionCoverEroder object. If field values
        are provided as parameters in this function with the documented
        field name, they will be created if the did not exist before.
        It won't replace preset values in landlab fields.

        If needed values are not provided or preset then they will be
        created using values from main ref paper.

        ## Parameters

        ----

        `grid`: `RasterModelGrid`
            (required) A grid.

        `flow_director`: :py:class:`~landlab.components.FlowDirectorSteepest`
            (required) A landlab flow director. Currently, must be
            :py:class:`~landlab.components.FlowDirectorSteepest`.

        `clobber`: `bool`
            currently not implemented. Defaults to False

        `discharge`: `float` or `grid`
            in case it is not provided as a landlab field flow it can be
            provided here as a constant or the grid values. Discharge units
            are in m^3/s. Defaults to 300 m^3/s if not provided.

        `flood_intermittency`: `float` or `grid`
            in case it is not provided as a landlab field flow it can be
            provided here as a constant or the grid values. Units are in
            percentages. Defaults to 0.05 = 5%

        `channel_width`: `float` or `grid`
            in case it is not provided as a landlab field flow it can be
            provided here as a constant or the grid values. Units are in
            m (meters). Defaults to 100 m

        `sediment_grain_size`: `float` or `grid`
            while it is currently a landlab field there's no calculation
            that keeps track of different sediment grain sizes as they
            move along the network. This could be done separately and
            updated here if needed. Units are in m. Defaults to 0.02 m

        `macroroughness`: `float` or `grid`
            scale of the macroroughness of the bed over which alluvium 
            cover is relevant on the erosional process.
            in case it is not provided as a landlab field flow it can be
            provided here as a constant or the grid values. Units are in
            m (meters). Defaults to 1 m

        `mean_alluvium_thickness`: `float` or `grid`
            initial depth of the alluvium cover of the bedrock.
            in case it is not provided as a landlab field flow it can be
            provided here as a constant or the grid values. Units are in
            m (meters). Defaults to 0.5 m

        `Cz`: `float` or `grid`
            Dimentionless Chezy resistance coeff calculated is
            as Cz=U/sqrt(tau/rho) as in Parker & Wong 2006.
            Defaults to 10.

        `beta`: `float`
            Related to sediment abrasion as beta = 3*alpha. Units of 1/m.
            Beta. Sediment Wear coefficient. See Sklar and Dietrich 2004.
            Defaults to 0.05*0.001

        `shear_coeff`: `float`
            Sediment transport capacity coefficient (eq 5c). Defaults to 4

        `shear_exp`: `float`
            Sediment transport capacity exponent (eq 5c). Defaults to 1.5

        `crit_shear`: `float`
            Dimentionless critical shear stress for incipient motion of
            sediment. Defaults to 0.0495

        `porosity`: `float`
            Bedload sediment porosity. Defaults to 0.35

        `spec_grav`: `float`
            Specific gravity of sediment. Defaults to 1.65

        `p0`: `float`
            Lower percentage of bed cover (deep pockets). Defaults to 0.05 = 5%

        `p1`: `float`
            Higher percentage for effective bed cover. Defaults to 0.95 = 95%

        `au`: `float`
            Alluvium calculation finite difference parameter, can be adjusted
            get smoother alluvium results. Defaults to ....

        `su`: `float`
            Slope calculation finite difference parameter, can be adjusted get
            smoother instabilities in slope calculations. Defaults to ....

        ## Examples

        ----

        """

        # add the extra required parameters to the grid if not provided
        BedRockAbrasionCoverEroder._preset_fields(grid, False, **kwargs)
        super().__init__(grid)

        self.corrected = corrected
        self.G = 9.80665  # gravity
        # smoothing parameters for the alluvium and slope calculations
        self.au = kwargs["au"] if "au" in kwargs else 0.9
        self.su = kwargs["su"] if "au" in kwargs else 0.1

        # other parameters
        self.Cz = kwargs["Cz"] if "Cz" in kwargs else 10
        self.wear_coefficient = kwargs["beta"] if "beta" in kwargs else 0.05 * 0.001

        self.ssalpha = kwargs["shear_coeff"] if "shear_coeff" in kwargs else 4
        self.ssna = kwargs["shear_exp"] if "shear_exp" in kwargs else 1.5
        self.sstau_star_c = kwargs["crit_shear"] if "crit_shear" in kwargs else 0.0495

        # another combination of parameters for equation 5c
        # self.ssalpha = 5.7
        # self.ssna = 1.5
        # self.sstau_star_c = 0.03

        self.porosity = kwargs["porosity"] if "porosity" in kwargs else 0.35
        self.spec_grav = kwargs["spec_grav"] if "spec_grav" in kwargs else 1.65

        # deep pockets cover and maximum effective cover
        self.p0 = kwargs["p0"] if "p0" in kwargs else 0.05
        self.p1 = kwargs["p1"] if "p1" in kwargs else 0.95

        # it must be a NetworkModelGrid
        if not isinstance(grid, NetworkModelGrid):
            msg = "NetworkSedimentTransporter: grid must be NetworkModelGrid"
            raise ValueError(msg)

        # flow routing should happen outside for now
        # supported flow directors
        if not isinstance(flow_director, FlowDirectorSteepest):
            msg = (
                "NetworkSedimentTransporter: flow_director must be "
                "FlowDirectorSteepest."
            )
            raise ValueError(msg)

        # topographic elevation is a copy of the
        # bedrock topography by default
        self._topo = self._grid.at_node["topographic__elevation"]
        if not self._grid.has_field("bedrock", at="node"):
            self._grid.add_field("bedrock", self._topo, at="node", copy=True)
        # and flow__sender_node to nodes
        if not self._grid.has_field("flow__sender_node", at="node"):
            self._add_flow_sender_node()
        # add upstream and downstream field to each link
        self._dnode = self._grid.at_node["flow__receiver_node"]
        self._unode = self._grid.at_node["flow__sender_node"]
        # adds outlets and sources
        self.outlets = self._find_outlets()
        self.sources = self._find_sources()
        # finds the joints for special calculations
        self.joints, self.ujoints = self._find_joints()
        # add output fields
        self.initialize_output_fields()
        # update channel slopes
        self.downstream_distance = self._calculate_reach_length()
        self.dx, self._j_dx = self._calculate_dx()
        self.slope = self._update_channel_slopes()
        # set conditions at the futher dowstream nodes
        # currently not handled as only the open boundary case is dealt with

    def run_one_step(self, dt, **kwargs):
        """
        It updates the mean alluvium thickness and the bedrock elevation based
        on the model by Zhang et al (2015,2018).

        Parameters for the boundary conditions can be provided such as:

        ## Parameters
        ----------
        `outlet`: `string`
            can be set as `"copy_downstream"`, `"open"`, or `"set_value"`. It
            defaults to `"open"` which limits the mean alluvium thickness to
            1 L (macroroughness unit). This means the alluvium fills up to
            that point and any excess goes out of the system.

        `source`: `string`
            can be set as `"copy_downstream"`, `"open"`, or `"set_value"`. It
            defaults to `"set_value"` which sets the sed_capacity at the upstream
            ends to the `q_in` parameter.

        `q_in`: `float` or `grid`
            values of sediment flux at the sources if `source` is set to `"set_value"`

        `q_out`: `float` or `grid`
            values of sediment flux at the outlets if `outlet` is set to `"set_value"`

        `limit_outlet`: `bool`
            if `True` if limits erosion and the alluvium thickness at the outlet using
            the keyword arguments `baselevel` and 1 L (macroroughness unit) respectively.
            Defaults to `True`

        `baselevel`: `float`
            lowest value the bedrock can go due to erosion at the outlets
        """
        # calculate parameters needed in the pde
        tau_star_crit = self.sstau_star_c
        tau_star = self._calculate_shear_star()
        self._calculate_fraction_alluvium_cover(self.p0, self.p1)
        self._calculate_corrected_fraction(self.p0)
        self._calculate_sed_capacity(tau_star, tau_star_crit)
        # boundary conditions
        self._boundary_conditions_precalc(**kwargs)
        # bed erosion (applied to the upstream node)
        self._bed_erosion(dt)
        # mean alluvium thickness change
        self._mean_alluvium_change(dt)
        # update slopes
        self._boundary_conditions_postcalc(**kwargs)
        self.slope = self._update_channel_slopes()

    def _find_joints(self):
        """
        Returns the ids of nodes at the joint locations and their
        upstream nodes.
        """
        nodes = np.arange(0, self._grid["node"].size, 1)
        mask = np.full_like(nodes, True, dtype="bool")
        # remove outlet nodes
        mask[self.outlets] = False
        candidates = nodes[mask]
        # Node maps to itself injectively if we go down and then up
        mapDownAndUp = self._unode[self._dnode[candidates]]
        # unless the downstream node is a joint
        passed = (mapDownAndUp - candidates) != 0
        # in that case the candidate maps down to a joint
        joints = self._dnode[candidates[passed]]

        ujoints = {}
        for joint in joints:
            # if node down is the joint
            ups = (self._dnode == joint)
            # then save them all in a dict
            ujoints[joint] = nodes[ups]

        return joints, ujoints

    def _add_flow_sender_node(self):
        """
        Adds the node field "flow__sender_node". This field has the
        upstream node based on the flow direction and IS ONLY MEANT
        to be used for finding the upstream source nodes and the
        previous to last node downstream. At junctions it returns
        only a single id ignoring one of the two tributaries.
        """
        node = np.arange(0, self._grid["node"].size, 1)
        sender = np.arange(0, self._grid["node"].size, 1)
        receiver = np.copy(self._grid["node"]["flow__receiver_node"])
        non_idem = (receiver != node)
        sender[receiver[non_idem]] = node[non_idem]
        self._grid.add_field(
            "flow__sender_node",
            sender,
            at="node"
        )

    def _find_outlets(self):
        """
        Finds the outlets node's IDs of a river network created with
        the flow director steepest component. Its dependency to the
        flow director are the flow directions from "flow__receiver_node"
        but these can be provided independently if needed.

        Warning! Landlab flags outlets as sinks and therefore are
        undistinguishable.
        """
        nodes = np.arange(0, self._grid["node"].size, 1)
        flow2self = (self._grid["node"]["flow__receiver_node"] == nodes)
        # not_sink = np.logical_not(self._grid["node"]["flow__sink_flag"])
        return nodes[flow2self]  # & not_sink]

    def _find_sources(self):
        """
        Finds the sources node's IDs of a river network created with
        the flow director steepest component. Its dependency to the
        flow director are the flags for sinks, and the flow directions.
        """
        nodes = np.arange(0, self._grid["node"].size, 1)
        flow_from_none = (self._grid["node"]["flow__sender_node"] == nodes)
        return nodes[flow_from_none]

    def _get_channels(self):
        """Returns a dictionary of key:value pairs where the keys are the source
          nodes and values are correponding list of nodes ordered from source to
          outlet"""
        channels = {}
        for source in self.sources:
            node = source
            channel = []
            channel.append(node)
            while node not in self.outlets:
                node = self._dnode[node]
                channel.append(node)
            channels[source] = channel
        return channels

    def _update_channel_slopes(self):
        """
        Returns the channel slopes using the mean alluvium cover and the
        bedrock.

        At joints it calculates a weighted average slope using the
        discharge from the tributaries as weights.

        Additional parameter used helps smooth peaks caused by the numerical
        instability of the central difference when calculating the slopes.
        The order of the approximation reduces in exchange for less
        instability and overall more realistic results.
        """

        c = self.su
        y = self._grid.at_node["bedrock"] + self._grid.at_node["mean_alluvium_thickness"]

        S = -(- c * y[self._unode]
              + (2 * c - 1) * y
              + (1 - c) * y[self._dnode]) / self.dx

        # joint upstream weighted slope
        for joint in self.joints:
            weight = (self._grid.at_node["discharge"][self.ujoints[joint]]
                      / np.sum(self._grid.at_node["discharge"][self.ujoints[joint]]))
            dnode = self._dnode[joint]
            Sjoint = -(- c * y[self.ujoints[joint]]
                       + (2 * c - 1) * y[joint]
                       + (1 - c) * y[dnode]) / self._j_dx[joint]
            Sjoint[Sjoint < 0] = 0  # as implemented on the original code model
            S[joint] = np.sum(weight * Sjoint)

        # edge cases for boundary calculations
        if (c == 0) or (c == 1):
            pass
        else:
            S[self.sources] = S[self.sources] / (1 - c)
            S[self.outlets] = S[self.outlets] / (c)
        S[S < 0] = 0  # as implemented on the original code model
        return S

    def _calculate_reach_length(self):
        """
        Returns the reach length downstream at every node.
        It is currently used in the calculations for dx in every
        diff equation and slope calculation (also the slope as
        S = dz/dx). This doesn't mean that dx is simply the
        downstream distance, even more so at junctions. See
        slope and alluvium change functions
        """
        dis_down = (np.square(self._grid.x_of_node - self._grid.x_of_node[self._dnode])
                    + np.square(self._grid.y_of_node - self._grid.y_of_node[self._dnode]))
        dis_down = np.sqrt(dis_down)
        return dis_down

    def _calculate_dx(self):
        """
        Uses downstream distances to calculate the appropiate dx
        used in the differential equations as the mean of upstream
        and downstream distance for every node. It also returns the
        a list of dx for each joint upstream node.

        dx at joint = weighted average (links upstream + link downstream)
        """
        dx = (self.downstream_distance[self._unode]
              + self.downstream_distance) / 2
        # fix sources and outlets
        dx[self.sources] = self.downstream_distance[self.sources]
        dx[self.outlets] = self.downstream_distance[self._unode[self.outlets]]
        # do dx joint calculations separately
        j_dx = {}
        for joint in self.joints:
            j_dx[joint] = (self.downstream_distance[self.ujoints[joint]]
                           + self.downstream_distance[joint]) / 2
        return dx, j_dx

    def _calculate_flow_depths(self):
        """
        Re-calculates the flow depth based on the hydraulic relation
        H = [Q^2 / g*S*B^2*Cz^2]^(1/3)
        """
        H = ((self._grid.at_node["discharge"] * self._grid.at_node["discharge"])
             / (self.G * self.slope
                * np.square(self._grid.at_node["channel_width"])
                * self.Cz * self.Cz)) ** (1 / 3)
        return H

    def _calculate_shear_star(self, method="Parker"):
        """
        Re-calculates the dimentionless bed shear stress at normal flow
        conditions based on the hydraulic relations derived by Parker
        tau_star = [Q^2 / g*B^2*Cz^2]^(1/3) * S^(2/3) / (R * D)
        """
        tau_star = (np.power(
            np.square(self._grid.at_node["discharge"] / self._grid.at_node["channel_width"])
            / (self.Cz * self.Cz)
            / self.G,
            1 / 3)
            * np.power(self.slope, 2 / 3)
            / self.spec_grav
            / self._grid.at_node["sediment_grain_size"])
        return tau_star

    def _calculate_fraction_alluvium_cover(self, p0=0.05, p1=0.95):
        """
        Calculates p, the alluvium cover from Zhang 2015. It uses a
        linear alluvium cover function. line between 0.05 and 0.95 from
        minimum (deep pockets) cover to maximum (effective) cover of the bed
        """

        if np.any(self._grid.at_node["mean_alluvium_thickness"] < 0):
            raise ArithmeticError("A previous calculation resulted in "
                                  + "negative mean_alluvium_thickness")
        chi = (
            self._grid.at_node["mean_alluvium_thickness"]
            / self._grid.at_node["macroroughness"])
        cover = p0 + chi * (p1 - p0)
        full_cover = (1 - p0) / (p1 - p0)
        cover = np.where(chi > full_cover, np.ones_like(cover), cover)
        cover = np.where(chi < 0, np.zeros_like(cover), cover)
        self._grid.at_node["fraction_alluvium_cover"] = cover

    def _calculate_corrected_fraction(self, p0=0.05):
        """
        Calculates pa, the corrected alluvium avaliable for transport from Zhang 2018.
        it uses yet another linear function to remove non-zero transport under lack
        of sediment conditions (in deep pockets).
        """
        p = np.copy(self._grid.at_node["fraction_alluvium_cover"])
        self._grid.at_node["fraction_alluvium_avaliable"] = (p - p0) / (1 - p0)
        # make sure no negative alluvium is avaliable for transport
        zeros = (self._grid.at_node["fraction_alluvium_avaliable"] < 0)
        self._grid.at_node["fraction_alluvium_avaliable"][zeros] = 0

    def _calculate_sed_capacity(self, tau_star, tau_star_crit=0.0495):
        """
        Calculates sediment flow capacity for a node and stores it at
        the upstream node field "sed_capacity" using Parker and Wong
        formulation q_ac = 4 sqrt( RgD ) D (tau* - tau_c*)^(3/2)
        """
        excess_shear = tau_star - tau_star_crit
        zeromask = np.ones_like(excess_shear)
        zeromask[excess_shear < 0] = 0  # as in the original code
        excess_shear = excess_shear * zeromask
        self._grid.at_node["sed_capacity"] = (
            np.power(self.spec_grav * self.G * self._grid.at_node["sediment_grain_size"], 0.5)
            * self._grid.at_node["sediment_grain_size"]
            * (self.ssalpha
               * np.power(excess_shear, self.ssna)))

    def _bed_erosion(self, dt):
        """
        Applies bed erosion discretizing the differential equation
        of the bed in the upstream nodes (representing the link downstream).
        Must be called AFTER updating sediment capacity,
        fraction of alluvium cover and fraction of avaliable alluvium.
        """
        if self.corrected:
            pa = self._grid.at_node["fraction_alluvium_avaliable"]
        else:
            pa = self._grid.at_node["fraction_alluvium_cover"]

        erosion = (self._grid.at_node["flood_intermittency"]
                   * self.wear_coefficient
                   * self._grid.at_node["sed_capacity"]
                   * pa * (1 - pa) * dt)
        erode = (erosion >= 0)
        self._grid.at_node["bedrock"][erode] = self._grid.at_node["bedrock"][erode] - erosion[erode]

    def _mean_alluvium_change(self, dt):
        """
        Adds/Removes the difference in mean alluvium thickness by
        discretizing the differential equation of the alluvium.
        Must be called after updating sediment capacity,
        fraction of alluvium cover and fraction of avaliable alluvium.
        """

        if self.corrected:
            p = self._grid.at_node["fraction_alluvium_avaliable"]
        else:
            p = self._grid.at_node["fraction_alluvium_cover"]
        q = self._grid.at_node["sed_capacity"]
        pq = p * q
        c = self.au

        dpq = (- c * pq[self._unode]
               + (2 * c - 1) * pq
               + (1 - c) * pq[self._dnode])
        # joint flux calculations
        up_joint_pq = [np.sum(pq[self.ujoints[joint]]) for joint in self.joints]
        up_joint_pq = np.array(up_joint_pq)
        dpq[self.joints] = (- c * up_joint_pq
                            + (2 * c - 1) * pq[self.joints]
                            + (1 - c) * pq[self._dnode[self.joints]])
        if (c == 1) or (c == 0):
            pass
        else:
            dpq[self.sources] = dpq[self.sources] / (1 - c)
            dpq[self.outlets] = dpq[self.outlets] / (c)

        cover_dif = (-self._grid.at_node["flood_intermittency"]
                     * dpq / self.dx
                     * dt / (1 - self.porosity)
                     / self._grid.at_node["fraction_alluvium_cover"])

        self._grid.at_node["mean_alluvium_thickness"] = self._grid.at_node["mean_alluvium_thickness"] + cover_dif
        self._grid.at_node["mean_alluvium_thickness"][self._grid.at_node["mean_alluvium_thickness"] < 0] = 0

    def _boundary_conditions_precalc(self, **kwargs):
        """
        sets boundary conditions for incoming flux and outgoing nodes.
        It currently defaults to an open boundary downstream and a
        maximum alluvium of 1 L (1 macro roughtness unit for full cover).

        The options for outlet and source are `"copy_downstream"`, `"open"`,
        `"set_value"`. It defaults outlet to `"open"` and source to `"set_value"`
        so the parameter q_in should be provided.

        `"set_values"` Currently sets the incoming and outgoing flux using the
        `q_in` and `q_out` parameters as a float or a vector of the same size as
        self.sources. Sets all sources/outlets to the same value if a float is
        provided.

        the self.sources property is an int np list of the nodes that are
        sources and similarly the self.outlets property for outlets.

        out_topo is the elevation outlet boundary condition. Since
        The component is not yet decided to be a single tree or a
        forest then we shall assume there's a single outlet and so
        self.outlets is a np array with a single int but there's no
        real limitation in the code. needs testing.

        Warning!
        it assumes the previous to last downstream node is unique (no
        junctions meet right at the outlet)
        """

        flux_in = np.zeros_like(self.sources, dtype=np.float64)
        flux_out = np.zeros_like(self.outlets, dtype=np.float64)
        source = kwargs["source"] if "source" in kwargs else "set_value"
        outlet = kwargs["outlet"] if "outlet" in kwargs else "open"
        # outlets
        if outlet == "open":
            pass

        if outlet == "copy_downstream":
            # Assuming the previous to last node is not ambiguous
            prev_node = self._grid.at_node["flow__sender_node"][self.outlets]
            # set outlets to mirror previous to last node
            fields = {"sed_capacity", "fraction_alluvium_cover", "fraction_alluvium_avaliable"}
            for field in fields:
                self._grid.at_node[field][self.outlets] = self._grid.at_node[field][prev_node]
            self._grid.at_node["mean_alluvium_thickness"][self.outlets] = self._grid.at_node["mean_alluvium_thickness"][prev_node]

        if outlet == "set_value":
            flux_out[:] = kwargs["q_out"]
            if self.corrected:
                p = self._grid.at_node["fraction_alluvium_avaliable"][self.sources]
            else:
                p = self._grid.at_node["fraction_alluvium_cover"][self.sources]

            self._grid.at_node["sed_capacity"][self.outlets] = flux_out

        # sources
        if source == "open":
            pass

        if source == "set_value":
            if self.corrected:
                p = self._grid.at_node["fraction_alluvium_avaliable"][self.sources]
            else:
                p = self._grid.at_node["fraction_alluvium_cover"][self.sources]
            flux_in[:] = kwargs["q_in"]
            self._grid.at_node["sed_capacity"][self.sources] = flux_in / p

    def _boundary_conditions_postcalc(self, **kwargs):
        """
        It handles the boundary conditions after the change in alluvium
        and the bed have already been dealt with.

        The options for outlet are "copy_downstream", "open", "set_value".
        It defaults to "open". It defaults outlet to "open" and source to
        "open"

        limit_outlet parameter decides whenever to bound the outlet by
        1L macroroughness unit above and by the baselevel parameter below.

        If outlet/source are "set_value" then this function is only
        aesthetic to match the elevation at the source and not let
        the outlet go below the given baselevel. In this case the slope
        at the outlet/source is not relevant in the calculations.

        (not implemented) The option "set_values" has several kwargs in case
        one wants to provide how the elevation at the downstream end
        should behave. Given that the user might do this in landlab
        fields directly and that seems like the intended way to do
        it in landlab I might never implement it.
        """

        source = kwargs["source"] if "source" in kwargs else "set_value"
        outlet = kwargs["outlet"] if "outlet" in kwargs else "open"
        limit_outlet = kwargs["limit_outlet"] if "limit_outlet" in kwargs else True
        baselevel = kwargs["baselevel"] if "baselevel" in kwargs else 0

        # outlets
        if outlet == "open":
            pass

        if outlet == "copy_downstream":
            # Assuming the previous to last node is not ambiguous
            prev_node = self._grid.at_node["flow__sender_node"][self.outlets]
            # set outlets to mirror previous to last node
            fields = {"mean_alluvium_thickness", "bedrock"}
            for field in fields:
                self._grid.at_node[field][self.outlets] = self._grid.at_node[field][prev_node]

        if outlet == "set_value":
            # set_value logic can now be done by changing directly field values
            # on every run_one_step
            pass

        if limit_outlet:
            # Prevent the allivium to go over 1 L
            over_alluvium = (self._grid.at_node["mean_alluvium_thickness"][self.outlets]
                             > self._grid.at_node["macroroughness"][self.outlets])
            out_alluvium = np.where(
                over_alluvium,
                self._grid.at_node["macroroughness"][self.outlets],
                self._grid.at_node["mean_alluvium_thickness"][self.outlets])
            self._grid.at_node["mean_alluvium_thickness"][self.outlets] = out_alluvium

            # Prevents the bedrock from going below baselevel
            under_baselevel = (self._grid.at_node["bedrock"][self.outlets] < baselevel)
            out_bedrock = np.where(
                under_baselevel,
                baselevel * np.ones_like(self._grid.at_node["bedrock"][self.outlets]),
                self._grid.at_node["bedrock"][self.outlets])
            self._grid.at_node["bedrock"][self.outlets] = out_bedrock

        # sources
        if source == "open":
            pass

        if source == "set_value":
            # if kwargs are provided then those should be used to set the
            # elevations and alluvium cover?? currently does nothing
            # copy_dowstream behaviour (not implemented yet)
            pass

    @staticmethod
    def sedimentograph(time, dt, Tc, rh=0.25, qm=0.000834, rqh=1, random=False, **kwargs):
        """
        Creates a sedimentograph to use as the feed on sources for the
        network.

        Parameters
        ----------
        time: float or np.array
            (required) The total time in seconds or an array of times with
            step dt.
        dt: float
            (required) time step in seconds.
        Tc: float
            (required) Time of a cycle (period) in seconds.
        rh: float
            fraction of time at high rate in a cycle in seconds. Defaults
            to 1/4
        qm: float
            Mean sediment feed rate over a cycle. Width averaged in m^2/s.
            Defaults to 0.000834
        rqh: float
            proportion of high feed sediment rate to mean feed sediment rate.
            Defaults to 1 (qh = qm)
        random: bool
            whenever to use a random sedimentograph or not. Defaults to False
        random_seed: int
            if provided it is the random seed for the sedimentograph
        """
        if (rh < 0) or (rh >= 1):
            raise ValueError("high feed ratio rh must be between"
                             + "0 (inclusive) and 1 (exclusive)")
        rl = 1 - rh  # percentage of time at low feed
        if rqh < 1:
            raise ValueError("rqh high rate sediment feed"
                             + "can't be smaller than one")
        if rqh > (1 / rh):
            raise ValueError("the high feed is too high for"
                             + "making the low feed negative")
        rql = (1 - rqh * rh) / (1 - rh)  # proportion of low feed ql/qm
        T = 0
        if time is float:
            T = np.arange(0, time, dt)
        else:
            T = time  # assume already a numpy time array

        sedgraph = np.zeros_like(T)
        n_period = int(Tc / dt)  # n indices per cycle
        n_low = int(n_period * rl)  # n indices at low feed per cycle
        n_high = n_period - n_low  # n indices at high feed per cycle
        if not random:
            qh = rqh * qm  # high feed rate
            ql = rql * qm  # low feed rate
            # always start at high feed
            sedgraph[:] = qh
            for i in range(len(T)):
                if (i % n_period) > n_high:
                    sedgraph[i] = ql
        else:  # uniform random distribution of rqh in [6,12] with 0.5 steps
            if "random_seed" in kwargs:
                rng = np.random.default_rng(kwargs["random_seed"])
            else:
                rng = np.random.default_rng()
            range_random = np.arange(6, 12.5, 0.5)
            for i in range(len(T)):
                if i % n_period == 0:
                    rqh = rng.choice(range_random)
                    rql = (1 - rqh * rh) / (1 - rh)
                    qh = rqh * qm  # high feed rate
                    ql = rql * qm  # low feed rate
                # always start at high feed
                if (i % n_period) < n_high:
                    sedgraph[i] = qh
                else:
                    sedgraph[i] = ql
        sed_data = {}
        sed_data["t"] = T
        sed_data["sedgraph"] = sedgraph
        sed_data["data"] = {"total_time": Tc, "rh": rh, "qm": qm, "rqh": rqh}
        return sed_data

    @staticmethod
    def _preset_fields(ngrid, all_ones=False, **kwargs):
        """
        presets all the required fields of the grid needed for an instance
        of componentcita. Will not add/replace fields already in the grid.
        Only valid kwargs will be addded.

        If all_ones is True it sets all such parameters
        to 1, otherwise it uses values from main ref of these parameters.
        For more info on the parameters set see non optional inputs
        of this component.

        tests are based on these predefined values, don't change them.

        If values are provided then use the kwargs to fill in fields.

        Used by default on __init__
        """
        # initial conditions if provided as parameters
        input_fields = {"discharge": 300,
                        "flood_intermittency": 0.05,
                        "channel_width": 100,
                        "sediment_grain_size": 0.02,
                        "sed_capacity": 0,
                        "macroroughness": 1,
                        "mean_alluvium_thickness": 0.5
                        }
        for field in input_fields:
            # if not currently a field then added it
            if not ngrid.has_field(field, at="node"):
                ngrid.add_zeros(field, at="node")
                if not (field in kwargs):
                    # if not provided then use default values
                    print(f"adding {field}")
                    ngrid.at_node[field][:] = input_fields[field]
            # if provided as parameters then replace anything
            if field in kwargs:
                if all_ones:
                    ngrid.at_node[field][:] = 1
                else:
                    ngrid.at_node[field][:] = kwargs[field]

    @staticmethod
    def _preset_network(which_network=0):
        """
        returns a network grid and the associated flow director from a list of
        predefined networks for use on small examples and tests. Currently only 1.
        """
        match which_network:
            case 0:
                y_of_node = (1, 1, 1, 1)
                x_of_node = (1, 2, 3, 4)
                nodes_at_link = ((0, 1), (1, 2), (2, 3))

                ngrid = NetworkModelGrid((y_of_node, x_of_node), nodes_at_link)
                scale = 0.4  # Test depend on this number !!!
                topo = np.array([4, 3, 2, 1]) * scale
                ngrid.add_field("topographic__elevation", topo)
                flow_director = FlowDirectorSteepest(ngrid)
                flow_director.run_one_step()
                return ngrid, flow_director

    # deprecated
    def _link_add_upstream_downstream_nodes(self):
        """
        Deprecated after overhaul

        It adds the fields "upstream_node" and "downstream_node" to links.
        The fields have the id of the upstream and downstream node from
        the link based on the Network model node ids. Array index is in
        correspondance with links index for link fields.
        """
        self._grid.nodes_at_link
        # making a copy instead of a reference.... I'm unsure yet if I should
        tail_nodes = np.copy(self._grid.nodes_at_link[:, 0])
        head_nodes = np.copy(self._grid.nodes_at_link[:, 1])
        upstream_nodes = np.where(
            self._grid["link"]["flow__link_direction"] == 1,
            tail_nodes,
            head_nodes)
        downstream_nodes = np.where(
            self._grid["link"]["flow__link_direction"] == -1,
            tail_nodes,
            head_nodes)
        self._grid.add_field(
            "upstream_node",
            upstream_nodes,
            at="node"
        )
        self._grid.add_field(
            "downstream_node",
            downstream_nodes,
            at="node"
        )
