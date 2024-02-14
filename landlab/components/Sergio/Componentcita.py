import numpy as np
import copy
from landlab import Component

from landlab.components import FlowDirectorSteepest
# from landlab.data_record import DataRecord
from landlab.grid.network import NetworkModelGrid


class Componentcita(Component):
    """
    GrainSizeBedRockInciser component place holder as it evolves into a big
    code mess. GSBRI is supposed to model the abrasion of bedrock from the
    ammount and grain size of particles in the stream based on the work from
    Zhang 2017.

    Created by Sergio V
    """

    _name = "GrainSizeBedRockInciser"

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
        "flood_intermittency": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Fraction of time when the river is morphodynamically active: former value used when bedrock morphodynamics is considered",
        },
        "channel_width": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Channel link average width",
        },
        "sediment_grain_size": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "mm",
            "mapping": "node",
            "doc": "Sediment grain size on the node (single size sediment).",
        },
        "specific_gravity": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "-",
            "mapping": "node",
            "doc": "Submerged specific gravity of sediment.",
        },
        "dimentionless_Chezy": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "-",
            "mapping": "node",
            "doc": "Dimentionless Chezy C coefficient calculated as Cz=U/sqrt(tau/rho).",
        },
        "sediment_porosity": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "-",
            "mapping": "node",
            "doc": "Porosity of the alluvium.",
        },
        "macroroughness": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Thickness of macroroughness layer. See Zhang paper",
        },
        "reach_length": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "River channel (link upstream + link downstream) lenght",
        },
        "wear_coefficient": {
            "dtype": float,
            "intent": "inout",
            "optional": True,
            "units": "m",
            "mapping": "node",
            "doc": "Beta. Wear coefficient. See Sklar and Dietrich 2004",
        },
        "abrasion_coefficient": {
            "dtype": float,
            "intent": "inout",
            "optional": True,
            "units": "m",
            "mapping": "node",
            "doc": "Alpha. Abrasion coefficient defined by Sternberg's law is. See Sklar and Dietrich 2004",
        },
        "uplift_rate": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "m/kyr",
            "mapping": "node",
            "doc": "local uplift rate. (set instead on the component parameters for constant uplift)",
        },
        "sedimentograph_info": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "m",
            "mapping": "node",
            "doc": "This is really used more as 4/2 parameters in the model, recheck when the modelling time comes. Mean bedload feed rate averaged over sedimentograph",
        },
        "sed_capacity": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Sediment capacity of the link stored at the upstream node of each link",
        },
        "channel_slope": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Channel slopes from bedrock elevation",
        },
        "mean_alluvium_thickness": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Mean alluvium thickness as described in L. Zhang 2015",
        },
        "bedrock": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Bedrock elevation",
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
        Creates a Componencita object. If field values are provided as
        parameters in this function with the documented field name,
        they will be created if the did not exist before. It won't
        replace preset values in landlab fields.

        If needed values are not provided or preset then they will be
        created using values from main ref paper.

        Parameters
        ----------
        grid: RasterModelGrid
            (required) A grid.
        flow_director: :py:class:`~landlab.components.FlowDirectorSteepest`
            (required) A landlab flow director. Currently, must be
            :py:class:`~landlab.components.FlowDirectorSteepest`.
        clobber: bool
            currently not implemented. Defaults to False
        discharge: float
            if not provided as a landlab field flow discharge in m^3/s.
            Defaults to 300 m^3/s constant along the network.
        Cz: float
            Dimentionless Chezy resistance coeff. Defaults to 10
        beta: float
            Related to sediment abrasion as beta = 3*alpha. Units of 1/m.
            Defaults to 0.05*0.001
        shear_coeff: float
            Sediment transport capacity coefficient (eq 5c). Defaults to 4
        shear_exp: float
            Sediment transport capacity exponent (eq 5c). Defaults to 1.5
        crit_shear: float
            Dimentionless critical shear stress for incipient motion of
            sediment. Defaults to 0.0495
        porosity: float
            Bedload sediment porosity. Defaults to 0.35
        spec_grav: float
            Specific gravity of sediment. Defaults to 1.65
        k_viscosity: float
            Kinematic viscosity of water. Defaults to 10**-6, the kinematic
            viscosity of water (at 20C)
        p0: float
            Lower percentage of bed cover (deep pockets). Defaults to 0.05 = 5%
        p1: float
            Higher percentage for effective bed cover. Defaults to 0.95 = 95%
        au: float
            Alluvium calculation finite difference parameter, can be adjusted
            get smoother alluvium results. Defaults to ....
        su: float
            Slope calculation finite difference parameter, can be adjusted get
            smoother instabilities in slope calculations. Defaults to ....
        Examples
        --------

        """
        super().__init__(grid)

        self.corrected = corrected
        self.G = 9.80665  # gravity
        # smoothing parameters for the alluvium and slope calculations
        self.au = kwargs["au"] if "au" in kwargs else 0.9
        self.su = kwargs["su"] if "au" in kwargs else 0.1
        # add the extra parameters to the grid if not provided
        Componentcita._preset_fields(self._grid, False, **kwargs)
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
        self.v = kwargs["k_viscosity"] if "k_viscosity" in kwargs else 10**-6

        self.p0 = kwargs["p0"] if "p0" in kwargs else 0.05
        self.p1 = kwargs["p1"] if "p1" in kwargs else 0.95

        # deep pockets cover and maximum effective cover
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
            self._grid.add_field("bedrock", copy.copy(self._topo), at="node")
        # and flow__sender_node to nodes
        if not self._grid.has_field("flow__sender_node", at="node"):
            self._add_flow_sender_node()
        # add upstream and downstream field to each link
        self._dnode = self._grid["node"]["flow__receiver_node"]
        self._unode = self._grid["node"]["flow__sender_node"]
        # adds outlets and sources
        self.outlets = self._find_outlets()
        self.sources = self._find_sources()
        # finds the joints for special calculations
        self.joints, self.ujoints = self._find_joints()
        # add ouput fields
        self.initialize_output_fields()
        # update channel slopes
        self._calculate_reach_length()
        self._update_channel_slopes()
        # set conditions at the futher dowstream nodes
        # don't know yet how to handle these

    def run_one_step(self, dt, q_in=-1):
        """
        It updates the mean alluvium thickness and the bedrock elevation based
        on the model by Zhang et al (2015,2018)
        """
        # calculate parameters needed in the pde
        tau_star_crit = self._critical_shear_star()  # can be ignored, we use 0.0495 from the paper
        tau_star_crit = self.sstau_star_c
        tau_star = self._calculate_shear_star()
        self._calculate_fraction_alluvium_cover(self.p0, self.p1)
        self._calculate_corrected_fraction(self.p0)
        self._calculate_sed_capacity(tau_star, tau_star_crit)
        # boundary conditions
        self._boundary_conditions_precalc(q_in=q_in)
        # bed erosion (applied to the upstream node)
        self._bed_erosion(dt)
        # mean alluvium thickness change
        self._mean_alluvium_change(dt)
        # update slopes
        self._boundary_conditions_postcalc()
        self._update_channel_slopes()

    def _find_joints(self):
        """
        Returns the ids of nodes at the joint locations and the upstream node.
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
        receiver = copy.copy(self._grid["node"]["flow__receiver_node"])
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

    def _update_channel_slopes(self):
        """
        Returns the channel slopes using the mean alluvium cover and the
        bedrock.

        At joints it calculates an weighted average slope using the
        discharge from the tributaries as weights.

        Additional parameter used helps smooth peaks caused by the numerical
        instability of the central difference when calculating the slopes.
        The order of the approximation reduces in exchange for less
        instability and overall more realistic results.
        """

        c = self.su
        dx = self._grid.at_node["reach_length"]
        y = self._grid.at_node["bedrock"] + self._grid.at_node["mean_alluvium_thickness"]

        S = -(- c * y[self._unode]
              + (2 * c - 1) * y
              + (1 - c) * y[self._dnode]) / dx

        # joint upstream weighted slope
        for joint in self.joints:
            weight = (self._grid.at_node["discharge"][self.ujoints[joint]]
                      / np.sum(self._grid.at_node["discharge"][self.ujoints[joint]]))
            dnode = self._dnode[joint]
            Sjoint = -(- c * y[self.ujoints[joint]]
                       + (2 * c - 1) * y[joint]
                       + (1 - c) * y[dnode]) / dx[joint]
            S[joint] = np.sum(weight * Sjoint)

        # edge cases for boundary calculations
        if (c == 0) or (c == 1):
            pass
        else:
            S[self.sources] = S[self.sources] / (1 - c)
            S[self.outlets] = S[self.outlets] / (c)
        S[S < 0] = 0  # as implemented on the original code
        self._grid.at_node["channel_slope"] = S

    def _calculate_reach_length(self):
        """
        This calculates the reach length based on the appropiate
        slope formula. In the default case every point slope is
        found with up and down nodes and so the reach length is
        the total distance between those.

        FUUUUUUCK I have to modify this to work in a network by
        using a more complicated formula, I'll need to store the
        upstream and downstream distances.

        not so Fuck now, but I still need to modify how it behaves 
        at joints.
        """

        dis_up = (np.square(self._grid.x_of_node[self._unode] - self._grid.x_of_node)
                  + np.square(self._grid.y_of_node[self._unode] - self._grid.y_of_node))
        dis_up = np.sqrt(dis_up)

        # joint upstream mean distance
        up_joint_dis = [np.mean(dis_up[self.ujoints[joint]]) for joint in self.joints]
        up_joint_dis = np.array(up_joint_dis)
        dis_up[self.joints] = up_joint_dis

        dis_down = (np.square(self._grid.x_of_node - self._grid.x_of_node[self._dnode])
                    + np.square(self._grid.y_of_node - self._grid.y_of_node[self._dnode]))
        dis_down = np.sqrt(dis_down)

        self._grid.at_node["reach_length"] = (dis_up + dis_down) / 2
        self._grid.at_node["reach_length"][self.sources] = dis_down[self.sources]
        self._grid.at_node["reach_length"][self.outlets] = dis_up[self.outlets]

    def _calculate_flow_depths(self):
        """
        Re-calculates the flow depth based on the hydraulic relation
        H = [Q^2 / g*S*B^2*Cz^2]^(1/3)
        """
        H = ((self._grid.at_node["discharge"] * self._grid.at_node["discharge"])
             / (self.G * self._grid.at_node["channel_slope"]
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
            * np.power(self._grid.at_node["channel_slope"], 2 / 3)
            / self.spec_grav
            / self._grid.at_node["sediment_grain_size"])
        return tau_star

    def _critical_shear_star(self, method="Parker"):
        """
        Returns vector of the dimentionless critical shear stress based on grain size
        diameters using the corresponding method. Only Parker is implemented now.
            method: "Parker"
                Brownlie corrected formula.
            method: "Soulsby"
                Soulsby and Whitehouse formulation
            method: "Meyer"
                Meyer-Peter and Muller formulation
            method: Wilcock and Crowe
                Wilcock potential formualtion. Requires addicional parameters.
        """
        Re = (np.sqrt(self.spec_grav * self.G
                      * self._grid.at_node["sediment_grain_size"])
              * self._grid.at_node["sediment_grain_size"] / self.v)
        Re6 = np.power(Re, -0.6)
        if method == "Parker":
            tau_c_star = 0.5 * (0.22 * Re6 + 0.06 * np.power(10, -7.7 * Re6))
        return tau_c_star

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
        p = copy.copy(self._grid.at_node["fraction_alluvium_cover"])
        self._grid.at_node["fraction_alluvium_avaliable"] = (p - p0) / (1 - p0)
        # make sure no negative alluvium is avaliable for transport
        zeros = (self._grid.at_node["fraction_alluvium_avaliable"] < 0)
        self._grid.at_node["fraction_alluvium_avaliable"][zeros] = 0

    def _calculate_sed_capacity(self, tau_star, tau_star_crit=0.0495):
        """
        Calculates sediment flow capacity for a link and stores it at
        the upstream node field "sed_capacity". It uses the formula
        bla bla bla and ignores tau star crit
        the real issue is that the threshold of motion doesn't
        depend on the grain size which concerns me...
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

        dx = self._grid.at_node["reach_length"]
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
        dpq[self.joints] = (-c * up_joint_pq
                            + (2 * c - 1) * pq[self.joints]
                            + (1 - c) * pq[self._dnode[self.joints]])
        if (c == 1) or (c == 0):
            pass
        else:
            dpq[self.sources] = dpq[self.sources] / (1 - c)
            dpq[self.outlets] = dpq[self.outlets] / (c)

        cover_dif = (-self._grid.at_node["flood_intermittency"]
                     * dpq / dx
                     * dt / (1 - self.porosity)
                     / self._grid.at_node["fraction_alluvium_cover"])

        self._grid.at_node["mean_alluvium_thickness"] = self._grid.at_node["mean_alluvium_thickness"] + cover_dif
        self._grid.at_node["mean_alluvium_thickness"][self._grid.at_node["mean_alluvium_thickness"] < 0] = 0

    def _boundary_conditions_precalc(self, outlet="open", source="set_value", **kwargs):
        """
        sets boundary conditions for incoming flux and outgoing nodes.
        It currently defaults to an open boundary downstream and a
        maximum alluvium of 1 L (1 macro roughtness unit for full cover).

        The options for outlet and source are "copy_downstream", "open",
        "set_value". It defaults outlet to "open" and source to "set_value"
        so the parameter q_in should be provided.

        "set_values" Currently sets the incoming and outgoing flux using the
        q_in and q_out parameters as a float or a vector of the same size as
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

        if source == "copy_downstream":
            raise ValueError("copy_downstream for the sources is not currently implemented")

    def _boundary_conditions_postcalc(self, outlet="open", source="open",
                                      limit_outlet=True, baselevel=0):
        """
        It handles the boundary conditions after the change in alluvium
        and the bed have already been dealt with.

        The options for outlet are "copy_downstream", "open", "set_value".
        It defaults to "open". It defaults outlet to "open" and source to
        "open"

        limit_outlet parameter decies whenever to bound the outlet by
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
            # if kwargs are provided then those should be used to set the
            # elevations and alluvium cover. Otherwise it defaults to
            # the copy_dowstream behaviour (not implemented yet)
            # Assuming the previous to last node is not ambiguous
            prev_node = self._grid.at_node["flow__sender_node"][self.outlets]
            fields = {"mean_alluvium_thickness", "bedrock"}
            for field in fields:
                self._grid.at_node[field][self.outlets] = self._grid.at_node[field][prev_node]

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

        if source == "copy_downstream":
            next_node = self._grid.at_node["flow__receiver_node"][self.sources]
            fields = {"mean_alluvium_thickness", "bedrock"}
            for field in fields:
                self._grid.at_node[field][self.sources] = self._grid.at_node[field][next_node]

        if source == "set_value":
            # if kwargs are provided then those should be used to set the
            # elevations and alluvium cover. Otherwise it defaults to
            # the copy_dowstream behaviour (not implemented yet)
            next_node = self._grid.at_node["flow__receiver_node"][self.sources]
            fields = {"mean_alluvium_thickness", "bedrock"}
            for field in fields:
                self._grid.at_node[field][self.sources] = self._grid.at_node[field][next_node]

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

        If values are provided (all should be provided) then use the kwargs
        for each of the fields. See non existing example.
        """
        valid_fields = {"discharge": 300,
                        "flood_intermittency": 0.05,
                        "channel_width": 100,
                        "sediment_grain_size": 0.02,
                        "sed_capacity": 0,
                        "macroroughness": 1,
                        "mean_alluvium_thickness": 0.5
                        }
        # replace valid fields with the given parameters
        for field in kwargs:
            if field in valid_fields:
                valid_fields[field] = kwargs[field]
        # fill fields if the values were not provided as landlab fields
        nodes = np.ones(ngrid.at_node.size)
        for field in valid_fields:
            if (not ngrid.has_field(field, at="node")):
                if all_ones:
                    ngrid.add_field(field, 1 * nodes, at="node")
                else:
                    ngrid.add_field(field, valid_fields[field] * nodes, at="node")

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
        tail_nodes = copy.copy(self._grid.nodes_at_link[:, 0])
        head_nodes = copy.copy(self._grid.nodes_at_link[:, 1])
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
