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
        "flood_discharge": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m^3/s",
            "mapping": "link",
            "doc": "Morphodynamically active discharge.",
        },
        "flood_intermittency": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "link",
            "doc": "Fraction of time when the river is morphodynamically active: former value used when bedrock morphodynamics is considered",
        },
        "channel_width": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "link",
            "doc": "Channel link average width",
        },
        "sediment_grain_size": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "mm",
            "mapping": "link",
            "doc": "Sediment grain size on the link (single size sediment).",
        },
        "specific_gravity": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "-",
            "mapping": "link",
            "doc": "Submerged specific gravity of sediment.",
        },
        "dimentionless_Chezy": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "-",
            "mapping": "link",
            "doc": "Dimentionless Chezy C coefficient calculated as Cz=U/sqrt(tau/rho).",
        },
        "sediment_porosity": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "-",
            "mapping": "link",
            "doc": "Porosity of the alluvium.",
        },
        "macroroughness": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "link",
            "doc": "Thickness of macroroughness layer. See Zhang paper",
        },
        "reach_length": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "link",
            "doc": "River channel (link) lenght",
        },
        "wear_coefficient": {
            "dtype": float,
            "intent": "inout",
            "optional": True,
            "units": "m",
            "mapping": "link",
            "doc": "Beta. Wear coefficient. See Sklar and Dietrich 2004",
        },
        "abrasion_coefficient": {
            "dtype": float,
            "intent": "inout",
            "optional": True,
            "units": "m",
            "mapping": "link",
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
            "mapping": "link",
            "doc": "This is really used more as 4/2 parameters in the model, recheck when the modelling time comes. Mean bedload feed rate averaged over sedimentograph",
        },
        "sedimentograph_period": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "m",
            "mapping": "link",
            "doc": "Period of sedimentograph",
        },
        "high_feed_period": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "m",
            "mapping": "link",
            "doc": "Duration of high feed rate of sedimentograph",
        },
        "sed_capacity": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Sediment capacity of the link stored at the upstream node of each link",
        },
        "upstream_node": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "link",
            "doc": "The upstream node id at every link",
        },
        "downstream_node": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "link",
            "doc": "The downstream node id at every link",
        },
        "channel_slope": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "link",
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

    def __init__(self, grid, flow_director, corrected=True, clobber=False, **kwargs):
        """
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
            flow discharge in m^3/s. Defaults to 300
        Cz: float
            Dimentionless Chezy resistance coeff. Defaults to 10
        beta: float
            Related to sediment abrasion as beta = 3*alpha. Units of 1/m. Defaults to 0.05*0.001
        shear_coeff: float
            Sediment transport capacity coefficient (eq 5c). Defaults to 4
        shear_exp: float
            Sediment transport capacity exponent (eq 5c). Defaults to 1.5
        crit_shear: float
            Dimentionless critical shear stress for incipient motion of sediemnt. Defaults to 0.0495
        porosity: float
            Bedload sediment porosity. Defaults to 0.35
        spec_grav: float
            Specific gravity of sediment. Defaults to 1.65
        k_viscosity: float
            Kinematic viscosity of water. Defaults to 10**-6, the kinematic viscosity of water (at 20C)
        p0: float
            Lower percentage of bed cover (deep pockets). Defaults to 0.05 = 5%
        p1: float
            Higher percentage for effective bed cover. Defaults to 0.95 = 95%

        Examples
        --------

        """
        super().__init__(grid)

        self.corrected = corrected
        self.G = 9.80665  # gravity
        self.Q = kwargs["discharge"] if "discharge" in kwargs else 300
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

        # topographic elevation is the bedrock topography for now
        self._topo = self._grid.at_node["topographic__elevation"]
        if not self._grid.has_field("bedrock", at="node"):
            self._grid.add_field("bedrock", copy.copy(self._topo), at="node")
        # add upstream and downstream field to each link
        if not self._grid.has_field("upstream_node", at="link"):
            self._add_upstream_downstream_nodes()
        self._unode = self._grid.at_link["upstream_node"]
        self._dnode = self._grid.at_link["downstream_node"]
        # and flow__sender_node to nodes
        self._add_flow_sender_node()
        # adds outlets and sources
        self.outlets = self._find_outlets
        self.sources = self._find_sources
        # add ouput fields
        self.initialize_output_fields()
        # update channel slopes
        self._update_channel_slopes()
        # set conditions at the futher dowstream nodes
        # don't know yet how to handle these

    def run_one_step(self, dt, omit=np.array([], np.int32)):
        """
        It updates the mean alluvium thickness and the bedrock elevation based
        on the model by Zhang et al (2015,2018)
        """
        # calculate parameters needed in the pde
        tau_star_crit = self._critical_shear_star()  # can be ignored since we use 0.0495 from the paper
        tau_star_crit = 0.0495
        tau_star = self._calculate_shear_star()
        self._calculate_fraction_alluvium_cover(self.p0, self.p1)
        self._calculate_corrected_fraction(self.p0)
        self._calculate_sed_capacity(tau_star, tau_star_crit, omit)
        # bed erosion (applied to the upstream node)
        self._bed_erosion(dt)
        # mean alluvium thickness change
        self._mean_alluvium_change(dt)
        # update slopes
        self._update_channel_slopes()

    def _add_upstream_downstream_nodes(self):
        """
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
            at="link"
        )
        self._grid.add_field(
            "downstream_node",
            downstream_nodes,
            at="link"
        )

    def _add_flow_sender_node(self):
        """
        Adds the node field "flow__sender_node". This field has the
        upstream node based on the flow direction and IS ONLY MEANT
        to be used for finding the upstream source nodes. For junctions
        it returns only a single id ignoring one of the two tributaries.
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
        flow director are the flags for sinks, and the flow directions.
        """
        nodes = np.arange(0, self._grid["node"].size, 1)
        flow2self = (self._grid["node"]["flow__receiver_node"]==nodes)
        not_sink = np.logical_not(self._grid["node"]["flow__sink_flag"])
        return nodes[flow2self & not_sink]

    def _find_sources(self):
        """
        Finds the sources node's IDs of a river network created with 
        the flow director steepest component. Its dependency to the
        flow director are the flags for sinks, and the flow directions.
        """
        nodes = np.arange(0, self._grid["node"].size, 1)
        flow_from_none = (self._grid["node"]["flow__sender_node"]==nodes)
        return nodes[flow_from_none]

    def _update_channel_slopes(self):
        """
        Returns the channel slopes by adding the slope of the mean alluvium cover
        and the bedrock slope.
        Re-calculate channel slopes during each timestep.
        """
        Sb = (
            (self._grid.at_node["bedrock"][self._unode]
                - self._grid.at_node["bedrock"][self._dnode])
            / self._grid.at_link["reach_length"])
        Sa = (
            (self._grid.at_node["mean_alluvium_thickness"][self._unode]
                - self._grid.at_node["mean_alluvium_thickness"][self._dnode])
            / self._grid.at_link["reach_length"])

        S = Sa + Sb
        S[S < 0] = 0
        self._grid.at_link["channel_slope"] = S

    def _calculate_flow_depths(self):
        """
        Re-calculates the flow depth based on the hydraulic relation
        H = [Q^2 / g*S*B^2*Cz^2]^(1/3)
        """
        H = ((self.Q * self.Q)
             / (self.G * self._grid.at_link["channel_slope"]
                * np.square(self._grid.at_link["channel_width"])
                * self.Cz * self.Cz)) ** (1 / 3)
        return H

    def _calculate_shear_star(self, method="Parker"):
        """
        Re-calculates the dimentionless bed shear stress at normal flow
        conditions based on the hydraulic relations derived by Parker
        tau_star = [Q^2 / g*B^2*Cz^2]^(1/3) * S^(2/3) / (R * D)
        """
        tau_star = (np.power((self.Q * self.Q)
                             / (self.G * self.Cz * self.Cz
                                * np.square(self._grid.at_link["channel_width"])
                                ), 1 / 3)
                    * np.power(self._grid.at_link["channel_slope"], 2 / 3)
                    / (self.spec_grav
                       * self._grid.at_link["sediment_grain_size"]))
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
                      * self._grid.at_link["sediment_grain_size"])
              * self._grid.at_link["sediment_grain_size"] / self.v)
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
        self._grid.at_node["fraction_alluvium_cover"] = np.zeros_like(self._grid.at_node["fraction_alluvium_cover"])
        chi = self._grid.at_node["mean_alluvium_thickness"][self._unode] / self._grid.at_link["macroroughness"]
        threshold = (1 - p0) / (p1 - p0)
        cover = np.where(chi < threshold, p0 + (p1 - p0) * chi, np.ones_like(chi))
        self._grid.at_node["fraction_alluvium_cover"][self._unode] = cover

    def _calculate_corrected_fraction(self, p0=0.05):
        """
        Calculates pa, the corrected alluvium avaliable for transport from Zhang 2018.
        it uses yet another linear function to remove non-zero transport under lack
        of sediment conditions (in deep pockets).
        """
        p = self._grid.at_node["fraction_alluvium_cover"]
        self._grid.at_node["fraction_alluvium_avaliable"] = (p - p0) / (1 - p0)
        # make sure no negative alluvium is avaliable for transport
        zeros = (self._grid.at_node["fraction_alluvium_avaliable"] < 0)
        self._grid.at_node["fraction_alluvium_avaliable"][zeros] = 0

    def _calculate_sed_capacity(self, tau_star, tau_star_crit=0.0495, omit=np.array([], dtype=np.int32)):
        """
        Calculates sediment flow capacity for a link and stores it at
        the upstream node field "sed_capacity". It uses the formula
        bla bla bla and ignores tau star crit
        the real issue is that the threshold of motion doesn't
        depend on the grain size which concerns me...
        """
        sed_cap_omit = self._grid.at_node["sed_capacity"][omit]
        excess_shear = tau_star - self.sstau_star_c
        zeromask = np.ones_like(excess_shear)
        zeromask[excess_shear < 0] = 0
        excess_shear = excess_shear * zeromask

        self._grid.at_node["sed_capacity"][self._unode] = (
            self.ssalpha * zeromask
            * np.power(self.spec_grav * self.G * self._grid.at_link["sediment_grain_size"], 0.5)
            * self._grid.at_link["sediment_grain_size"]
            * np.power(excess_shear, self.ssna))
        self._grid.at_node["sed_capacity"][omit] = sed_cap_omit
        nanarr = np.isnan(self._grid.at_node["sed_capacity"])
        if np.any(nanarr):
            raise ArithmeticError("A shear calculation yield nans")

    def _bed_erosion(self, dt):
        """
        Applies bed erosion discretizing the differential equation
        of the bed in the upstream nodes (representing the link downstream).
        Must be called after updating sediment capacity,
        fraction of alluvium cover and fraction of avaliable alluvium.
        """
        pa = 0
        if self.corrected:
            pa = self._grid.at_node["fraction_alluvium_avaliable"][self._unode]
        else:
            pa = self._grid.at_node["fraction_alluvium_cover"][self._unode]
        erosion = (self._grid.at_link["flood_intermittency"]
                   * self.wear_coefficient
                   * self._grid.at_node["sed_capacity"][self._unode]
                   * pa * (1 - pa) * dt)
        self._grid.at_node["bedrock"][self._unode] = self._grid.at_node["bedrock"][self._unode] - erosion
        self._grid.at_node["bedrock"][self._grid.at_node["bedrock"] < 0] = 0

    def _mean_alluvium_change(self, dt):
        """
        Adds/Removes the difference in mean alluvium thickness by 
        discretizing the differential equation of the alluvium
        in the upstream nodes (representing the link downstream).
        Must be called after updating sediment capacity,
        fraction of alluvium cover and fraction of avaliable alluvium.
        """
        dx = self._grid.at_link["reach_length"]
        dpq = 0
        if self.corrected:
            dpq = ((self._grid.at_node["fraction_alluvium_avaliable"][self._unode]
                    * self._grid.at_node["sed_capacity"][self._unode])
                   - (self._grid.at_node["fraction_alluvium_avaliable"][self._dnode]
                      * self._grid.at_node["sed_capacity"][self._dnode]))
        else:
            dpq = ((self._grid.at_node["fraction_alluvium_cover"][self._unode]
                    * self._grid.at_node["sed_capacity"][self._unode])
                   - (self._grid.at_node["fraction_alluvium_cover"][self._dnode]
                      * self._grid.at_node["sed_capacity"][self._dnode]))

        cover_dif = (-self._grid.at_link["flood_intermittency"]
                     * dpq / dx
                     * dt / (1 - self.porosity)
                     / self._grid.at_node["fraction_alluvium_cover"][self._unode])
        self._grid.at_node["mean_alluvium_thickness"][self._unode] = self._grid.at_node["mean_alluvium_thickness"][self._unode] + cover_dif
        self._grid.at_node["mean_alluvium_thickness"][self._grid.at_node["mean_alluvium_thickness"] < 0] = 0

    def _sedimentograph(self, **kwargs):
        Tc = 40
        Th = 2.5
        rh = Tc / Th
        rl = 1 - rh
        random_seed = 2
        np.random.seed(random_seed)
        low_random = 6
        high_random = 12
        step_random = 0.5
        range_random = np.arange(low_random, high_random + step_random, step_random)
        range_random_mean = np.mean(range_random)
        qaf_m = 0.000834

    def _set_boundary_conditions(self, open_outlet=False, q_up=-1.0, t=-1.0, q_out=-1):
        """
        sets boundary conditions for incoming flux and outgoing nodes
        as well as boundary conditions on the bed.

        Currently it sets the incoming and outgoing flux using the
        q_up and q_out parameters as a constant or a time function, but
        it sets all sources/outlets at the same value.
        self.sources is an int np list of the nodes that are sources and
        similarly self.outlets for outlets.

        out_topo is the elevation outlet boundary condition. Since
        The component is not yet decided to be a single tree or a
        forest then we shall assume there's a single outlet and so
        self.outlets is a np attay with a single int

        I should add a way of giving different values at different sources
        but not today
        """
        # set outlets elevations
        self._grid.at_node["bedrock"][self.outlets]

        # set fluxes in and out
        flux_in = np.zeros_like(self.sources, dtype=np.float64)
        flux_out = np.zeros_like(self.outlets, dtype=np.float64)
        if isinstance(q_up, float):
            if q_up < 0:
                raise ValueError("negative flux at sources")
            flux_in.fill(q_up)
        else:
            if t < 0:
                raise ValueError("the t parameter was not provided")
            flux_in.fill(q_up(t))

        if isinstance(q_out, float):
            if q_out < 0:
                raise ValueError("negative flux at outlets")
            flux_out.fill(q_out)
        else:
            if t < 0:
                raise ValueError("the t parameter was not provided")
            flux_out.fill(q_out(t))

        self._grid.at_node["sed_capacity"][self.sources] = flux_in
        self._grid.at_node["sed_capacity"][self.outlets] = flux_out



    @staticmethod
    def _preset_fields(ngrid, all_ones=False):
        """
        presets all the required fields of the grid needed for an instance
        of componentcita. If all_ones is True it sets all such parameters
        to 1, otherwise it uses commonly found values of these parameters.
        For more info on the parameters set see non optional inputs
        of this component.
        """

        nodes1 = np.ones(ngrid.at_node.size)
        links1 = np.ones(ngrid.at_link.size)
        if all_ones:
            ngrid.add_field("reach_length", copy.copy(links1), at="link")
            ngrid.add_field("flood_discharge", copy.copy(links1), at="link")
            ngrid.add_field("flood_intermittency", copy.copy(links1), at="link")
            ngrid.add_field("channel_width", copy.copy(links1), at="link")
            ngrid.add_field("sediment_grain_size", copy.copy(links1), at="link")
            ngrid.add_field("sed_capacity", copy.copy(nodes1), at="node")
            ngrid.add_field("macroroughness", copy.copy(links1), at="link")
            ngrid.add_field("mean_alluvium_thickness", copy.copy(nodes1), at="node")
        else:
            ngrid.add_field("reach_length", copy.copy(100 * links1), at="link")
            ngrid.add_field("flood_discharge", copy.copy(300 * links1), at="link")
            ngrid.add_field("flood_intermittency", copy.copy(0.05 * links1), at="link")
            ngrid.add_field("channel_width", copy.copy(100 * links1), at="link")
            ngrid.add_field("sediment_grain_size", copy.copy(0.02 * links1), at="link")
            ngrid.add_field("sed_capacity", copy.copy(0 * nodes1), at="node")
            ngrid.add_field("macroroughness", copy.copy(1 * links1), at="link")
            ngrid.add_field("mean_alluvium_thickness", copy.copy(0.5 * nodes1), at="node")

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

