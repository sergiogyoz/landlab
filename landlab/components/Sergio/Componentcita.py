import numpy as np
import copy
from landlab import Component

from landlab.components import FlowDirectorSteepest
from landlab.data_record import DataRecord
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
            "intent": "out",
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

    def __init__(self, grid, flow_director, clobber=False, **kwargs):
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
        
        Examples
        --------
        
        """
        super().__init__(grid)

        self.G = 9.80665  # gravity
        self.Q = kwargs["discharge"] if "discharge" in kwargs else 300
        self.Cz = kwargs["Cz"] if "Cz" in kwargs else 10
        self.wear_coefficient = kwargs["beta"] if "beta" in kwargs else 0.05 * 0.001

        self.ssalpha = kwargs["shear_coeff"] if "shear_coeff" in kwargs else 4
        self.ssna = kwargs["shear_exp"] if "shear_exp" in kwargs else 1.5
        self.sstau_star_c = kwargs["crit_shear"] if "crit_shear" in kwargs else 0.0495

        self.porosity = kwargs["porosity"] if "porosity" in kwargs else 0.35
        self.spec_grav = kwargs["spec_grav"] if "spec_grav" in kwargs else 1.65
        self.v = kwargs["k_viscosity"] if "k_viscosity" in kwargs else 10**-6

        # another combination of parameters for equation 5c
        # self.ssalpha = 5.7
        # self.ssna = 1.5
        # self.sstau_star_c = 0.03

        # it must be a NetworkModelGrid
        if not isinstance(grid, NetworkModelGrid):
            msg = "NetworkSedimentTransporter: grid must be NetworkModelGrid"
            raise ValueError(msg)

        # self._fd = flow_director  # should I assume flow routing happens outside for now
        # supported flow directors, ommited for now
        if not isinstance(flow_director, FlowDirectorSteepest):
            msg = (
                "NetworkSedimentTransporter: flow_director must be "
                "FlowDirectorSteepest."
            )
            raise ValueError(msg)

        # local topo var to ease access
        self._topo = self._grid.at_node["topographic__elevation"]
        # topographic elevation is the bedrock topography for now
        if not self._grid.has_field("bedrock"):
            self._grid.add_field("bedrock", copy.copy(self._topo), at="node")
        # add upstream and downstream field to each link
        if not self._grid.has_field("upstream_node"):
            self._add_upstream_downstream_nodes()
        self._unode = self._grid.at_link["upstream_node"]
        self._dnode = self._grid.at_link["downstream_node"]
        # add ouput fields
        self.initialize_output_fields()
        # update channel slopes
        self._grid.at_link["channel_slope"] = self._update_channel_slopes()
        # set conditions at the futher dowstream nodes
        #  don't know yet how to handle these

    def run_one_step(self, dt, urate=-1):
        """
        It updates the mean alluvium thickness and the bedrock elevation based
        on the model by Zhang et al (2015,2018)
        """
        # uplift should be taken out of the code
        if urate >= 0:
            self._uplift(urate, dt)
        else:
            self._uplift(1 * 10**-3, dt)
        # bed erosion (applied to the upstream node)
        tau_star_crit = self._critical_shear_star()  # can be ignored since we use 0.0495 from the paper
        tau_star_crit = 0.0495
        tau_star = self._calculate_shear_star()
        self._calculate_fraction_alluvium_cover()
        self._calculate_sed_capacity(tau_star, tau_star_crit)
        self._bed_erosion(dt)
        # mean alluvium thickness change
        dx = self._grid.at_link["reach_length"]
        dpq = (self._grid.at_node["fraction_alluvium_cover"][self._unode] * self._grid.at_node["sed_capacity"][self._unode]
               - self._grid.at_node["fraction_alluvium_cover"][self._dnode] * self._grid.at_node["sed_capacity"][self._dnode])
        cover_dif = (-self._grid.at_link["flood_intermittency"]
                     * dpq / dx
                     * dt / (1 - self.porosity)
                     / self._grid.at_node["fraction_alluvium_cover"][self._unode])
        self._grid.at_node["mean_alluvium_thickness"][self._unode] = self._grid.at_node["mean_alluvium_thickness"][self._unode] + cover_dif
        # update slopes
        self._update_channel_slopes()

    def _uplift(self, dt, urate=-1,):
        """
        uplift rate should be given in units of m/kyr. dt units are kyr
        """
        if urate < 0:
            self._grid.at_node["bedrock"] = self._grid.at_node["bedrock"] + self._grid.at_node["uplift_rate"] * dt
        else:
            self._grid.at_node["bedrock"] = self._grid.at_node["bedrock"] + urate * dt

    def _add_upstream_downstream_nodes(self):
        """
        It adds the fields "upstream_node" and "downstream_node" to links.
        each field has the id of the upstream and downstream node based on the 
        Network model node ids. Array index is in correspondance with links array index.
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
        return S

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
        print(self.Q)
        print(self.G)
        print(self.Cz)
        print(self._grid.at_link["channel_width"])
        print(self._grid.at_link["channel_slope"])
        print(self._grid.at_link["sediment_grain_size"])
        
        tau_star = (((self.Q * self.Q)
                     / (self.G * self.Cz * self.Cz
                        * np.square(self._grid.at_link["channel_width"])
                        )) ** (1 / 3)
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
        Calculates the p (or pa) alluvium cover from Zhang. It uses a
        linear alluvium cover function. line between 0.05 and 0.95 from
        minimum (deep pockets) cover to maximum (effective) cover of the bed
        """
        self._grid.at_node["fraction_alluvium_cover"] = np.zeros_like(self._grid.at_node["fraction_alluvium_cover"])
        chi = self._grid.at_node["mean_alluvium_thickness"][self._unode] / self._grid.at_link["macroroughness"]
        threshold = (1 - p0) / (p1 - p0)
        print(chi)
        cover = np.where(chi < threshold, p0 + (p1 - p0) * chi, np.ones_like(chi))
        self._grid.at_node["fraction_alluvium_cover"][self._unode] = cover
        print(self._grid.at_node["fraction_alluvium_cover"])

    def _calculate_sed_capacity(self, tau_star, tau_star_crit=0.0495):
        """
        Calculates sediment flow capacity for a link and stores it at
        the upstream node field "sed_capacity". It uses the formula
        bla bla bla and ignores tau star crit
        the real issue is that the threshold of motion doesn't
        depend on the grain size which is concerning to me...
        """
        excess_shear = tau_star - self.sstau_star_c
        excess_shear[excess_shear < 0] = 0
        self._grid.at_node["sed_capacity"][self._unode] = (self.ssalpha
            * ((self.spec_grav * self.G * self._grid.at_link["sediment_grain_size"])**0.5)
            * self._grid.at_link["sediment_grain_size"]
            * (excess_shear)**(self.ssna))

    def _bed_erosion(self, dt):
        """
        Calculates bed erosion discretizing the differential equation
        of the bed in the upstream nodes (representing the link downstream).
        Must be called after updating sediment capacity and
        fraction of alluvium cover.
        """
        erosion = (self._grid.at_link["flood_intermittency"]
                   * self.wear_coefficient
                   * self._grid.at_node["sed_capacity"][self._unode]
                   * self._grid.at_node["fraction_alluvium_cover"][self._unode]
                   * (1 - self._grid.at_node["fraction_alluvium_cover"][self._unode])
                   * dt)
        self._grid.at_node["bedrock"][self._unode] = self._grid.at_node["bedrock"][self._unode] - erosion

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
            ngrid.add_field("reach_length", links1, at="link")
            ngrid.add_field("flood_discharge", links1, at="link")
            ngrid.add_field("flood_intermittency", links1, at="link") 
            ngrid.add_field("channel_width", links1, at="link")
            ngrid.add_field("sediment_grain_size", links1, at="link")
            ngrid.add_field("sed_capacity", nodes1, at="node")
            ngrid.add_field("macroroughness", links1, at="link")
        else:
            ngrid.add_field("reach_length", 100 * links1, at="link")
            ngrid.add_field("flood_discharge", 300 * links1, at="link")
            ngrid.add_field("flood_intermittency", 0.05 * links1, at="link") 
            ngrid.add_field("channel_width", 100 * links1, at="link")
            ngrid.add_field("sediment_grain_size", 0.02 * links1, at="link")
            ngrid.add_field("sed_capacity", 0 * nodes1, at="node")
            ngrid.add_field("macroroughness", 1 * links1, at="link")

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



