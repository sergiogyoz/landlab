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
    """

    _name = "GrainSizeBedRockInciser"

    _unit_agnostic = False

    _info = {
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "channel_slope": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "link",
            "doc": "Channel slopes from topographic elevation",
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
            "optional": False,
            "units": "-",
            "mapping": "link",
            "doc": "Dimentionless Chezy C coefficient calculated as Cz=U/sqrt(tau/rho).",
        },
        "sediment_porosity": {
            "dtype": float,
            "intent": "in",
            "optional": False,
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
            "intent": "in",
            "optional": False,
            "units": "km",
            "mapping": "link",
            "doc": "Thickness of macroroughness layer. See Zhang 2015",
        },
        "wear_coefficient": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "link",
            "doc": "Wear coefficient. See Sklar and Dietrich 2004",
        },
        "uplift_rate": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "mm/year",
            "mapping": "link",
            "doc": "local uplift rate. (set instead on the model parameter creation)",
        },
        "sedimentograph_info": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "link",
            "doc": "This is really used more as 4/2 parameters in the model, recheck when the modelling time comes. Mean bedload feed rate averaged over sedimentograph",
        },
        "sedimentograph_period": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "link",
            "doc": "Period of sedimentograph",
        },
        "high_feed_period": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "link",
            "doc": "Duration of high feed rate of sedimentograph",
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

    def __init__(self, grid, flow_director):
        """
        Parameters
        ----------
        grid: RasterModelGrid
            A grid.
        flow_director: :py:class:`~landlab.components.FlowDirectorSteepest`
            A landlab flow director. Currently, must be
            :py:class:`~landlab.components.FlowDirectorSteepest`.
        """
        super().__init__(grid)
        D = 2/1000  # default to 2mm in units of m
        G = 9.80665  # gravity
        R = 1.65  # specific gravity of sediment
        Q = 300  # flow discharge m3/s
        Cz = 10  # Dimentionless Chezy resistance coeff

        # it must be a NetworkModelGrid 
        if not isinstance(grid, NetworkModelGrid):
            msg = "NetworkSedimentTransporter: grid must be NetworkModelGrid"
            raise ValueError(msg)
        self.smooth = sm
        self.topo = self._grid.at_node["topographic__elevation"]
        # self._fd = flow_director  # should I assume flow routing happens outside for now
        # supported flow directors, ommited for now
        if not isinstance(flow_director, FlowDirectorSteepest):
            msg = (
                "NetworkSedimentTransporter: flow_director must be "
                "FlowDirectorSteepest."
            )
            raise ValueError(msg)

        self.initialize_output_fields()  # are there any? so far I haven't define any as such

    def run_one_step(self, dt):
        self._grid.at_node["topographic__elevation"] = self._grid.at_node["topographic__elevation"] + dt

    def _upstream__downstream_nodes(self):
        """
        It adds the fields "upstream_node" and "downstream_node" to links.
        each field has the id of the upstream and downstream node based on the 
        Network model node ids.
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
        """Re-calculate channel slopes during each timestep."""
        upstream_nodes = self._fd.upstream_node_at_link()
        downstream_nodes = self._fd.downstream_node_at_link()
        self._grid.at_link["channel_slope"] = (
            (
                self._grid.at_node["topographic__elevation"][upstream_nodes]
                - self._grid.at_node["topographic__elevation"][downstream_nodes]
            )
            / self._grid.at_link["reach_length"])

    def _update_flow_depths(self):
        """
        Re-calculates the flow depth based on the hydraulic relation
        H = (Q2 / g*S*B^2*Cz^2 )
        """
        pass
