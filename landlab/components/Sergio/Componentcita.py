import numpy as np
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

    def __init__(self, grid, flow_director, sm=0.5):
        """
        Parameters
        ----------
        grid: RasterModelGrid
            A grid.
        flow_director: :py:class:`~landlab.components.FlowDirectorSteepest`
            A landlab flow director. Currently, must be
            :py:class:`~landlab.components.FlowDirectorSteepest`.
        smooth: float
            Smoothing parameter to smooth along gradient direction. Must
            be between 0 and 1
        """

        # must be a NetworkModelGrid ??
        """if not isinstance(grid, NetworkModelGrid):
            msg = "NetworkSedimentTransporter: grid must be NetworkModelGrid"
            raise ValueError(msg)
        """
        super().__init__(grid)
        self.smooth = sm
        self.topo = self._grid.at_node["topographic__elevation"]
        self._fd = flow_director
        """if not isinstance(flow_director, FlowDirectorSteepest):
            msg = (
                "NetworkSedimentTransporter: flow_director must be "
                "FlowDirectorSteepest."
            )
            raise ValueError(msg)"""

        self.initialize_output_fields()

    def run_one_step(self, dt):
        self._grid.at_node["topographic__elevation"] = self._grid.at_node["topographic__elevation"] + dt

    def update_dumb_heights(self):
        for i in range(len(self._grid.at_node["topographic__elevation"])):
            self._grid.at_node["topographic__elevation"][i] = self._grid.at_node["topographic__elevation"][i] + 1

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
