import numpy as np
from landlab import Component


class DumbC(Component):
    """
    Component doc
    """
    _name = "SuperDuperComponent"

    _unit_agnostic = True

    _info = {
        "dumb_height": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "multiplies the height values 'cause... Idk, math",
        },
        "topographic__elevation": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
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

    def __init__(self, grid, bh=20, s=1):
        """
        Parameters
        ----------
        grid: RasterModelGrid
            A grid.
        a_base_height: float, optional
            A height that I will add to my dumb_height values at every node.
        """
        super().__init__(grid)

        self.a_base_height = bh
        self.spread = s

        if "dumb_height" not in self.grid.at_node:
            self._grid.add_zeros("dumb_height", at="node")

        self.dhs = self._grid.at_node["dumb_height"]
        self._grid.at_node["topographic__elevation"] = self.dhs + self.a_base_height

    def run_one_step(self, dt):
        delta_spread = dt * self.spread
        self._grid.at_node["topographic__elevation"] = self._grid.at_node["topographic__elevation"] + np.random.standard_normal(size=self.dhs.shape) * delta_spread

    def update_dumb_heights(self):
        for i in range(len(self._grid.at_node["topographic__elevation"])):
            self._grid.at_node["topographic__elevation"][i] = self._grid.at_node["topographic__elevation"][i] + np.random.standard_normal() * self.spread