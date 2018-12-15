import re

from itertools import combinations

import numpy as np
import xarray as xr
from scipy.spatial import Delaunay, Voronoi

from ...utils import jaggedarray
from ..sort.sort import reverse_one_to_one
from .voronoi import id_array_contains


class VoronoiDelaunay(object):

    """Represent a scipy.spatial Voronoi as a landlab graph."""

    def __init__(self, xy_of_node):
        """A Voronoi with landlab-style names.

        Parameters
        ----------
        xy_of_node : ndarray of float, shape *(N, 2)*
            Coordinates of nodes.

        Examples
        --------
        >>> import numpy as np
        >>> import scipy
        >>> from landlab.graph.voronoi.voronoi_to_graph import VoronoiDelaunay

        >>> xy_of_node = [
        ...     [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        ...     [0.5, 1.0], [1.5, 1.0], [2.5, 1.0],
        ...     [0.0, 2.0], [1.0, 2.0], [2.0, 2.0],
        ... ]
        >>> graph = VoronoiDelaunay(xy_of_node)
        >>> voronoi = scipy.spatial.Voronoi(xy_of_node)

        >>> np.all(graph.x_of_node == voronoi.points[:, 0])
        True
        >>> np.all(graph.y_of_node == voronoi.points[:, 1])
        True

        >>> np.all(graph.x_of_corner == voronoi.vertices[:, 0])
        True
        >>> np.all(graph.y_of_corner == voronoi.vertices[:, 1])
        True

        >>> np.all(graph.corners_at_face == voronoi.ridge_vertices)
        True
        """
        delaunay = Delaunay(xy_of_node)
        voronoi = Voronoi(xy_of_node)

        mesh = xr.Dataset(
            {
                "node": xr.DataArray(
                    data=np.arange(len(voronoi.points)),
                    coords={
                        "x_of_node": xr.DataArray(voronoi.points[:, 0], dims=("node",)),
                        "y_of_node": xr.DataArray(voronoi.points[:, 1], dims=("node",)),
                    },
                    dims=("node",),
                ),
                "corner": xr.DataArray(
                    data=np.arange(len(voronoi.vertices)),
                    coords={
                        "x_of_corner": xr.DataArray(
                            voronoi.vertices[:, 0], dims=("corner",)
                        ),
                        "y_of_corner": xr.DataArray(
                            voronoi.vertices[:, 1], dims=("corner",)
                        ),
                    },
                    dims=("corner",),
                ),
            }
        )
        mesh.update(
            {
                "nodes_at_link": xr.DataArray(
                    np.asarray(voronoi.ridge_points, dtype=int), dims=("link", "Two")
                ),
                "nodes_at_patch": xr.DataArray(
                    np.asarray(delaunay.simplices, dtype=int), dims=("patch", "Three")
                ),
                "corners_at_face": xr.DataArray(
                    voronoi.ridge_vertices, dims=("face", "Two")
                ),
                "corners_at_cell": xr.DataArray(
                    self._corners_at_cell(voronoi.regions),
                    dims=("cell", "max_corners_per_cell"),
                ),
                "n_corners_at_cell": xr.DataArray(
                    [len(cell) for cell in voronoi.regions], dims=("cell",)
                ),
                "nodes_at_face": xr.DataArray(
                    np.asarray(voronoi.ridge_points, dtype=int), dims=("face", "Two")
                ),
                "cell_at_node": xr.DataArray(voronoi.point_region, dims=("node",)),
            }
        )
        self._mesh = mesh

    @staticmethod
    def _corners_at_cell(regions):
        jagged = jaggedarray.JaggedArray(regions)
        return np.asarray(
            jaggedarray.unravel(jagged.array, jagged.offset, pad=-1), dtype=int
        )

    @property
    def x_of_node(self):
        """x-coordinate of nodes."""
        return self._mesh["x_of_node"].values

    @property
    def y_of_node(self):
        """y-coordinate of nodes."""
        return self._mesh["y_of_node"].values

    @property
    def x_of_corner(self):
        """x-coordinate of corners."""
        return self._mesh["x_of_corner"].values

    @property
    def y_of_corner(self):
        """y-coordinate of corners."""
        return self._mesh["y_of_corner"].values

    @property
    def nodes_at_patch(self):
        """Nodes that form a patch."""
        return self._mesh["nodes_at_patch"].values

    @property
    def nodes_at_link(self):
        """Nodes at link ends."""
        return self._mesh["nodes_at_link"].values

    @property
    def nodes_at_face(self):
        """Nodes at either side of a face."""
        return self._mesh["nodes_at_face"].values

    @property
    def corners_at_face(self):
        """corners at face ends."""
        return self._mesh["corners_at_face"].values

    @property
    def corners_at_cell(self):
        """Corners that from a cell."""
        return self._mesh["corners_at_cell"].values

    @property
    def n_corners_at_cell(self):
        """Number of corners forming a cell."""
        return self._mesh["n_corners_at_cell"].values

    @property
    def cell_at_node(self):
        """Cell that surrounds a node."""
        return self._mesh["cell_at_node"].values


class VoronoiDelaunayToGraph(VoronoiDelaunay):
    def __init__(self, xy_of_node, perimeter_links=None):
        """A Voronoi with landlab-style names.

        Parameters
        ----------
        xy_of_node : ndarray of float, shape *(N, 2)*
            Coordinates of nodes.
        perimeter_links : ndarray of int, shape *(N, 2)*
            Node pairs of links on the perimeter of the graph.

        Examples
        --------
        >>> import numpy as np
        >>> import scipy
        >>> from landlab.graph.voronoi.voronoi_to_graph import VoronoiDelaunay

        >>> xy_of_node = [
        ...     [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        ...     [0.5, 1.0], [1.5, 1.0], [2.5, 1.0],
        ...     [0.0, 2.0], [1.0, 2.0], [2.0, 2.0],
        ... ]

        Create this hex graph::

            6 - 7 - 8
            |\ / \ / \
            | 3 - 4 - 5
            |/ \ / \ /
            0 - 1 - 2

        >>> graph = VoronoiDelaunayToGraph(xy_of_node)
        >>> len(graph.nodes_at_patch)
        9
        >>> corner = np.argmin(graph.x_of_corner)
        >>> graph.x_of_corner[corner], graph.y_of_corner[corner]
        (-0.75, 1.0)

        Create this hex graph::

            6 - 7 - 8
             \ / \ / \
              3 - 4 - 5
             / \ / \ /
            0 - 1 - 2

        >>> perimeter_links = np.array(
        ...     [[0, 1], [1, 2], [2, 5], [5, 8], [8, 7], [7, 6], [6, 3], [3, 0]], dtype=int
        ... )
        >>> graph = VoronoiDelaunayToGraph(xy_of_node, perimeter_links=perimeter_links)
        >>> len(graph.nodes_at_patch)
        8

        >>> graph.node_at_cell
        array([4])
        """
        super(VoronoiDelaunayToGraph, self).__init__(xy_of_node)

        if perimeter_links is not None:
            self._perimeter_links = np.asarray(perimeter_links, dtype=int).reshape(
                (-1, 2)
            )
        else:
            self._perimeter_links = None

        mesh = self._mesh
        mesh.update(
            {
                "links_at_patch": xr.DataArray(
                    self._links_at_patch(
                        mesh["nodes_at_link"].values, mesh["nodes_at_patch"].values
                    ),
                    dims=("patch", "Three"),
                ),
                "node_at_cell": xr.DataArray(
                    reverse_one_to_one(mesh["cell_at_node"].values), dims=("cell",)
                ),
            }
        )
        mesh.update(
            {
                "faces_at_cell": xr.DataArray(
                    self._links_at_patch(
                        mesh["corners_at_face"].values, mesh["corners_at_cell"].values
                    ),
                    dims=("cell", "max_faces_per_cell"),
                )
            }
        )
        self.drop_corners(self.unbound_corners())
        self.drop_perimeter_faces()
        self.drop_perimeter_cells()

    @staticmethod
    def _links_at_patch(nodes_at_link, nodes_at_patch):
        from ..sort.sort import map_sorted_rolling_pairs

        link_at_nodes = np.argsort(nodes_at_link[:, 0])
        links_at_patch_ = map_sorted_rolling_pairs(
            nodes_at_link[link_at_nodes],
            link_at_nodes,
            nodes_at_patch,
        )

        return links_at_patch_

    def is_perimeter_face(self):
        """Identify faces that are on the perimeter.

        A face is on the perimeter if one of it's ends is undefined (has an
        id of -1).

        Returns
        -------
        ndarray of bool, shape *(n_faces,)*
            *True* where faces are on the perimeter.
        """
        return np.any(self.corners_at_face == -1, axis=1)

    def is_perimeter_cell(self):
        """Identify cells that are unbound.

        A corner is unbound (or on the perimeter) if one of it's corners
        is undefined (has an id of -1) or has fewer than three sides.

        Returns
        -------
        ndarray of bool, shape *(n_cells,)*
            *True* where cells are on the perimeter.
        """
        is_not_a_cell = id_array_contains(
            self.corners_at_cell, self.n_corners_at_cell, -1
        )
        is_not_a_cell |= self.n_corners_at_cell < 3

        return is_not_a_cell

    def is_perimeter_link(self):
        """Identify links that are on the perimeter of the graph.

        If *perimeter_links* was provided when the VoronoiDelaunayToGraph was
        created, these are the perimeter links. Otherwise, perimeter links
        are those that cross perimeter faces.

        Returns
        -------
        ndarray of bool, shape *(n_links,)*
            *True* where links are on the perimeter.
        """
        from ..sort.sort import pair_isin_sorted_list

        if self._perimeter_links is not None:
            is_perimeter_link = pair_isin_sorted_list(
                self._perimeter_links,
                self.nodes_at_link,
                sorter=np.argsort(self._perimeter_links[:, 0]),
            )
        else:
            is_perimeter_link = self.is_perimeter_face()
        return is_perimeter_link

    def unbound_corners(self):
        """Identify corners that are not contained within the bounds of the graph.

        A corner is unbound if it is only connected to perimeter faces.

        Returns
        -------
        ndarray of int
            Array of corners that are not contained in the graph.
        """
        faces_to_drop = np.where(
            self.is_perimeter_face()
            & (self.is_perimeter_link() != self.is_perimeter_face())
        )

        unbound_corners = self.corners_at_face[faces_to_drop].reshape((-1,))
        return unbound_corners[unbound_corners >= 0]

    def is_bound_corner(self):
        """Identify corners that are contained within the bounds of the graph.

        Returns
        -------
        ndarray of bool, shape *(n_corners,)*
            *True* where corners are bound.
        """
        corners = np.full(self._mesh.dims["corner"], True)
        corners[self.unbound_corners()] = False

        return corners

    def drop_corners(self, corners):
        """Drop corners and associated elements for the graph.

        First given corners are dropped from the graph. Once this is done,
        some links may no longer have any corners on either side of them.
        Where this is the case, drop those links. Once this is done, there
        will be some patches that are no longer bound. Where this is the
        case, drop those patches.

        Parameters
        ----------
        corners : array of int
            Corners to drop from the graph.
        """
        # Remove the corners
        corners_to_drop = np.asarray(corners, dtype=int)
        self.drop_element(corners_to_drop, at="corner")

        # Remove bad links
        is_a_link = np.any(self._mesh["corners_at_face"].values != -1, axis=1)
        self.drop_element(np.where(~is_a_link)[0], at="link")

        # Remove the bad patches
        is_a_patch = np.all(self._mesh["links_at_patch"] >= 0, axis=1)
        self.drop_element(np.where(~is_a_patch)[0], at="patch")

    def drop_perimeter_faces(self):
        """Drop perimeter faces from a graph."""
        self.drop_element(np.where(self.is_perimeter_face())[0], at="face")

    def drop_perimeter_cells(self):
        """Drop perimeter cells from a graph."""
        self.drop_element(np.where(self.is_perimeter_cell())[0], at="cell")

    def drop_element(self, ids, at="node"):
        """Drop elements, and their associated elements, from a graph.

        Parameters
        ----------
        ids : ndarray of int
            Ids of the elements.
        at : str, optional
            Name of the element to drop.
        """
        dropped_ids = np.asarray(ids, dtype=int)
        dropped_ids.sort()
        is_a_keeper = np.full(self._mesh.dims[at], True)
        is_a_keeper[dropped_ids] = False

        if at == "patch":
            prefix = re.compile("^{at}(es)?_at_".format(at=at))
        else:
            prefix = re.compile("^{at}(s)?_at_".format(at=at))
        suffix = re.compile("at_{at}$".format(at=at))

        at_ = {}
        if at in self._mesh.coords:
            x = self._mesh["x_of_{at}".format(at=at)].values[is_a_keeper]
            y = self._mesh["y_of_{at}".format(at=at)].values[is_a_keeper]
            data = np.arange(len(x))

            at_[at] = xr.DataArray(
                data=data,
                coords={
                    "x_of_{at}".format(at=at): xr.DataArray(x, dims=(at,)),
                    "y_of_{at}".format(at=at): xr.DataArray(y, dims=(at,)),
                },
                dims=(at,),
            )
            self._mesh = self._mesh.drop(
                ["x_of_{at}".format(at=at), "y_of_{at}".format(at=at)]
            )

        for name, var in self._mesh.variables.items():
            if suffix.search(name):
                at_[name] = xr.DataArray(var.values[is_a_keeper], dims=var.dims)
        self._mesh = self._mesh.drop(list(at_))
        self._mesh.update(at_)

        for name, var in self._mesh.variables.items():
            if prefix.search(name):
                array = var.values.reshape((-1,))
                array[np.in1d(array, dropped_ids)] = -1
                for id_ in dropped_ids[::-1]:
                    array[array > id_] -= 1

    @property
    def links_at_patch(self):
        """Links that form a patch."""
        return self._mesh["links_at_patch"].values

    @property
    def node_at_cell(self):
        """Node contained by a cell."""
        return self._mesh["node_at_cell"].values

    @property
    def faces_at_cell(self):
        """Faces that form a cell."""
        return self._mesh["faces_at_cell"].values
