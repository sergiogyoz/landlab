import numpy as np
import pytest
from scipy.spatial import Delaunay, Voronoi

from ..voronoi_to_graph import VoronoiDelaunay, VoronoiDelaunayToGraph


@pytest.fixture
def xy_3x3_hex():
    return [
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.5, 1.0],
        [1.5, 1.0],
        [2.5, 1.0],
        [0.0, 2.0],
        [1.0, 2.0],
        [2.0, 2.0],
    ]


@pytest.fixture
def hex_graph():
    """
    Create this hex graph::

        6 - 7 - 8
         \ / \ / \
          3 - 4 - 5
         / \ / \ /
        0 - 1 - 2
    """
    xy = [
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.5, 1.0],
        [1.5, 1.0],
        [2.5, 1.0],
        [0.0, 2.0],
        [1.0, 2.0],
        [2.0, 2.0],
    ]
    perimeter_links = np.array(
        [[0, 1], [1, 2], [2, 5], [5, 8], [8, 7], [7, 6], [6, 3], [3, 0]], dtype=int
    )
    return VoronoiDelaunayToGraph(xy, perimeter_links=perimeter_links)


def test_voronoi_delaunay_elements(xy_3x3_hex):
    """Test element mapping from scipy Voronoi to landlab Graph"""
    voronoi = Voronoi(xy_3x3_hex)
    delaunay = Delaunay(xy_3x3_hex)
    graph = VoronoiDelaunay(xy_3x3_hex)

    assert np.all(graph.x_of_node == voronoi.points[:, 0])
    assert np.all(graph.y_of_node == voronoi.points[:, 1])

    assert np.all(graph.x_of_corner == voronoi.vertices[:, 0])
    assert np.all(graph.y_of_corner == voronoi.vertices[:, 1])

    assert np.all(graph.corners_at_face == voronoi.ridge_vertices)

    assert np.all(graph.nodes_at_face == voronoi.ridge_points)
    assert np.all(graph.cell_at_node == voronoi.point_region)

    assert len(graph.corners_at_cell) == len(voronoi.regions)
    for cell, corners in enumerate(voronoi.regions):
        assert graph.n_corners_at_cell[cell] == len(corners)
        assert np.all(graph.corners_at_cell[cell, : len(corners)] == corners)
        assert np.all(graph.corners_at_cell[cell, len(corners) :] == -1)

    assert np.all(graph.nodes_at_patch == delaunay.simplices)


def test_voronoi_delaunay_elements_are_int(hex_graph):
    """Test that element ids are int"""
    assert hex_graph.corners_at_face.dtype == np.int
    assert hex_graph.nodes_at_face.dtype == np.int
    assert hex_graph.cell_at_node.dtype == np.int
    assert hex_graph.corners_at_cell.dtype == np.int
    assert hex_graph.nodes_at_patch.dtype == np.int


def test_voronoi_delaunay_coords_are_float(hex_graph):
    """Test that element coordinates are float"""
    assert hex_graph.x_of_node.dtype == np.float
    assert hex_graph.y_of_node.dtype == np.float
    assert hex_graph.x_of_corner.dtype == np.float
    assert hex_graph.y_of_corner.dtype == np.float


def test_graph_number_of_elements(hex_graph):
    """Test the number of elements"""
    assert len(hex_graph.x_of_node) == 9
    assert len(hex_graph.y_of_node) == 9
    assert len(hex_graph.nodes_at_link) == 16
    assert len(hex_graph.links_at_patch) == 8

    assert len(hex_graph.x_of_corner) == 8
    assert len(hex_graph.y_of_corner) == 8
    assert len(hex_graph.corners_at_face) == 8
    assert len(hex_graph.nodes_at_face) == 8
    assert len(hex_graph.faces_at_cell) == 1


def test_graph_nodes_at_link(hex_graph):
    """Test nodes_at_link array"""
    sorted = np.lexsort((hex_graph.nodes_at_link[:, 1], hex_graph.nodes_at_link[:, 0]))
    assert np.all(
        hex_graph.nodes_at_link[sorted]
        == [
            [0, 3],
            [1, 0],
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 4],
            [3, 4],
            [3, 6],
            [4, 8],
            [5, 2],
            [5, 4],
            [5, 8],
            [7, 3],
            [7, 4],
            [7, 6],
            [7, 8],
        ]
    )


def test_graph_nodes_at_patch(hex_graph):
    """Test nodes_at_patch array"""
    sorted = np.lexsort(
        (
            hex_graph.nodes_at_patch[:, 2],
            hex_graph.nodes_at_patch[:, 1],
            hex_graph.nodes_at_patch[:, 0],
        )
    )

    assert np.all(
        hex_graph.nodes_at_patch[sorted]
        == [
            [1, 2, 4],
            [1, 3, 0],
            [3, 1, 4],
            [3, 7, 6],
            [4, 2, 5],
            [7, 3, 4],
            [8, 4, 5],
            [8, 7, 4],
        ]
    )


def test_graph_nodes_at_face(hex_graph):
    """Test nodes_at_face array"""
    sorted = np.lexsort(
        (
            hex_graph.nodes_at_face[:, 1],
            hex_graph.nodes_at_face[:, 0],
        )
    )

    assert np.all(
        hex_graph.nodes_at_face[sorted]
        == [
            [1, 3],
            [1, 4],
            [2, 4],
            [3, 4],
            [4, 8],
            [5, 4],
            [7, 3],
            [7, 4],
        ]
    )


def test_graph_cell_at_node(hex_graph):
    """Test cell_at_node array"""
    assert np.all(hex_graph.cell_at_node == [-1, -1, -1, -1, 0, -1, -1, -1, -1])


def test_graph_node_at_cell(hex_graph):
    """Test node_at_cell array"""
    assert np.all(hex_graph.node_at_cell == [4])
