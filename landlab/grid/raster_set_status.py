def set_status_at_node_on_edges(grid, right=None, top=None, left=None,
                                bottom=None):
    """Set node status on grid edges.

    Parameters
    ----------
    grid : RasterModelGrid
        A grid.
    right : int, optional
        Node status along right edge.
    top : int, optional
        Node status along top edge.
    left : int, optional
        Node status along left edge.
    bottom : int, optional
        Node status along bottom edge.

    Examples
    --------
    >>> from landlab import RasterModelGrid, CLOSED_BOUNDARY

    >>> grid = RasterModelGrid((3, 4))
    >>> grid.status_at_node # doctest: +NORMALIZE_WHITESPACE
    array([1, 1, 1, 1,
           1, 0, 0, 1,
           1, 1, 1, 1], dtype=int8)

    >>> grid.set_status_at_node_on_edges(right=CLOSED_BOUNDARY)
    >>> grid.status_at_node # doctest: +NORMALIZE_WHITESPACE
    array([1, 1, 1, 4,
           1, 0, 0, 4,
           1, 1, 1, 4], dtype=int8)

    >>> from landlab import FIXED_GRADIENT_BOUNDARY
    >>> grid = RasterModelGrid((3, 4))

    The status of a corner is set along with its clockwise edge. That is,
    if setting the status for the top and right edges, the upper-right corner
    has the status of the right edge.

    >>> grid.set_status_at_node_on_edges(top=CLOSED_BOUNDARY,
    ...     right=FIXED_GRADIENT_BOUNDARY)
    >>> grid.status_at_node # doctest: +NORMALIZE_WHITESPACE
    array([1, 1, 1, 2,
           1, 0, 0, 2,
           4, 4, 4, 2], dtype=int8)

    In the above example, if you wanted the corner to have the status of the
    top edge, you need to make two calls to `set_status_at_node_on_edges`,

    >>> grid = RasterModelGrid((3, 4))
    >>> grid.set_status_at_node_on_edges(right=FIXED_GRADIENT_BOUNDARY)
    >>> grid.set_status_at_node_on_edges(top=CLOSED_BOUNDARY)
    >>> grid.status_at_node # doctest: +NORMALIZE_WHITESPACE
    array([1, 1, 1, 2,
           1, 0, 0, 2,
           4, 4, 4, 4], dtype=int8)

    An example that sets all of the edges shows how corners are set.

    >>> grid.set_status_at_node_on_edges(right=1, top=2, left=1, bottom=4)
    >>> grid.status_at_node # doctest: +NORMALIZE_WHITESPACE
    array([1, 4, 4, 4,
           1, 0, 0, 1,
           2, 2, 2, 1], dtype=int8)

    This method cannot be used to set TRACKS_CELL_BOUNDARY conditions (3),
    as more information is required (which nodes are tracked?). Use
    `grid.set_looped_boundaries` instead.

    >>> from landlab import TRACKS_CELL_BOUNDARY
    >>> grid.set_status_at_node_on_edges(
    ...     left=TRACKS_CELL_BOUNDARY)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    NotImplementedError: ...
    """
    status_at_edge = (('bottom', bottom), ('left', left), ('top', top),
                      ('right', right), )

    for edge, val in status_at_edge:
        if val is not None:
            nodes = grid.nodes_at_edge(edge)
            grid.status_at_node[nodes] = val

    if right is not None and bottom is not None:
        lr = grid.nodes_at_right_edge[0]
        grid.status_at_node[lr] = bottom
