from landlab import RasterModelGrid, HexModelGrid

grid = RasterModelGrid((4, 5))

grid.status_at_node[6] = grid.BC_NODE_IS_CLOSED
grid.active_adjacent_nodes_at_node[(-1, 6, 2), ]