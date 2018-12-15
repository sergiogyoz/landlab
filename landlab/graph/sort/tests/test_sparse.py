import numpy as np
from numpy.testing import assert_array_equal

from landlab.graph.sort.sort import (
    map_sorted_pairs,
    map_sorted_rolling_pairs,
    pair_isin_sorted_list,
)
from landlab.graph.sort.ext.sparse import offset_to_sorted_block


def test_pair_isin_same_as_slow_way():
    """Test the fast way gives same result as the slow way."""
    n_src_pairs = 2 ** 18
    src = np.random.randint(n_src_pairs * 2, size=n_src_pairs * 2, dtype=int).reshape(
        (-1, 2)
    )
    src = src[np.lexsort((src[:, 1], src[:, 0]))]

    pairs = np.random.randint(n_src_pairs * 2, size=n_src_pairs * 2, dtype=int).reshape(
        (-1, 2)
    )

    out = pair_isin_sorted_list(src, pairs)

    def _slow_way(src, pairs):
        out = np.empty(len(pairs), dtype=bool)
        src_pairs = set()
        for pair in src:
            src_pairs.add(tuple(pair))
            src_pairs.add(tuple(pair[::-1]))
        for n, pair in enumerate(pairs):
            if tuple(pair) in src_pairs:
                out[n] = True
            else:
                out[n] = False
        return out

    slow_out = _slow_way(src, pairs)

    assert np.all(slow_out == out)

    ids = np.random.randint(n_src_pairs, size=n_src_pairs // 10)

    out = pair_isin_sorted_list(src, src[ids])
    assert np.all(out)

    out = _slow_way(src, src[ids])
    assert np.all(out)


def test_pair_isin():
    src = np.asarray([[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]], dtype=np.int)

    pairs = np.asarray([[1, 1], [3, 1], [1, 2], [5, 1]], dtype=int)
    out = pair_isin_sorted_list(src, pairs)

    assert np.all(out == [True, True, True, False])


def test_pair_isin_bigger():
    src = np.asarray(
        [[0, 1], [1, 2], [2, 5], [5, 8], [8, 7], [7, 6], [6, 3], [3, 0]], dtype=np.int
    )
    sorted = np.argsort(src[:, 0])
    pairs = np.asarray(
        [
            [5, 2],
            [5, 4],
            [5, 8],
            [1, 2],
            [1, 4],
            [1, 0],
            [1, 3],
            [2, 4],
            [0, 6],
            [0, 3],
            [7, 3],
            [7, 4],
            [7, 6],
            [7, 8],
            [3, 6],
            [3, 4],
            [4, 8],
        ],
        dtype=int,
    )
    out = pair_isin_sorted_list(src[sorted], pairs)

    # assert np.all(
    assert_array_equal(
        out,
        [
            True,
            False,
            True,
            True,
            False,
            True,
            False,
            False,
            False,
            True,
            False,
            False,
            True,
            True,
            True,
            False,
            False,
        ],
    )


def test_pair_isin_with_out_keyword():
    """Test using the out keyword."""
    src = np.asarray([[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]], dtype=np.int)

    pairs = np.asarray([[1, 1], [3, 1], [1, 2], [5, 1]], dtype=int)
    out = np.empty(len(pairs), dtype=bool)
    rtn = pair_isin_sorted_list(src, pairs, out=out)

    assert rtn is out
    assert np.all(out == [True, True, True, False])


def test_pair_isin_with_sorter_keyword():
    """Test using the out keyword."""
    src = np.asarray([[4, 1], [1, 1], [0, 1], [2, 1], [3, 1]], dtype=np.int)
    pairs = np.asarray([[1, 1], [3, 1], [1, 2], [5, 1]], dtype=int)

    out = pair_isin_sorted_list(src, pairs, sorter=np.argsort(src[:, 0]))

    assert np.all(out == [True, True, True, False])


def test_map_pairs():
    src = np.asarray([[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]], dtype=np.int)
    data = np.arange(len(src), dtype=int)

    pairs = np.asarray([[1, 1], [3, 1]], dtype=int)
    out = map_sorted_pairs(src, data, pairs)

    assert np.all(out == [1, 3])


def test_map_pairs_with_out_keyword():
    src = np.asarray([[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]], dtype=np.int)
    data = np.arange(len(src), dtype=int)

    pairs = np.asarray([[1, 1], [3, 1]], dtype=int)
    out = np.empty(len(pairs), dtype=np.int)
    rtn = map_sorted_pairs(src, data, pairs, out=out)

    assert rtn is out
    assert np.all(out == [1, 3])


def test_map_pairs_at_end():
    src = np.asarray([[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]], dtype=np.int)
    data = np.arange(len(src), dtype=int)

    pairs = np.asarray([[1, 1], [3, 1], [4, 1], [1, 4], [4, 2]], dtype=int)
    out = map_sorted_pairs(src, data, pairs)

    assert np.all(out == [1, 3, 4, 4, -1])


def test_map_pairs_transposed():
    src = np.asarray([[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]], dtype=np.int)
    data = np.arange(len(src), dtype=int)

    pairs = np.asarray([[1, 0], [1, 3]], dtype=int)
    out = map_sorted_pairs(src, data, pairs)

    assert np.all(out == [0, 3])


def test_map_pairs_all_missing():
    src = np.asarray([[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]], dtype=np.int)
    data = np.arange(len(src), dtype=int)

    pairs = np.asarray([[5, 1], [1, 42]], dtype=int)
    out = map_sorted_pairs(src, data, pairs)

    assert np.all(out == [-1, -1])


def test_map_pairs_with_sorter_keyword():
    src = np.asarray([[1, 1], [0, 1], [3, 1], [4, 1], [2, 1]], dtype=np.int)
    data = np.arange(len(src), dtype=int)
    sorted = [1, 0, 4, 2, 3]

    pairs = np.asarray([[1, 1], [3, 1]], dtype=int)
    out = map_sorted_pairs(src, data, pairs, sorter=sorted)

    assert np.all(out == [0, 2])


def test_map_pairs_big_data():
    n_src_pairs = 2 ** 20
    src = np.random.randint(n_src_pairs * 2, size=n_src_pairs * 2, dtype=int).reshape(
        (-1, 2)
    )
    src = src[np.lexsort((src[:, 1], src[:, 0]))]
    data = np.arange(n_src_pairs, dtype=int)

    ids = np.random.randint(n_src_pairs, size=n_src_pairs // 10)
    pairs = src[ids]
    out = map_sorted_pairs(src, data, pairs)

    # mismatch = np.where(out != ids)[0]
    # if mismatch:
    #     print(mismatch)
    #     print(out[mismatch])
    #     print(ids[mismatch])
    # assert_array_equal(out, ids)
    assert np.all(out == ids)


def test_map_rolling_pairs():
    src = np.asarray([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]], dtype=np.int)
    data = np.asarray([0, 1, 2, 3, 4], dtype=int)

    pairs = np.asarray([[0, 1, 2, 3], [0, 2, 3, 4]], dtype=int)
    out = map_sorted_rolling_pairs(src, data, pairs)

    assert np.all(out == [[0, 1, 2, -1], [-1, 2, 3, 4]])


def test_map_rolling_pairs_2():
    src = np.asarray(
        [
            [0, 6],
            [0, 3],
            [1, 2],
            [1, 4],
            [1, 0],
            [1, 3],
            [2, 4],
            [3, 6],
            [3, 4],
            [4, 8],
            [5, 4],
            [5, 8],
            [5, 2],
            [7, 3],
            [7, 4],
            [7, 6],
            [7, 8],
        ],
        dtype=int,
    )
    data = np.asarray(
        [8, 9, 3, 4, 5, 6, 7, 14, 15, 16, 1, 2, 0, 10, 11, 12, 13], dtype=int
    )
    pairs = np.asarray(
        [
            [4, 2, 5],
            [3, 6, 0],
            [8, 4, 5],
            [1, 2, 4],
            [1, 3, 0],
            [3, 1, 4],
            [7, 3, 4],
            [3, 7, 6],
            [8, 7, 4],
        ],
        dtype=int,
    )
    out = map_sorted_rolling_pairs(src, data, pairs)

    assert np.all(
        out
        == [
            [7, 0, 1],
            [14, 8, 9],
            [16, 1, 2],
            [3, 7, 4],
            [6, 9, 5],
            [6, 4, 15],
            [10, 15, 11],
            [10, 12, 14],
            [13, 11, 16],
        ]
    )


def test_map_rolling_pairs_with_out_keyword():
    src = np.asarray([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]], dtype=np.int)
    data = np.asarray([0, 1, 2, 3, 4], dtype=int)

    pairs = np.asarray([[0, 1, 2, 3], [0, 2, 3, 4]], dtype=int)
    out = np.empty_like(pairs)
    rtn = map_sorted_rolling_pairs(src, data, pairs, out=out)

    assert rtn is out
    assert np.all(out == [[0, 1, 2, -1], [-1, 2, 3, 4]])


def test_map_rolling_pairs_with_sorter_keyword():
    src = np.asarray([[0, 1], [2, 3], [4, 0], [1, 2], [3, 4]], dtype=np.int)
    data = np.asarray([0, 1, 2, 3, 4], dtype=int)
    sorted = [0, 3, 1, 4, 2]

    pairs = np.asarray([[0, 1, 2, 3], [0, 2, 3, 4]], dtype=int)
    out = map_sorted_rolling_pairs(src, data, pairs, sorter=sorted)

    assert np.all(out == [[0, 3, 1, -1], [-1, 1, 4, 2]])


def test_offset_2d():
    pairs = np.asarray([[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]], dtype=np.int)
    out = np.empty(len(pairs), dtype=int)
    offset_to_sorted_block(pairs, out)

    assert np.all(out == [0, 1, 2, 3, 4])


def test_offset_2d_with_missing_at_start():
    pairs = np.asarray([[1, 1], [2, 1], [3, 1], [4, 1]], dtype=np.int)
    out = np.empty(5, dtype=int)
    offset_to_sorted_block(pairs, out)

    assert np.all(out == [0, 0, 1, 2, 3])


def test_offset_2d_with_missing_in_middle():
    pairs = np.asarray([[0, 1], [1, 1], [3, 1], [4, 1]], dtype=np.int)
    out = np.empty(5, dtype=int)
    offset_to_sorted_block(pairs, out)

    assert np.all(out == [0, 1, 2, 2, 3])


def test_offset_with_different_strides():
    pairs = np.arange(5, dtype=int).reshape((-1, 1))
    out = np.empty(len(pairs), dtype=int)
    for _ in range(5):
        offset_to_sorted_block(pairs, out)
        assert np.all(out == [0, 1, 2, 3, 4])
        pairs = np.hstack((pairs, pairs))


def test_offset_with_sort_out():
    pairs = np.asarray([[1, 1], [2, 1], [3, 1], [4, 1]], dtype=np.int)
    out = np.empty(4, dtype=int)
    offset_to_sorted_block(pairs, out)

    assert np.all(out == [0, 0, 1, 2])


def test_offset_with_long_out():
    pairs = np.asarray([[1, 1], [2, 1], [3, 1], [4, 1]], dtype=np.int)
    out = np.empty(7, dtype=int)
    offset_to_sorted_block(pairs, out)

    assert np.all(out == [0, 0, 1, 2, 3, 4, 4])
