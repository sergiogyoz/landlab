import numpy as np
cimport numpy as np
cimport cython

from libc.stdlib cimport malloc, free


ctypedef np.uint8_t uint8



cdef _count_sorted_blocks(long *array, long len, long stride, long *count, long n_values):
    cdef int i
    cdef int value

    i = 0
    for value in range(n_values):
        count[value] = 0
        while i < len * stride and value == array[i]:
            count[value] += 1
            i += stride


cdef _offset_to_sorted_blocks(long *array, long len, long stride, long *offset, long n_values):
    cdef long i
    cdef long value
    cdef long block
    cdef long last_count, this_count

    _count_sorted_blocks(array, len, stride, offset, n_values)

    last_count = offset[0]
    offset[0] = 0
    for block in range(1, n_values):
        this_count = offset[block]
        offset[block] = offset[block - 1] + last_count
        last_count = this_count


@cython.boundscheck(False)
@cython.wraparound(False)
def offset_to_sorted_block(
    np.ndarray[long, ndim=2, mode="c"] sorted_ids not None,
    np.ndarray[long, ndim=1, mode="c"] offset_to_block not None,
):
    cdef long n_ids = sorted_ids.shape[0]
    cdef long n_blocks = offset_to_block.shape[0]

    _offset_to_sorted_blocks(
        &sorted_ids[0, 0],
        n_ids,
        sorted_ids.shape[1],
        &offset_to_block[0],
        n_blocks,
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def pair_isin_sorted_list(
    np.ndarray[long, ndim=2, mode="c"] src_pairs not None,
    np.ndarray[long, ndim=2, mode="c"] pairs not None,
    np.ndarray[uint8, ndim=1, mode="c"] out not None,
):
    cdef long n
    cdef long pair
    cdef long n_pairs = out.shape[0]
    cdef long n_values = src_pairs.shape[0]
    cdef long *data = <long *>malloc(n_values * sizeof(long))
    cdef SparseMatrixInt mat

    for n in range(n_values):
        data[n] = 1
    try:
        mat = sparse_matrix_alloc_with_tuple(&src_pairs[0, 0], data, n_values, 0)
        for pair in range(n_pairs):
            out[pair] = sparse_matrix_get_or_transpose(mat, pairs[pair, 0], pairs[pair, 1])
    finally:
        free(data)


@cython.boundscheck(False)
@cython.wraparound(False)
def map_pairs_to_values(
    np.ndarray[long, ndim=2, mode="c"] src_pairs not None,
    np.ndarray[long, ndim=1, mode="c"] data not None,
    np.ndarray[long, ndim=2, mode="c"] pairs not None,
    np.ndarray[long, ndim=1, mode="c"] out not None,
):
    cdef long pair
    cdef long n_pairs = out.shape[0]
    cdef long n_values = data.shape[0]
    cdef long val
    cdef SparseMatrixInt mat

    mat = sparse_matrix_alloc_with_tuple(&src_pairs[0, 0], &data[0], n_values, -1)

    for pair in range(n_pairs):
        out[pair] = sparse_matrix_get_or_transpose(mat, pairs[pair, 0], pairs[pair, 1])


@cython.boundscheck(False)
@cython.wraparound(False)
def map_rolling_pairs_to_values(
    np.ndarray[long, ndim=2, mode="c"] src_pairs not None,
    np.ndarray[long, ndim=1, mode="c"] data not None,
    np.ndarray[long, ndim=2, mode="c"] pairs not None,
    np.ndarray[long, ndim=2, mode="c"] out not None,
):
    cdef long n_values = data.shape[0]
    cdef long n_pairs = pairs.shape[0]
    cdef long pair
    cdef SparseMatrixInt mat

    mat = sparse_matrix_alloc_with_tuple(&src_pairs[0, 0], &data[0], n_values, -1)

    for pair in range(n_pairs):
        _map_rolling_pairs(mat, &pairs[pair, 0], &out[pair, 0], pairs.shape[1])


cdef _map_rolling_pairs(SparseMatrixInt mat, long *pairs, long *out, long size):
    cdef long n
    cdef long val

    for n in range(size - 1):
        out[n] = sparse_matrix_get_or_transpose(mat, pairs[n], pairs[n + 1])

    n = size - 1
    out[n] = sparse_matrix_get_or_transpose(mat, pairs[n], pairs[0])


cdef struct SparseMatrixInt:
    long *values
    long *offset_to_row
    long *col
    long col_start
    long col_stride
    long n_rows
    long n_cols
    long no_val


cdef SparseMatrixInt sparse_matrix_alloc_with_data(
    long n_rows,
    long n_cols,
    long *rows,
    long *cols,
    long *values,
    long n_values,
    long no_val,
):
    cdef SparseMatrixInt mat
    cdef long *offset = <long *>malloc((n_rows + 1) * sizeof(long))

    _offset_to_sorted_blocks(rows, n_values, 1, offset, n_rows)

    mat.values = values
    mat.offset_to_row = offset
    mat.col = cols
    mat.col_start = 0
    mat.col_stride = 1
    mat.n_rows = n_rows
    mat.n_cols = n_cols
    mat.no_val = no_val

    return mat

cdef SparseMatrixInt sparse_matrix_alloc_with_tuple(
    long *rows_and_cols,
    long *values,
    long n_values,
    long no_val,
):
    cdef long n_rows
    cdef long n_cols
    cdef long i
    cdef SparseMatrixInt mat
    cdef long *offset

    n_rows = 0
    n_cols = 0
    for i in range(0, n_values * 2, 2):
        if rows_and_cols[i] > n_rows:
            n_rows = rows_and_cols[i]
        if rows_and_cols[i + 1] > n_cols:
            n_cols = rows_and_cols[i + 1]
    n_rows += 1
    n_cols += 1
    offset = <long *>malloc((n_rows + 1) * sizeof(long))

    _offset_to_sorted_blocks(rows_and_cols, n_values, 2, offset, n_rows + 1)

    mat.values = values
    mat.offset_to_row = offset
    mat.col = rows_and_cols
    mat.col_start = 1
    mat.col_stride = 2
    mat.n_rows = n_rows
    mat.n_cols = n_cols
    mat.no_val = no_val

    return mat


cdef sparse_matrix_free(SparseMatrixInt mat):
    free(mat.offset_to_row)


cdef long sparse_matrix_get_or_transpose(SparseMatrixInt mat, long row, long col):
    cdef long val
    val = sparse_matrix_get(mat, row, col)
    if val == mat.no_val:
        val = sparse_matrix_get(mat, col, row)
    return val


cdef long sparse_matrix_get(SparseMatrixInt mat, long row, long col):
    cdef long start
    cdef long stop
    cdef long n
    cdef long i

    if row < 0:
        return mat.no_val
    elif row >= mat.n_rows:
        return mat.no_val
    elif col < 0:
        return mat.no_val
    elif col >= mat.n_cols:
        return mat.no_val

    start = mat.offset_to_row[row]
    stop = mat.offset_to_row[row + 1]

    i = mat.col_start + start * mat.col_stride
    for n in range(start, stop):
        if mat.col[i] == col:
            return mat.values[n]
        i += mat.col_stride

    return mat.no_val
