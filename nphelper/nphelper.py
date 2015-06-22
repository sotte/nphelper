from __future__ import division
import numpy as np
from numpy import atleast_2d, asanyarray, dot, empty, hstack, vstack, zeros
from numpy.linalg import LinAlgError
from numpy.core import double, intp, Inf
import copy


###############################################################################
def _assertRank2(*arrays):
    for a in arrays:
        if len(a.shape) != 2:
            raise LinAlgError('%d-dimensional array given. Array must be '
                              'two-dimensional' % len(a.shape))


def block(tup_tup):
    """
    Create block arrays similar to Matlab's "square bracket stacking":

            [A A; B B]

    You can create a block array with the same notation you use for
    `np.array`.

    Note: This function is hopefully going to be part of numpy 1.10.0

    Parameters
    ----------
    tup_tup : sequence of sequence of ndarrays
        1-D arrays are treated as row vectors.

    Returns
    -------
    stacked : ndarray
        The 2-D array formed by stacking the given arrays.

    See Also
    --------
    hstack : Stack arrays in sequence horizontally (column wise).
    vstack : Stack arrays in sequence vertically (row wise).
    dstack : Stack arrays in sequence depth wise (along third dimension).
    concatenate : Join a sequence of arrays together.
    vsplit : Split array into a list of multiple sub-arrays vertically.

    Examples
    --------
    Stacking in a row:
    >>> A = np.array([[1, 2, 3]])
    >>> B = np.array([[2, 3, 4]])
    >>> block([A, B])
    array([[1, 2, 3, 2, 3, 4]])

    >>> A = np.array([[1, 1], [1, 1]])
    >>> B = 2 * A
    >>> block([A, B])
    array([[1, 1, 2, 2],
           [1, 1, 2, 2]])

    >>> # the tuple notation also works
    >>> block((A, B))
    array([[1, 1, 2, 2],
           [1, 1, 2, 2]])

    >>> # block matrix with arbitrary shaped elements
    >>> One = np.array([[1, 1, 1]])
    >>> Two = np.array([[2, 2, 2]])
    >>> Three = np.array([[3, 3, 3, 3, 3, 3]])
    >>> four = np.array([4, 4, 4, 4, 4, 4])
    >>> five = np.array([5])
    >>> six = np.array([6, 6, 6, 6, 6])
    >>> Zeros = np.zeros((2, 6), dtype=int)
    >>> block([[One, Two],
    ...        [Three],
    ...        [four],
    ...        [five, six],
    ...        [Zeros]])
    array([[1, 1, 1, 2, 2, 2],
           [3, 3, 3, 3, 3, 3],
           [4, 4, 4, 4, 4, 4],
           [5, 6, 6, 6, 6, 6],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0]])

    >>> # 1-D vectors are treated as row arrays
    >>> a = np.array([1, 1])
    >>> b = np.array([2, 2])
    >>> block([a, b])
    array([[1, 1, 2, 2]])
    """
    if isinstance(tup_tup[0], list) or isinstance(tup_tup[0], tuple):
        result = vstack([hstack(row) for row in tup_tup])
    else:
        result = hstack(tup_tup)
    return atleast_2d(result)


def cartesian_product(arrays, out=None):
    """
    Generate the cartesian product [1]_ of the input arrays.

    `cartesian_product` is the numpy version of Python's `itertools.product`
    [2]_.

    Note: This function is hopefully going to be part of numpy 1.10.0

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray, optional
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (n, len(arrays)) containing the cartesian products
        formed of the input arrays where
        `n = prod(a.shape[0] for a in arrays)`.

    Raises
    ------
    ValueError
        If input is the wrong shape (the input must be a list of 1-D arrays.
    ValueError
        If input contains less than two arrays.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Cartesian_product
    .. [2] https://docs.python.org/3.4/library/itertools.html#itertools.product

    Examples
    --------
    >>> cartesian_product([[1, 2], [3, 4]])
    array([[1, 3],
           [1, 4],
           [2, 3],
           [2, 4]])

    >>> cartesian_product(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """
    if len(arrays) < 2:
        msg = "need at least two array to calculate the cartesian product"
        raise ValueError(msg)

    arrays = [np.asarray(a) for a in arrays]

    for a in arrays:
        if a.ndim != 1:
            raise ValueError("accepting only 1-D arrays")

    dtype = np.result_type(*arrays)
    n = np.prod([arr.size for arr in arrays])

    out = np.empty((len(arrays), n), dtype=dtype) if out is None else out.T

    for j, arr in enumerate(arrays):
        n //= arr.size
        out.shape = (len(arrays), -1, arr.size, n)
        out[j] = arr[np.newaxis, :, np.newaxis]
    out.shape = (len(arrays), -1)

    return out.T


###############################################################################
# multi_dot
def multi_dot(arrays):
    """
    Compute the dot product of two or more arrays in a single function call,
    while automatically selecting the fastest evaluation order.

    `multi_dot` chains `numpy.dot` and uses optimal parenthesization
    of the matrices [1]_ [2]_. Depending on the shapes of the matrices,
    this can speed up the multiplication a lot.

    If the first argument is 1-D it is treated as a row vector.
    If the last argument is 1-D it is treated as a column vector.
    The other arguments must be 2-D.

    Think of `multi_dot` as::

        def multi_dot(arrays): return functools.reduce(np.dot, arrays)


    Parameters
    ----------
    arrays : sequence of array_like
        If the first argument is 1-D it is treated as row vector.
        If the last argument is 1-D it is treated as column vector.
        The other arguments must be 2-D.

    Returns
    -------
    output : ndarray
        Returns the dot product of the supplied arrays.

    See Also
    --------
    dot : dot multiplication with two arguments.

    References
    ----------

    .. [1] Cormen, "Introduction to Algorithms", Chapter 15.2, p. 370-378
    .. [2] http://en.wikipedia.org/wiki/Matrix_chain_multiplication

    Examples
    --------
    `multi_dot` allows you to write::

    >>> from nphelper import multi_dot
    >>> # Prepare some data
    >>> A = np.random.random((10000, 100))
    >>> B = np.random.random((100, 1000))
    >>> C = np.random.random((1000, 5))
    >>> D = np.random.random((5, 333))
    >>> # the actual dot multiplication
    >>> multi_dot([A, B, C, D])  # doctest: +SKIP

    instead of::

    >>> np.dot(np.dot(np.dot(A, B), C), D)  # doctest: +SKIP
    >>> # or
    >>> A.dot(B).dot(C).dot(D)  # doctest: +SKIP


    Example: multiplication costs of different parenthesizations
    ------------------------------------------------------------

    The cost for a matrix multiplication can be calculated with the
    following function::

        def cost(A, B): return A.shape[0] * A.shape[1] * B.shape[1]

    Let's assume we have three matrices
    :math:`A_{10x100}, B_{100x5}, C_{5x50}$`.

    The costs for the two different parenthesizations are as follows::

        cost((AB)C) = 10*100*5 + 10*5*50   = 5000 + 2500   = 7500
        cost(A(BC)) = 10*100*50 + 100*5*50 = 50000 + 25000 = 75000

    """
    n = len(arrays)
    # optimization only makes sense for len(arrays) > 2
    if n < 2:
        raise ValueError("Expecting at least two arrays.")
    elif n == 2:
        return dot(arrays[0], arrays[1])

    arrays = [asanyarray(a) for a in arrays]

    # save original ndim to reshape the result array into the proper form later
    ndim_first, ndim_last = arrays[0].ndim, arrays[-1].ndim
    # Explicitly convert vectors to 2D arrays to keep the logic of the internal
    # _multi_dot_* functions as simple as possible.
    if arrays[0].ndim == 1:
        arrays[0] = atleast_2d(arrays[0])
    if arrays[-1].ndim == 1:
        arrays[-1] = atleast_2d(arrays[-1]).T
    _assertRank2(*arrays)

    # _multi_dot_three is much faster than _multi_dot_matrix_chain_order
    if n == 3:
        result = _multi_dot_three(arrays[0], arrays[1], arrays[2])
    else:
        order = _multi_dot_matrix_chain_order(arrays)
        result = _multi_dot(arrays, order, 0, n - 1)

    # return proper shape
    if ndim_first == 1 and ndim_last == 1:
        return result[0, 0]  # scalar
    elif ndim_first == 1 or ndim_last == 1:
        return result.ravel()  # 1-D
    else:
        return result


def _multi_dot_three(A, B, C):
    """
    Find the best order for three arrays and do the multiplication.

    For three arguments `_multi_dot_three` is approximately 15 times faster
    than `_multi_dot_matrix_chain_order`

    """
    # cost1 = cost((AB)C)
    cost1 = (A.shape[0] * A.shape[1] * B.shape[1] +  # (AB)
             A.shape[0] * B.shape[1] * C.shape[1])   # (--)C
    # cost2 = cost((AB)C)
    cost2 = (B.shape[0] * B.shape[1] * C.shape[1] +  # (BC)
             A.shape[0] * A.shape[1] * C.shape[1])   # A(--)

    if cost1 < cost2:
        return dot(dot(A, B), C)
    else:
        return dot(A, dot(B, C))


def _multi_dot_matrix_chain_order(arrays, return_costs=False):
    """
    Return a np.array that encodes the optimal order of mutiplications.

    The optimal order array is then used by `_multi_dot()` to do the
    multiplication.

    Also return the cost matrix if `return_costs` is `True`

    The implementation CLOSELY follows Cormen, "Introduction to Algorithms",
    Chapter 15.2, p. 370-378.  Note that Cormen uses 1-based indices.

        cost[i, j] = min([
            cost[prefix] + cost[suffix] + cost_mult(prefix, suffix)
            for k in range(i, j)])

    """
    n = len(arrays)
    # p stores the dimensions of the matrices
    # Example for p: A_{10x100}, B_{100x5}, C_{5x50} --> p = [10, 100, 5, 50]
    p = [a.shape[0] for a in arrays] + [arrays[-1].shape[1]]
    # m is a matrix of costs of the subproblems
    # m[i,j]: min number of scalar multiplications needed to compute A_{i..j}
    m = zeros((n, n), dtype=double)
    # s is the actual ordering
    # s[i, j] is the value of k at which we split the product A_i..A_j
    s = empty((n, n), dtype=intp)

    for l in range(1, n):
        for i in range(n - l):
            j = i + l
            m[i, j] = Inf
            for k in range(i, j):
                q = m[i, k] + m[k+1, j] + p[i]*p[k+1]*p[j+1]
                if q < m[i, j]:
                    m[i, j] = q
                    s[i, j] = k  # Note that Cormen uses 1-based index

    return (s, m) if return_costs else s


def _multi_dot(arrays, order, i, j):
    """Actually do the multiplication with the given order."""
    if i == j:
        return arrays[i]
    else:
        return dot(_multi_dot(arrays, order, i, order[i, j]),
                   _multi_dot(arrays, order, order[i, j] + 1, j))


###############################################################################
# along functions
def _npiterslice(array, axis):
    """Create slices of the  array along the given axis.

    return an 1D array with len darray.shape[axis]

    TODO add doc
    TODO should this be a public function?
    """
    slices = [slice(0, None)] * array.ndim
    for i in range(array.shape[axis]):
        slices_copy = copy.deepcopy(slices)
        slices_copy[axis] = slice(i, i+1)
        yield array[slices_copy]


def along(f, array, axis):
    """
    Aplly the given function along all slices of array

    TODO add doc

    >>> A = np.ones((2, 3, 4))
    >>> along(np.max, A, 0)
    array([ 1.,  1.])
    >>> along(np.max, A, 1)
    array([ 1.,  1.,  1.])
    >>> along(np.max, A, 2)
    array([ 1.,  1.,  1.,  1.])

    >>> along(np.sum, A, 0)  # 2 * 4 -> 12
    array([ 12.,  12.])
    >>> along(np.sum, A, 1)  # 2 * 4 -> 8
    array([ 8.,  8.,  8.])
    >>> along(np.sum, A, 2)  # 2 * 3 -> 3
    array([ 6.,  6.,  6.,  6.])
    """
    return np.array([f(a) for a in _npiterslice(array, axis)])


def maxalong(array, axis):
    return along(np.max, array, axis)


def minalong(array, axis):
    return along(np.min, array, axis)


def sumalong(array, axis):
    return along(np.sum, array, axis)


def meanlong(array, axis):
    return along(np.mean, array, axis)


def stdalong(array, axis):
    return along(np.std, array, axis)


def varalong(array, axis):
    return along(np.var, array, axis)
