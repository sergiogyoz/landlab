import numpy as np

from .layerstack import LayerStack


class LayerGridStack(LayerStack):

    """A stack of 2D grid layers stacked on top of one another.

    Examples
    --------
    >>> from landlab.layers import LayerGridStack
    >>> layers = LayerGridStack(z0=[0, 1, 2, 3])
    >>> layers.base
    array([ 0.,  1.,  2.,  3.])
    >>> layers.top
    array([ 0.,  1.,  2.,  3.])
    >>> layers.z
    array([[ 0.,  0.,  0.,  0.]])

    >>> layers.add(2.5)
    >>> layers.base
    array([ 0.,  1.,  2.,  3.])
    >>> layers.top
    array([ 2.5,  3.5,  4.5,  5.5])
    >>> layers.z
    array([[ 0. ,  0. ,  0. ,  0. ],
           [ 2.5,  2.5,  2.5,  2.5]])

    >>> layers = LayerGridStack(z0=[0, 1, 2, 3], fields=('age', ))
    >>> layers.age
    array([], shape=(0, 4), dtype=float64)
    >>> layers.add(1.5, age=1.)
    >>> layers.age
    array([[ 1.,  1.,  1.,  1.]])
    >>> layers.z
    array([[ 0. ,  0. ,  0. ,  0. ],
           [ 1.5,  1.5,  1.5,  1.5]])

    >>> layers.add(2., age=2.)
    >>> layers.age
    array([[ 1.,  1.,  1.,  1.],
           [ 2.,  2.,  2.,  2.]])
    >>> layers.z
    array([[ 0. ,  0. ,  0. ,  0. ],
           [ 1.5,  1.5,  1.5,  1.5],
           [ 3.5,  3.5,  3.5,  3.5]])
    """

    def __init__(self, n_grains=1, z0=0., dz=1., **kwds):
        self._z0 = np.asarray(z0, dtype=float)
        self._z = np.empty((4, ) + self._z0.shape, dtype=float)
        self._top = 0

        self._z[self._top] = 0.

        super(LayerGridStack, self).__init__(n_grains=n_grains, **kwds)
