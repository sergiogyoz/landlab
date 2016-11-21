import numpy as np

# from .layers import Layers, LayerFields
# from .layergridstack import LayerGridStack
from .layerstack import LayerStack


class LayerPointStack(LayerStack):

    """A 1D stack of layers piled on top of one another.

    Parameters
    ----------
    n_grains : int
        Number of grain types to track.
    z0 : float
        Elevation to the base of the stack.

    Examples
    --------
    >>> from landlab.layers import LayerPointStack
    >>> layers = LayerPointStack(z0=2.)
    >>> layers.base
    2.0
    >>> layers.top
    2.0
    >>> layers.z
    array([ 0.])

    >>> layers.add(2.5)
    >>> layers.base
    2.0
    >>> layers.top
    4.5
    >>> layers.z
    array([ 0. ,  2.5])

    >>> layers = LayerPointStack(fields=('age', ))
    >>> layers.age
    array([], dtype=float64)
    >>> layers.add(1.5, age=1.)
    >>> layers.age
    array([ 1.])
    >>> layers.z
    array([ 0. ,  1.5])

    >>> layers.add(2., age=2.)
    >>> layers.age
    array([ 1.,  2.])
    >>> layers.z
    array([ 0. ,  1.5,  3.5])
    """

    def __init__(self, n_grains=1, z0=0., dz=1., **kwds):
        self._z = np.arange(4, dtype=float) * dz
        self._z0 = z0
        self._top = 0

        super(LayerPointStack, self).__init__(n_grains=n_grains, **kwds)
