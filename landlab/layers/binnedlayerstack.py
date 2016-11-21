import numpy as np
import bisect

from .layers import Layers, LayerFields


class BinnedLayerFields(LayerFields):

    def _add_field(self, name, **kwds):
        if name not in self._fields:
            self._fields[name] = np.empty(self.allocated, **kwds)

    def add(self, dz, **kwds):
        """Add properties to the top of a stack.

        Parameters
        ----------
        dz : float
            The amount of sediment to add to the stack.
        """
        bins = self.extract(self.thickness - dz)
        n_bins = len(bins)
        for name, val in kwds.items():
            val = np.asarray(val)
            array = getattr(self, name)[-(n_bins - 1):]
            array[1:] = val

            array[0] = (
                (bins[1] - bins[0]) * val +
                (bins[0] - self.z[-n_bins]) * array[0]
            ) / (bins[1] - self.z[-n_bins])


class BinnedLayerStack(Layers, BinnedLayerFields):

    """A stack of layers piled on top of one another.

    Parameters
    ----------
    n_grains : int
        Number of grain types to track.
    z0 : float
        Elevation to the base of the stack.
    dz : float
        Thickness of new layer bins.

    Examples
    --------
    >>> from landlab.layers import BinnedLayerStack
    >>> layers = BinnedLayerStack(z0=2.)
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
    array([ 0. ,  1. ,  2. ,  2.5])

    >>> layers = BinnedLayerStack(fields=('age', ))
    >>> layers.age
    array([], dtype=float64)
    >>> layers.add(1.5, age=1.)
    >>> layers.age
    array([ 1.,  1.])
    >>> layers.z
    array([ 0. ,  1. ,  1.5])

    >>> layers.add(2., age=2.)
    >>> layers.age
    array([ 1. ,  1.5,  2. ,  2. ])
    >>> layers.z
    array([ 0. ,  1. ,  2. ,  3. ,  3.5])
    """

    def __init__(self, n_grains=1, z0=0., dz=1., **kwds):
        self._z = np.arange(4, dtype=float) * dz
        self._z0 = z0
        self._dz = float(dz)
        self._top = 0

        super(BinnedLayerStack, self).__init__(n_grains=n_grains, **kwds)

    @property
    def z(self):
        """Elevation to bottom of each layer.
        
        Examples
        --------
        >>> from landlab.layers import BinnedLayerStack
        >>> layers = BinnedLayerStack(z0=3.)
        >>> layers.z
        array([ 0.])
        >>> layers.add(1.5)
        >>> layers.z
        array([ 0. ,  1. ,  1.5])
        """
        return self._z[:self._top + 1]

    @property
    def dz(self):
        """Thickness of new bins.
        
        Examples
        --------
        >>> from landlab.layers import BinnedLayerStack
        >>> layers = BinnedLayerStack()
        >>> layers.dz
        1.0

        >>> layers = BinnedLayerStack(dz=2)
        >>> layers.dz
        2.0
        """
        return self._dz

    @property
    def nlayers(self):
        return self._top

    @property
    def allocated(self):
        return self._z.size

    def resize(self, newsize):
        newsize = int(newsize)
        if newsize < self.allocated:
            return

        new_allocated = (newsize >> 3) + 6 + newsize

        new_z = np.arange(0., new_allocated * self.dz, self.dz)
        new_z[:self.allocated] = self._z
        self._z = new_z

        super(BinnedLayerStack, self).resize(newsize)

    def add(self, dz, **kwds):
        """Add sediment to a column.

        Parameters
        ----------
        dz : float
            Amount of sediment to add.
        """
        if dz < 0:
            return self.remove(- dz)

        fill_to = self._z[self._top] + dz

        if not self.is_empty():
            self._z[self._top] = self._z[self._top - 1] + self._dz

        if fill_to > self._z[-1]:
            self.resize(self.allocated +
                        (fill_to - self._z[-1]) / self._dz + 1)

        new_top = bisect.bisect_left(self._z[self._top:], fill_to) + self._top

        self._z[new_top] = fill_to
        self._top = new_top

        super(BinnedLayerStack, self).add(dz, **kwds)

    def remove(self, dz):
        """Remove sediment from the top of a column.

        Parameters
        ----------
        dz : float
            Amount of sediment to remove.
        """
        if dz < 0:
            return self.add(- dz)

        new_z = self._z[self._top] - dz
        if new_z < 0.:
            new_z, new_top = 0., 0
        else:
            new_top = bisect.bisect_left(self._z[:self._top], new_z)
        # self._z[self._top:] *= self._dz
        self._z[self._top] = self._z[self._top - 1] + self._dz
        self._top = new_top
        self._z[self._top] = new_z
