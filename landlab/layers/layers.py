"""Base class for different types of layers and layered fields."""

import numpy as np
import bisect


class LayerFields(object):

    """Attach data fields to layers."""

    def __new__(cls, *args, **kwds):
        for field in ('f', ) + kwds.get('fields', ()):
            setattr(cls, field, property(lambda x, field=field: x[field],
                                         doc=field))

        return object.__new__(cls)

    def __init__(self, *args, **kwds):
        self._fields = dict()
        for field in kwds.get('fields', ()):
            self._add_field(field)

        self._fields['f'] = np.empty((self.allocated,
                                      kwds.get('n_grains', 1) - 1))

    def _add_field(self, name, **kwds):
        if name not in self._fields:
            try:
                self._z0.shape
            except AttributeError:
                self._fields[name] = np.empty(self.allocated, **kwds)
            else:
                self._fields[name] = np.empty((self.allocated, ) + self._z0.shape, **kwds)

    def add(self, dz, **kwds):
        """Add properties to the top of a stack.

        Parameters
        ----------
        dz : float
            The amount of sediment to add to the stack.
        """
        # if not kwds.contains(self._fields):
        #     pass
        # bins = self.extract(self.thickness - dz)
        # n_bins = len(bins)
        for name, val in kwds.items():
            val = np.asarray(val)
            array = getattr(self, name)
            array[-1] = val

    def resize(self, *args, **kwds):
        """Resize field arrays."""
        for name, array in self._fields.items():
            self._fields[name] = np.resize(array, self.allocated)

    def reduce(self, dz, name):
        bin_dz = np.diff(self.extract(self.thickness - dz))
        return np.sum(getattr(self, name)[-len(bin_dz):] * bin_dz / dz)

    @property
    def fields(self):
        """Names of fields tracked."""
        return self._fields.keys()

    def __getitem__(self, name):
        return self._fields[name][:self._top]


class Layers(object):

    @property
    def base(self):
        """Elevation of the bottom of the column.
        
        Examples
        --------
        >>> from landlab.layers import BinnedLayerStack
        >>> layers = BinnedLayerStack()
        >>> layers.base
        0.0
        >>> layers.base += 2.
        >>> layers.base
        2.0
        """
        return self._z0

    @base.setter
    def base(self, new_base):
        self._z0 = new_base

    @property
    def thickness(self):
        return self.z[-1]

    @property
    def top(self):
        """Elevation of the top of the column.
        
        Examples
        --------
        >>> from landlab.layers import BinnedLayerStack
        >>> layers = BinnedLayerStack()
        >>> layers.base, layers.top
        (0.0, 0.0)
        >>> layers.top = 2.
        >>> layers.base, layers.top
        (2.0, 2.0)
        """
        return self.base + self._z[self._top]

    @top.setter
    def top(self, new_top):
        self._z0 += new_top - self.top

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
        try:
            self._base.shape
        except AttributeError:
            return self._z[:self._top + 1]
        else:
            return self._z[:self._top + 1].reshape((-1, ) + self.base.shape)

    @property
    def nlayers(self):
        return self._top

    @property
    def size(self):
        return self._top + 1

    @property
    def allocated(self):
        # return self._z.size
        return self._z.shape[0]

    def resize(self, newsize):
        newsize = int(newsize)
        if newsize < self.allocated:
            return

        new_allocated = (newsize >> 3) + 6 + newsize

        new_z = np.empty_like(self._z, (new_allocated, ) + self._z.shape[1:])
        new_z[:self.allocated] = self._z
        self._z = new_z

        super(Layers, self).resize(newsize)

    def is_empty(self):
        """Check if the stack has any layers.
        
        Examples
        --------
        >>> from landlab.layers import BinnedLayerStack
        >>> layers = BinnedLayerStack()
        >>> layers.is_empty()
        True

        >>> layers.add(.1)
        >>> layers.is_empty()
        False
        """
        return self._top == 0

    def lift(self, dz):
        """Lift the base of the stack."""
        self.base += dz

    def lower(self, dz):
        """Lower the base of the stack."""
        self.lift(- dz)

    def layer_at(self, z, lower=False):
        """Find the layer containing a particular elevation.

        Parameters
        ----------
        z : float
            Elevation as measured from the bottom of the stack.

        Returns
        -------
        int
            Layer number that contains the elevation.
        """
        if z < 0. or z > self.thickness:
            raise ValueError('elevation is outside the column')
        if lower:
            return bisect.bisect_left(self._z[:self._top + 1], z) - 1
        else:
            return bisect.bisect_right(self._z[:self._top + 1], z) - 1

    def extract(self, start=0, stop=None):
        """Extract layers from a column.

        Parameters
        ----------
        start : float, optional
            Starting elevation in stack.
        stop : float, optional
            Stopping elevation in stack.

        Examples
        --------
        >>> from landlab.layers import LayerPointStack
        >>> layers = LayerPointStack()
        >>> layers.add(5.5)
        >>> layers.z
        array([ 0. ,  5.5])

        >>> layers.extract(.5, 4.2)
        array([ 0.5,  4.2])

        >>> layers.extract(1., 4.)
        array([ 1.,  4.])
        """
        if stop is None:
            stop = self.top
        start, stop = sorted((start, stop))
        start = np.maximum(start, 0.)
        stop = np.minimum(stop, self.thickness)

        bottom = self.layer_at(start)
        top = self.layer_at(stop, lower=True) + 1

        z = self._z[bottom:top + 1].copy()
        z[0], z[-1] = start, stop

        return z
