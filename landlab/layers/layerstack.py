import numpy as np
import bisect


class LayerFields(object):
    def __new__(cls, *args, **kwds):
        for field in kwds.get('fields', ()):
            setattr(cls, field, property(lambda self: self[field], doc=field))
        return object.__new__(cls, *args, **kwds)

    def __init__(self, *args, **kwds):
        super(LayerFields, self).__init__(*args, **kwds)

        self._fields = dict()
        for field in kwds.get('fields', ()):
            self._add_field(field)

    def _add_field(self, name, **kwds):
        if name not in self._fields:
            self._fields[name] = np.empty(self.size, **kwds)

    def add(self, dz, **kwds):
        bins = self.extract(self.top - dz)
        n_bins = len(bins)
        for name, val in kwds.items():
            array = getattr(self, name)[-n_bins:]
            array[1:] = val

            array[0] = (
                (bins[1] - bins[0]) * val +
                (bins[0] - self.z[-n_bins]) * array[0]
            ) / (bins[1] - self.z[-n_bins])

    def resize(self, *args, **kwds):
        for name, array in self._fields.items():
            self._fields[name] = np.resize(array, self.size)

    def reduce(self, dz, name):
        bins = np.diff(self.extract(self.top - dz))
        return np.sum(getattr(self, name)[-len(bins):] * bins / dz)

    @property
    def fields(self):
        return self._fields.keys()

    def __getitem__(self, name):
        return self._fields[name][:self._top + 1]


class LayerStack(LayerFields):
    def __init__(self, n_grains=1, z0=0., dz=1., **kwds):
        self._z = np.arange(10, dtype=float) * dz
        self._z0 = z0
        self._dz = float(dz)

        self._top = 0

        super(LayerStack, self).__init__(**kwds)

    @property
    def base(self):
        """Elevation of the bottom of the column."""
        return self._z0

    @base.setter
    def base(self, new_base):
        self._z0 = new_base

    @property
    def top(self):
        """Elevation of the top of the column."""
        return self.base + self._z[self._top]

    @top.setter
    def top(self, new_top):
        self._z0 += new_top - self.top

    @property
    def z(self):
        """Elevation to bottom of each layer."""
        return self._z[:self._top + 1]

    @property
    def dz(self):
        """Thickness of new bins."""
        return self._dz

    @property
    def size(self):
        return self._z.size

    def resize(self, min=None):
        min = min or int(self.size * 1.25)

        n_bins = int(self.size * 1.25)
        while n_bins < min:
            n_bins = int(n_bins * 1.25)

        self._z = np.append(
            self._z,
            np.arange(1, n_bins - self._z.size + 1) * self._dz + self._z[-1])

        super(LayerStack, self).resize(min=min)

    def is_empty(self):
        return self._top == 0

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
            self.resize(min=self.size + (fill_to - self._z[-1]) / self._dz + 1)

        new_top = bisect.bisect_left(self._z[self._top:], fill_to) + self._top

        self._z[new_top] = fill_to
        self._top = new_top

        super(LayerStack, self).add(dz, **kwds)

    def remove(self, dz):
        """Remove sediment from the top of a column.

        Parameters
        ----------
        dz : float
            Amount of sediment to remove.
        """
        if thickness < 0:
            return self.add(- thickness)

        new_z = self._z[self._top] - dz
        new_top = bisect.bisect_left(self._z[:self._top], new_z)
        self._z[self._top:] *= self._dz
        self._top = new_top
        self._z[self._top] = new_z

    def lift(self, dz):
        """Lift the base of the stack."""
        self.base += dz

    def lower(self, dz):
        """Lower the base of the stack."""
        self.lift(- dz)

    def extract(self, start=0, stop=None):
        if stop is None:
            stop = self.top
        start, stop = sorted((start, stop))
        start = np.maximum(start, 0.)
        stop = np.minimum(stop, self.top)

        bottom = bisect.bisect(self._z[:self._top], start) - 1
        top = bisect.bisect_left(self._z[:self._top + 1], stop)

        z = self._z[bottom:top + 1].copy()
        z[0], z[-1] = start, stop

        return z
