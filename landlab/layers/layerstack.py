import numpy as np

from .layers import Layers, LayerFields


class LayerStack(Layers, LayerFields):

    def add(self, dz, **kwds):
        """Add sediment to a column.

        Parameters
        ----------
        dz : float
            Amount of sediment to add.
        """
        if dz < 0:
            return self.remove(- dz)

        if self._top + 1 > self.allocated:
            self.resize(self.allocated + 1)

        self._top += 1
        self._z[self._top] = self._z[self._top - 1] + dz

        super(Layers, self).add(dz, **kwds)

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

        self._z[new_top:self._top + 1] = new_z
        self._z[self._top] = new_z

    def resize(self, newsize):
        newsize = int(newsize)
        if newsize < self.allocated:
            return

        new_allocated = (newsize >> 3) + 6 + newsize

        new_z = np.empty_like(self._z, (new_allocated, ) + self._z.shape[1:])
        new_z[:self.allocated] = self._z
        self._z = new_z

        super(LayerGridStack, self).resize(newsize)
