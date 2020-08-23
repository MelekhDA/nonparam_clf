import numpy as np

__all__ = [
    'RaisedIndicator'
]


class RaisedIndicator:
    """Indicator function"""

    def __init__(self, epsi=1e-2):
        self._epsi = epsi

    @property
    def epsi(self):
        return self._epsi

    @epsi.setter
    def epsi(self, epsi):
        self._epsi = epsi

    def indicator(self, x1, x2):
        """indicator function"""
        if np.isnan(x1) or np.isnan(x2):
            return 1

        if x1 == x2:
            return 1 + self._epsi

        return self._epsi
