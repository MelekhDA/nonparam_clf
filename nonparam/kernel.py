from typing import Union
import math

__all__ = [
    'RaisedKernel'
]


class RaisedKernel:
    """Kernel function"""

    KERNELS = [
        'uniform', 'traingular',
        'parabolic', 'cubic'
    ]

    def __init__(self, kernel, epsi=1e-2):
        assert kernel in self.KERNELS, f'Select a kernel from the list {RaisedKernel.KERNELS}'

        self._epsi = epsi
        self._kernel = getattr(self, f'_{kernel}_kernel')

    @property
    def epsi(self):
        return self._epsi

    @epsi.setter
    def epsi(self, epsi):
        self._epsi = epsi

    def kernel(self, z: Union[float, int]) -> Union[float, int]:
        """Wrapper for the used kernel"""
        return self._kernel(z)

    def _uniform_kernel(self, z: Union[float, int]) -> Union[float, int]:
        """Uniform kernel"""

        if math.fabs(z) <= 1:
            return 0.5 + self._epsi

        return self._epsi

    def _traingular_kernel(self, z: Union[float, int]) -> Union[float, int]:
        """Triangular kernel"""

        if math.fabs(z) <= 1:
            return 1 - math.fabs(z) + self._epsi

        return self._epsi

    def _parabolic_kernel(self, z: Union[float, int]) -> Union[float, int]:
        """Parabolic kernel"""

        if math.fabs(z) <= 1:
            return 0.75 * (1 - z ** 2) + self._epsi

        return self._epsi

    def _cube_kernel(self, z: Union[float, int]) -> Union[float, int]:
        """Cube kernel"""

        if math.fabs(z) <= 1:
            return (1 + 2 * math.fabs(z)) * (1 - math.fabs(z)) ** 2 + self._epsi

        return self._epsi
