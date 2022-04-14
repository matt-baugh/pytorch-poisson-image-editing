from typing import Optional, Tuple, List

from torch import Tensor

from .functional import blend, construct_green_function


class CachedPoissonBlender:

    def __init__(self, init_to_cache: Optional[List[Tuple[Tuple, Optional[int]]]] = None):

        self.green_function_cache = {}

        if init_to_cache is not None:
            for s, c_d in init_to_cache:
                self.add_to_cache(s, c_d)

    def blend(self, target: Tensor, source: Tensor, mask: Tensor, corner_coord: Tensor, mix_gradients: bool,
              channels_dim: Optional[int] = None, integration_mode: str = 'origin') -> Tensor:

        cache_key = (source.shape, channels_dim)

        if cache_key not in self.green_function_cache:
            self.add_to_cache(*cache_key)

        return blend(target, source, mask, corner_coord, mix_gradients, channels_dim,
                     self.green_function_cache[cache_key], integration_mode)

    def add_to_cache(self, shape: Tuple[int], channels_dim: Optional[int] = None):
        cache_key = (shape, channels_dim)
        if cache_key not in self.green_function_cache:
            self.green_function_cache[cache_key] = construct_green_function(*cache_key, requires_pad=True)

    def clear_cache(self):
        self.green_function_cache = {}
