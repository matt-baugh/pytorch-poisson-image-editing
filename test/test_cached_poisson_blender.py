from time import time
from typing import Tuple

import pytest
import numpy as np
import torch
from torch import Tensor

from pietorch import blend, CachedPoissonBlender


@pytest.fixture
def setup_blend_objects() -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    target = torch.ones(1000, 1000)

    source = torch.zeros(500, 500)
    mask = torch.zeros_like(source, dtype=torch.float)
    radii = centre = np.array(source.shape) // 2

    corner = torch.tensor([250, 250])

    for i in range(-radii[0], radii[0]):
        for j in range(-radii[1], radii[1]):
            distance = np.sqrt(i ** 2 + j ** 2)
            if distance <= 50:
                coord = tuple(np.array([i, j]) + centre)
                source[coord] = np.sin(distance * np.pi / 5) / 4
                mask[coord] = 1

    return target, source, mask, corner


def test_cache_results_same(setup_blend_objects):

    func_res = blend(*setup_blend_objects, True)

    cache_res = CachedPoissonBlender().blend(*setup_blend_objects, True)

    assert torch.allclose(func_res, cache_res)


def test_automatic_cache_is_faster(setup_blend_objects):

    cache = CachedPoissonBlender()

    start_uncached = time()
    res_1 = cache.blend(*setup_blend_objects, True)
    uncached_time = time() - start_uncached

    assert len(cache.green_function_cache) == 1, 'Green function has not been added to cache.'

    start_cached = time()
    res_2 = cache.blend(*setup_blend_objects, True)
    cached_time = time() - start_cached

    assert cached_time < uncached_time
    assert torch.allclose(res_1, res_2)


def test_precompute_is_faster(setup_blend_objects):

    start_uncached = time()
    func_res = blend(*setup_blend_objects, True)
    uncached_time = time() - start_uncached

    _, source, _, _ = setup_blend_objects
    cache = CachedPoissonBlender([(source.shape, None)])
    assert len(cache.green_function_cache) == 1, 'Green function has not been added to cache.'

    start_cached = time()
    res_2 = cache.blend(*setup_blend_objects, True)
    cached_time = time() - start_cached

    assert cached_time < uncached_time
    assert torch.allclose(func_res, res_2)


def test_manual_cache_faster(setup_blend_objects):

    start_uncached = time()
    func_res = blend(*setup_blend_objects, True)
    uncached_time = time() - start_uncached

    _, source, _, _ = setup_blend_objects
    cache = CachedPoissonBlender()
    cache.add_to_cache(source.shape, None)
    assert len(cache.green_function_cache) == 1, 'Green function has not been added to cache.'

    start_cached = time()
    res_2 = cache.blend(*setup_blend_objects, True)
    cached_time = time() - start_cached

    assert cached_time < uncached_time
    assert torch.allclose(func_res, res_2)


def test_clear_cache(setup_blend_objects):

    cache = CachedPoissonBlender()
    assert len(cache.green_function_cache) == 0, 'Cache should initially be empty.'

    _, source, _, _ = setup_blend_objects
    cache.add_to_cache(source.shape, None)
    assert len(cache.green_function_cache) == 1, 'Green function has not been added to cache.'

    cache.clear_cache()
    assert len(cache.green_function_cache) == 0, 'Cache has not been cleared.'


def test_double_add_no_cache_increase(setup_blend_objects):

    cache = CachedPoissonBlender()
    assert len(cache.green_function_cache) == 0, 'Cache should initially be empty.'

    _, source, _, _ = setup_blend_objects
    cache.add_to_cache(source.shape, None)
    assert len(cache.green_function_cache) == 1, 'Green function has not been added to cache.'

    cache.add_to_cache(source.shape, None)
    assert len(cache.green_function_cache) == 1, 'Duplicate key was added to cache.'
