from typing import Optional, Tuple

import pytest
import torch
from torch import Tensor

from pietorch import blend, construct_green_function


def make_blend_inputs(channels_dim: Optional[int]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Small, fast inputs: a smooth target with a ring-shaped source blended into its centre."""
    size, patch = 64, 24
    yy, xx = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing='ij')
    target = 0.5 + 0.3 * torch.sin(3 * yy) * torch.cos(2 * xx)

    py, px = torch.meshgrid(torch.arange(patch) - patch / 2, torch.arange(patch) - patch / 2, indexing='ij')
    distance = torch.sqrt(py ** 2 + px ** 2)
    source = torch.sin(distance * torch.pi / 5) / 4
    mask = (distance <= patch / 3).float()

    if channels_dim is not None:
        target = torch.stack([target, 1 - target, target / 2], dim=channels_dim)
        source = torch.stack([source, -source, source / 2], dim=channels_dim)

    corner = torch.tensor([20, 20])
    return target, source, mask, corner


@pytest.mark.parametrize('channels_dim', [None, 0, -1])
@pytest.mark.parametrize('mix_gradients', [False, True])
def test_precomputed_green_function_matches_default(channels_dim, mix_gradients):
    """Passing a precomputed Green function must give the same result as letting `blend` build one.

    Regression test: a truthiness check on the `green_function` tensor made every call that passed one raise.
    """
    target, source, mask, corner = make_blend_inputs(channels_dim)
    norm_channels_dim = None if channels_dim is None else channels_dim % source.dim()
    green_function = construct_green_function(source.shape, norm_channels_dim, requires_pad=True)

    default_res = blend(target, source, mask.clone(), corner, mix_gradients, channels_dim)
    precomputed_res = blend(target, source, mask.clone(), corner, mix_gradients, channels_dim,
                            green_function=green_function)

    assert torch.allclose(default_res, precomputed_res)


# The 'meta' device lets us exercise the device checks without needing a GPU.
@pytest.mark.parametrize('argument', ['source', 'mask', 'green_function'])
def test_mismatched_device_raises(argument):
    target, source, mask, corner = make_blend_inputs(None)
    kwargs = {'source': source, 'mask': mask,
              'green_function': construct_green_function(source.shape, None, requires_pad=True)}
    kwargs[argument] = kwargs[argument].to('meta')

    with pytest.raises(ValueError, match=argument):
        blend(target, kwargs['source'], kwargs['mask'], corner, False, green_function=kwargs['green_function'])
