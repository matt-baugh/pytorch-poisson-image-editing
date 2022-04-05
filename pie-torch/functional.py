from itertools import chain
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

PAD_AMOUNT = 4
stability_value = 1e-8
INTEGRATION_MODES = ['origin']  # TODO: Implement more, test results


def blend(target: Tensor, source: Tensor, mask: Tensor, mode: str, batch_dim: int = None, channels_dim: int = None,
          green_function: Tensor = None, pad_mode: str = 'constant', integration_mode: str = 'origin'):
    # If green_function is provided, it should match the padded image size
    num_dims = len(target.shape)
    assert integration_mode in INTEGRATION_MODES, f'Invalid integration mode {integration_mode}, should be one of ' \
                                                  f'{INTEGRATION_MODES}'

    # determine dimensions to operate on
    chosen_dimensions = [d for d in range(num_dims) if d != batch_dim and d != channels_dim]

    # Pad images in operating dimensions
    pad_amounts = [PAD_AMOUNT if d in chosen_dimensions else 0 for d in range(num_dims)]
    pad = tuple(chain(*[[p, p] for p in reversed(pad_amounts)]))

    target_pad = F.pad(target, pad, pad_mode)
    source_pad = F.pad(source, pad, pad_mode)

    # Pad with zeroes, as don't blend within padded region
    mask_pad = F.pad(mask, pad, 'constant')

    # Compute gradients
    target_grads = [compute_gradient(target_pad, d) for d in chosen_dimensions]
    source_grads = [compute_gradient(source_pad, d) for d in chosen_dimensions]

    # Blend gradients
    blended_grads = target_grads

    # Compute laplacian
    laplacian = torch.sum(torch.stack([compute_gradient(grad, grad_dim)
                                       for grad, grad_dim in zip(blended_grads, chosen_dimensions)]),
                          dim=0)

    # Compute green function if not provided
    if green_function is None:
        green_function = construct_green_function(laplacian.shape, batch_dim, channels_dim, requires_pad=False)
    else:
        for d in range(num_dims):
            if d in chosen_dimensions:
                assert green_function.shape[d] == laplacian.shape[d], f'Green function has mismatched shape on ' \
                                                                      f'dimension {d}: expected {laplacian.shape[d]},' \
                                                                      f' got {green_function.shape[d]}.'
            else:
                assert green_function.shape[d] == 1, f'Green function should have size 1 in non-chosen dimension ' \
                                                     f'{d}: has {green_function.shape[d]}.'

    # Apply green function convolution
    init_blended = torch.fft.ifftn(torch.fft.fftn(laplacian, dim=chosen_dimensions) * green_function,
                                   dim=chosen_dimensions)

    # Use boundaries to determine integration constant, and extract inner blended image
    if integration_mode == 'origin':
        integration_constant = init_blended[tuple([0] * num_dims)]
    else:
        assert False, 'Invalid integration constant, how did you get here?'

    inner_blended = init_blended[tuple([slice(PAD_AMOUNT, -PAD_AMOUNT) if i in chosen_dimensions else slice(s)
                                 for i, s in enumerate(init_blended.shape)])]

    return inner_blended - integration_constant


def compute_gradient(image: Tensor, dim: int):
    num_dims = len(image.shape)
    trailing_dimensions = num_dims - 1 - dim
    pad = tuple([1, 1] + [0] * (2 * trailing_dimensions))
    image_pad = F.pad(image, pad, mode='replicate')

    front = image_pad[tuple([slice(2 if i == dim else 0, s) for i, s in enumerate(image_pad.shape)])]
    back = image_pad[tuple([slice(0, -2 if i == dim else s) for i, s in enumerate(image_pad.shape)])]
    return (front - back) / 2


def construct_green_function(shape: Tuple[int], batch_dim: int = None, channels_dim: int = None, requires_pad=True):
    num_dims = len(shape)
    chosen_dimensions = [d for d in range(num_dims) if d != batch_dim and d != channels_dim]

    # Aim is to match chosen dimensions of input shape (padding if necessary), but set others to size 1.
    padding = 2 * PAD_AMOUNT if requires_pad else 0
    shape = [(d + padding) if d in chosen_dimensions else 1 for d in range(num_dims)]
    kernel_centre = [1 if d in chosen_dimensions else 0 for d in range(num_dims)]

    dirac_kernel = torch.zeros(shape)
    dirac_kernel[tuple(kernel_centre)] = 1

    laplace_kernel = torch.zeros(shape)
    laplace_kernel[tuple(kernel_centre)] = 2 * len(chosen_dimensions)
    for c_d in chosen_dimensions:
        for i in [0, 2]:
            laplace_kernel[tuple([i if d == c_d else k for d, k in enumerate(kernel_centre)])] = -1

    dirac_kernel_fft = torch.fft.fftn(dirac_kernel, dim=chosen_dimensions)
    laplace_kernel_fft = torch.fft.fftn(laplace_kernel, dim=chosen_dimensions)
    return -dirac_kernel_fft / (laplace_kernel_fft + stability_value)
