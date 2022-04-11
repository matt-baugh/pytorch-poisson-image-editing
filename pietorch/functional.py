from itertools import chain
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

PAD_AMOUNT = 4
stability_value = 1e-8
INTEGRATION_MODES = ['origin']  # TODO: Implement more, test results


def blend(target: Tensor, source: Tensor, mask: Tensor, mix_gradients: bool, channels_dim: int = None,
          green_function: Tensor = None, integration_mode: str = 'origin'):
    # If green_function is provided, it should match the padded image size
    num_dims = len(target.shape)
    assert integration_mode in INTEGRATION_MODES, f'Invalid integration mode {integration_mode}, should be one of ' \
                                                  f'{INTEGRATION_MODES}'

    # determine dimensions to operate on
    chosen_dimensions = [d for d in range(num_dims) if d != channels_dim]

    # Pad images in operating dimensions
    pad_amounts = [PAD_AMOUNT if d in chosen_dimensions else 0 for d in range(num_dims)]
    pad = tuple(chain(*[[p, p] for p in reversed(pad_amounts)]))

    target_pad = F.pad(target, pad)
    source_pad = F.pad(source, pad)

    # Pad with zeroes, as don't blend within padded region
    if channels_dim is not None:
        del pad_amounts[channels_dim]
        pad = tuple(chain(*[[p, p] for p in reversed(pad_amounts)]))
    mask_pad = F.pad(mask, pad)

    # Compute gradients
    target_grads = [compute_gradient(target_pad, d) for d in chosen_dimensions]
    source_grads = [compute_gradient(source_pad, d) for d in chosen_dimensions]

    if mix_gradients:
        source_grads = [torch.where(torch.ge(torch.abs(t_g), torch.abs(s_g)), t_g, s_g) for t_g, s_g in zip(target_grads, source_grads)]

    # Blend gradients (MIXING IS DONE AT INDIVIDUAL DIMENSION LEVEL!
    # TODO: errode mask first
    if channels_dim is not None:
        mask_pad = mask_pad.unsqueeze(channels_dim)

    blended_grads = [t_g * (1 - mask_pad) + s_g * mask_pad for t_g, s_g in zip(target_grads, source_grads)]

    # Compute laplacian
    laplacian = torch.sum(torch.stack([compute_gradient(grad, grad_dim)
                                       for grad, grad_dim in zip(blended_grads, chosen_dimensions)]),
                          dim=0)

    # Compute green function if not provided
    if green_function is None:
        green_function = construct_green_function(laplacian.shape, channels_dim, requires_pad=False)
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
        integration_constant = init_blended[tuple([slice(1) if i in chosen_dimensions else slice(s)
                                                   for i, s in enumerate(init_blended.shape)])]
    else:
        assert False, 'Invalid integration constant, how did you get here?'

    inner_blended = init_blended[tuple([slice(PAD_AMOUNT, -PAD_AMOUNT) if i in chosen_dimensions else slice(s)
                                 for i, s in enumerate(init_blended.shape)])]

    return torch.real(inner_blended - integration_constant)


def compute_gradient(image: Tensor, dim: int) -> Tensor:
    image_pad = torch.cat([image[tuple([slice(1) if i == dim else slice(s) for i, s in enumerate(image.shape)])],
                           image,
                           image[tuple([slice(-1, s) if i == dim else slice(s) for i, s in enumerate(image.shape)])]],
                          dim=dim)

    front = image_pad[tuple([slice(2 if i == dim else 0, s) for i, s in enumerate(image_pad.shape)])]
    back = image_pad[tuple([slice(0, -2 if i == dim else s) for i, s in enumerate(image_pad.shape)])]
    return (front - back) / 2


def construct_green_function(shape: Tuple[int], channels_dim: int = None, requires_pad=True)\
        -> Tensor:
    num_dims = len(shape)
    chosen_dimensions = [d for d in range(num_dims) if d != channels_dim]
    # Aim is to match chosen dimensions of input shape (padding if necessary), but set others to size 1.
    padding = 2 * PAD_AMOUNT if requires_pad else 0
    shape = [(s + padding) if i in chosen_dimensions else 1 for i, s in enumerate(shape)]
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
