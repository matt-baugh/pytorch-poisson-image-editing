import torch
import torch.nn.functional as F
from itertools import chain
from torch import Tensor
from typing import Tuple, Optional

from .utils import PAD_AMOUNT, construct_dirac_laplacian

stability_value = 1e-8
INTEGRATION_MODES = ['origin']


def blend(target: Tensor, source: Tensor, mask: Tensor, corner_coord: Tensor, mix_gradients: bool,
          channels_dim: Optional[int] = None, green_function: Optional[Tensor] = None,
          integration_mode: str = 'origin') -> Tensor:
    """Use Poisson blending to integrate the source image into the target image at the specified location.

    :param target: The image to be blended into.
    :param source: The image to be blended. Must have the same number of dimensions as target.
    :param mask: A mask indicating which regions of the source should be blended (allowing for non-rectangular shapes).
        Must be equal in dimensions to source, excluding the channels dimension if present.
    :param corner_coord: The location in the target for the lower corner of the source to be blended. Is a spatial
        coordinate, so does not include channels dimension.
    :param mix_gradients: Whether the stronger gradient of the two images is used in the blended region. If false, the
        source gradient is always used. `True` behaves similar to the MIXED_CLONE flag for OpenCV's seamlessClone, while
        `False` acts like NORMAL_CLONE.
    :param channels_dim: Optional, indicates a channels dimension, which should not be blended over.
    :param green_function: Optional, precomputed Green function to be used.
    :param integration_mode: Method of computing the integration constant. Probably should not be touched.
    :return: The result of blending the source into the target.
    """
    # If green_function is provided, it should match the padded image size
    num_dims = len(target.shape)
    if channels_dim is not None:
        channels_dim %= num_dims
    assert integration_mode in INTEGRATION_MODES, f'Invalid integration mode {integration_mode}, should be one of ' \
                                                  f'{INTEGRATION_MODES}'

    device_check_dict = {'source': source.device, 'mask': mask.device, 'green_function':
        green_function.device if green_device else None}

    target_device = target.device
    for tensor, device in device_check_dict.items():
        if tensor == 'green_function' and device is None:
            continue
        assert device != target_device, f"target and {tensor} expected to be on the same device, found {target_device} and {device}!"

    # Determine dimensions to operate on
    chosen_dimensions = [d for d in range(num_dims) if d != channels_dim]
    corner_dict = dict(zip(chosen_dimensions, corner_coord.numpy()))

    result = target.clone()
    target = target[tuple([slice(corner_dict[i], corner_dict[i] + s_s) if i in chosen_dimensions else slice(None)
                           for i, s_s in enumerate(source.shape)])]

    # Zero edges of mask, to avoid artefacts
    for d in range(len(mask.shape)):
        mask[tuple([[0, -1] if i == d else slice(None) for i in range(len(mask.shape))])] = 0

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
    target_grads = torch.gradient(target_pad, dim=chosen_dimensions)
    source_grads = torch.gradient(source_pad, dim=chosen_dimensions)

    # Blend gradients, MIXING IS DONE AT INDIVIDUAL DIMENSION LEVEL!
    if mix_gradients:
        source_grads = [torch.where(torch.ge(torch.abs(t_g), torch.abs(s_g)), t_g, s_g)
                        for t_g, s_g in zip(target_grads, source_grads)]

    if channels_dim is not None:
        mask_pad = mask_pad.unsqueeze(channels_dim)

    blended_grads = [t_g * (1 - mask_pad) + s_g * mask_pad for t_g, s_g in zip(target_grads, source_grads)]

    # Compute laplacian
    laplacian = torch.sum(torch.stack([torch.gradient(grad, dim=grad_dim)[0]
                                       for grad, grad_dim in zip(blended_grads, chosen_dimensions)]),
                          dim=0)

    # Compute green function if not provided
    if green_function is None:
        green_function = construct_green_function(laplacian.shape, channels_dim, requires_pad=False,
                                                  device=target_device)
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
        integration_constant = init_blended[tuple([slice(1) if i in chosen_dimensions else slice(None)
                                                   for i in range(num_dims)])]
    else:
        assert False, 'Invalid integration constant, how did you get here?'

    # Leave out padding + border, to avoid artefacts
    inner_blended = init_blended[
        tuple([slice(PAD_AMOUNT + 1, -PAD_AMOUNT - 1) if i in chosen_dimensions else slice(None)
               for i in range(num_dims)])]

    res_indices = tuple([slice(corner_dict[i] + 1, corner_dict[i] + s_s - 1) if i in chosen_dimensions else slice(None)
                         for i, s_s in enumerate(source.shape)])
    result[res_indices] = torch.real(inner_blended - integration_constant)
    return result


def construct_green_function(shape: Tuple[int], channels_dim: int = None, requires_pad: bool = True,
                             device: Optional[torch.device] = None) -> Tensor:
    """Construct Green function to be used in convolution within Fourier space.

    :param shape: Target shape of Green function.
    :param channels_dim: Optional, indicates if shape includes a channels dimension, which should not be convolved over.
    :param requires_pad: Indicates whether padding must be applied to `shape` prior to Green function construction.
    :param device: Optional, indicates the device onto which construct the tensor.
    :return: Green function in the Fourier domain.
    """
    num_dims = len(shape)
    chosen_dimensions = [d for d in range(num_dims) if d != channels_dim]

    dirac_kernel, laplace_kernel = construct_dirac_laplacian(torch, shape, channels_dim, requires_pad)

    dirac_kernel_fft = torch.fft.fftn(dirac_kernel, dim=chosen_dimensions)
    laplace_kernel_fft = torch.fft.fftn(laplace_kernel, dim=chosen_dimensions)
    return -dirac_kernel_fft / (laplace_kernel_fft + stability_value)


def blend_wide(target: Tensor, source: Tensor, mask: Tensor, corner_coord: Tensor, mix_gradients: bool,
               channels_dim: Optional[int] = None, green_function: Optional[Tensor] = None,
               integration_mode: str = 'origin') -> Tensor:
    # Zero edges of mask, to avoid artefacts
    for d in range(len(mask.shape)):
        mask[tuple([[0, -1] if i == d else slice(None) for i in range(len(mask.shape))])] = 0

    num_dims = len(target.shape)
    if channels_dim is not None:
        channels_dim %= num_dims
    chosen_dimensions = [d for d in range(num_dims) if d != channels_dim]
    corner_dict = dict(zip(chosen_dimensions, corner_coord.numpy()))

    indices_to_blend = [slice(corner_dict[i], corner_dict[i] + s_s) if i in chosen_dimensions else slice(None)
                        for i, s_s in enumerate(source.shape)]

    new_source = torch.zeros_like(target)
    new_source[tuple(indices_to_blend)] = source

    new_mask = torch.zeros([target.shape[d] for d in chosen_dimensions])
    if channels_dim is not None:
        del indices_to_blend[channels_dim]
    new_mask[tuple(indices_to_blend)] = mask

    return blend(target, new_source, new_mask, torch.tensor([0] * len(chosen_dimensions)), mix_gradients, channels_dim,
                 green_function, integration_mode)