from typing import Tuple, Optional

import numpy as np
import scipy.fft
from scipy.fft import fftn, ifftn
from scipy import ndimage

from .functional import stability_value, INTEGRATION_MODES
from .utils import PAD_AMOUNT, construct_dirac_laplacian


def blend_numpy(target: np.ndarray, source: np.ndarray, mask: np.ndarray, corner_coord: np.ndarray, mix_gradients: bool,
                channels_dim: Optional[int] = None, green_function: Optional[np.ndarray] = None,
                integration_mode: str = 'origin') -> np.ndarray:
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
    assert integration_mode in INTEGRATION_MODES, f'Invalid integration mode {integration_mode}, should be one of ' \
                                                  f'{INTEGRATION_MODES}'

    # Determine dimensions to operate on
    chosen_dimensions = [d for d in range(num_dims) if d != channels_dim]  # TODO: allow for negative dimensions
    corner_dict = dict(zip(chosen_dimensions, corner_coord))

    result = target.copy()
    target = target[tuple([slice(corner_dict[i], corner_dict[i] + s_s) if i in chosen_dimensions else slice(t_s)
                           for i, (t_s, s_s) in enumerate(zip(target.shape, source.shape))])]

    # Zero edges of mask, to avoid artefacts
    for d in range(len(mask.shape)):
        mask[tuple([[0, -1] if i == d else slice(s) for i, s in enumerate(mask.shape)])] = 0

    # Pad images in operating dimensions
    pad_amounts = [PAD_AMOUNT if d in chosen_dimensions else 0 for d in range(num_dims)]
    pad = [[p, p] for p in pad_amounts]

    target_pad = np.pad(target, pad)
    source_pad = np.pad(source, pad)

    # Pad with zeroes, as don't blend within padded region
    if channels_dim is not None:
        del pad[channels_dim]
    mask_pad = np.pad(mask, pad)

    # Compute gradients
    target_grads = np.gradient(target_pad, axis=chosen_dimensions)
    source_grads = np.gradient(source_pad, axis=chosen_dimensions)

    # Blend gradients, MIXING IS DONE AT INDIVIDUAL DIMENSION LEVEL!
    if mix_gradients:
        source_grads = [np.where(np.abs(t_g) >= np.abs(s_g), t_g, s_g)
                        for t_g, s_g in zip(target_grads, source_grads)]

    if channels_dim is not None:
        mask_pad = np.expand_dims(mask_pad, channels_dim)

    blended_grads = [t_g * (1 - mask_pad) + s_g * mask_pad for t_g, s_g in zip(target_grads, source_grads)]

    # Compute laplacian
    laplacian = sum([np.gradient(grad, axis=grad_dim) for grad, grad_dim in zip(blended_grads, chosen_dimensions)])

    # Compute green function if not provided
    if green_function is None:
        green_function = construct_green_function_numpy(laplacian.shape, channels_dim, requires_pad=False)
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
    init_blended = ifftn(fftn(laplacian, axes=chosen_dimensions) * green_function, axes=chosen_dimensions)

    # Use boundaries to determine integration constant, and extract inner blended image
    if integration_mode == 'origin':
        integration_constant = init_blended[tuple([slice(1) if i in chosen_dimensions else slice(s)
                                                   for i, s in enumerate(init_blended.shape)])]
    else:
        assert False, 'Invalid integration constant, how did you get here?'

    # Leave out padding + border, to avoid artefacts
    inner_blended = init_blended[tuple([slice(PAD_AMOUNT + 1, -PAD_AMOUNT - 1) if i in chosen_dimensions else slice(s)
                                        for i, s in enumerate(init_blended.shape)])]

    res_indices = tuple([slice(corner_dict[i] + 1, corner_dict[i] + s_s - 1) if i in chosen_dimensions else slice(t_s)
                         for i, (t_s, s_s) in enumerate(zip(target.shape, source.shape))])
    result[res_indices] = np.real(inner_blended - integration_constant)
    return result


def construct_green_function_numpy(shape: Tuple[int], channels_dim: int = None, requires_pad: bool = True) \
        -> np.ndarray:
    """Construct Green function to be used in convolution within Fourier space.

    :param shape: Target shape of Green function.
    :param channels_dim: Optional, indicates if shape includes a channels dimension, which should not be convolved over.
    :param requires_pad: Indicates whether padding must be applied to `shape` prior to Green function construction.
    :return: Green function in the Fourier domain.
    """
    num_dims = len(shape)
    chosen_dimensions = [d for d in range(num_dims) if d != channels_dim]

    dirac_kernel, laplace_kernel = construct_dirac_laplacian(np, shape, channels_dim, requires_pad)

    dirac_kernel_fft = fftn(dirac_kernel, axes=chosen_dimensions)
    laplace_kernel_fft = fftn(laplace_kernel, axes=chosen_dimensions)
    return -dirac_kernel_fft / (laplace_kernel_fft + stability_value)


def blend_wide_numpy(target: np.ndarray, source: np.ndarray, mask: np.ndarray, corner_coord: np.ndarray,
                     mix_gradients: bool, channels_dim: Optional[int] = None,
                     green_function: Optional[np.ndarray] = None, integration_mode: str = 'origin') -> np.ndarray:
    # Zero edges of mask, to avoid artefacts
    for d in range(len(mask.shape)):
        mask[tuple([[0, -1] if i == d else slice(s) for i, s in enumerate(mask.shape)])] = 0

    num_dims = len(target.shape)
    chosen_dimensions = [d for d in range(num_dims) if d != channels_dim]  # TODO: allow for negative dimensions
    corner_dict = dict(zip(chosen_dimensions, corner_coord))

    indices_to_blend = [slice(corner_dict[i], corner_dict[i] + s_s) if i in chosen_dimensions else slice(t_s)
                        for i, (t_s, s_s) in enumerate(zip(target.shape, source.shape))]

    new_source = np.zeros_like(target)
    new_source[tuple(indices_to_blend)] = source

    new_mask = np.zeros([target.shape[d] for d in chosen_dimensions])
    if channels_dim is not None:
        del indices_to_blend[channels_dim]
    new_mask[tuple(indices_to_blend)] = mask

    return blend_numpy(target, new_source, new_mask, np.array([0] * len(chosen_dimensions)), mix_gradients,
                       channels_dim, green_function, integration_mode)


# Trying to replicate OpenCV's blending, not going well...
# Source material:
# https://github.com/opencv/opencv/blob/3f4ffe7844cead4e406bfb0067e9dae2ff9247f3/modules/photo/src/seamless_cloning_impl.cpp#L323
# https://web.media.mit.edu/~raskar/photo/code.pdf
def blend_dst_numpy(target: np.ndarray, source: np.ndarray, mask: np.ndarray, corner_coord: np.ndarray,
                    mix_gradients: bool, channels_dim: Optional[int] = None):
    print('WIP, USE CAREFULLY!')

    num_dims = len(target.shape)
    # Determine dimensions to operate on
    chosen_dimensions = [d for d in range(num_dims) if d != channels_dim]  # TODO: allow for negative dimensions
    corner_dict = dict(zip(chosen_dimensions, corner_coord))

    result = target.copy()
    target = target[tuple([slice(corner_dict[i], corner_dict[i] + s_s) if i in chosen_dimensions else slice(t_s)
                           for i, (t_s, s_s) in enumerate(zip(target.shape, source.shape))])]

    # Zero edges of mask, to avoid artefacts
    for d in range(len(mask.shape)):
        mask[tuple([[0, -1] if i == d else slice(s) for i, s in enumerate(mask.shape)])] = 0

    # Compute gradients
    grad_kernel_1 = np.array([0, -1, 1])
    target_grads = [ndimage.correlate1d(target, grad_kernel_1, axis=d, mode='mirror') for d in chosen_dimensions]
    source_grads = [ndimage.correlate1d(source, grad_kernel_1, axis=d, mode='mirror') for d in chosen_dimensions]

    # Blend gradients, MIXING IS DONE AT INDIVIDUAL DIMENSION LEVEL!
    if mix_gradients:
        source_grads = [np.where(np.abs(t_g) >= np.abs(s_g), t_g, s_g)
                        for t_g, s_g in zip(target_grads, source_grads)]

    if channels_dim is not None:
        mask = np.expand_dims(mask, channels_dim)

    blended_grads = [t_g * (1 - mask) + s_g * mask for t_g, s_g in zip(target_grads, source_grads)]

    # Compute laplacian
    grad_kernel_2 = np.array([-1, 1, 0])
    laplacian = sum([ndimage.correlate1d(grad, grad_kernel_2, axis=grad_dim, mode='mirror')
                     for grad, grad_dim in zip(blended_grads, chosen_dimensions)])

    boundary_points = target.copy()
    # Zero all pixels except boundaries
    boundary_points[tuple([slice(1, s - 1) if i in chosen_dimensions else slice(s)
                           for i, s in enumerate(boundary_points.shape)])] = 0
    # Compute laplacian for boundaries
    lap_kernel = np.zeros([3 if i in chosen_dimensions else 1 for i in range(num_dims)])
    k_centre = [1 if i in chosen_dimensions else 0 for i in range(num_dims)]
    lap_kernel[tuple(k_centre)] = -2 * len(chosen_dimensions)
    for c_d in chosen_dimensions:
        for c_c in [0, 2]:
            lap_kernel[tuple([c_c if i == c_d else k for i, k in enumerate(k_centre)])] = 1

    boundary_points = ndimage.correlate(boundary_points, lap_kernel, mode='mirror')
    # boundary_points = sum([np.gradient(g, axis=i)
    #                        for i, g in zip(chosen_dimensions, np.gradient(boundary_points, axis=chosen_dimensions))])

    # Subtract boundary's influence from laplacian
    mod_diff = laplacian - boundary_points
    mod_diff = mod_diff[tuple([slice(1, -1) if i in chosen_dimensions else slice(s)
                               for i, s in enumerate(mod_diff.shape)])]

    # I identified the dst type as both reference implementations implement the DST via a DFFT, and the matlab code
    # implementation of the inverse also matches the inverse formulation for type 1.
    transformed = scipy.fft.dstn(mod_diff, type=1, axes=chosen_dimensions)

    eigenvalues = [2 * np.cos(np.pi * (np.arange(mod_diff.shape[i]) + 1) / (target.shape[i] - 1)) - 2
                   for i in chosen_dimensions]

    denorm = sum([e.reshape(tuple([-1 if j == i else 1 for j in range(num_dims)]))
                  for i, e in zip(chosen_dimensions, eigenvalues)])
    transformed /= denorm

    blended = scipy.fft.idstn(transformed, type=1, axes=chosen_dimensions)

    res_indices = tuple([slice(corner_dict[i] + 1, corner_dict[i] + 1 + s_s) if i in chosen_dimensions else slice(t_s)
                         for i, (t_s, s_s) in enumerate(zip(target.shape, blended.shape))])
    result[res_indices] = blended
    return result
