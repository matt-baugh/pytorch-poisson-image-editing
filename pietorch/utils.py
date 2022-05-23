from typing import Tuple

PAD_AMOUNT = 4


def construct_dirac_laplacian(lib, shape: Tuple[int], channels_dim: int = None, requires_pad: bool = True):
    num_dims = len(shape)
    chosen_dimensions = [d for d in range(num_dims) if d != channels_dim]
    # Aim is to match chosen dimensions of input shape (padding if necessary), but set others to size 1.
    padding = 2 * PAD_AMOUNT if requires_pad else 0
    shape = [(s + padding) if i in chosen_dimensions else 1 for i, s in enumerate(shape)]
    kernel_centre = [1 if d in chosen_dimensions else 0 for d in range(num_dims)]

    dirac_kernel = lib.zeros(shape)
    dirac_kernel[tuple(kernel_centre)] = 1

    laplace_kernel = lib.zeros(shape)
    laplace_kernel[tuple(kernel_centre)] = 2 * len(chosen_dimensions)
    for c_d in chosen_dimensions:
        for i in [0, 2]:
            laplace_kernel[tuple([i if d == c_d else k for d, k in enumerate(kernel_centre)])] = -1

    return dirac_kernel, laplace_kernel
