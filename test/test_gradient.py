import numpy as np
import pytest
import torch

from pietorch import compute_gradient


@pytest.mark.parametrize("num_dims", range(1, 5))
def test_gradient_shape_unchanged(num_dims):

    # Repeat for fun
    for _ in range(100):
        shape = np.random.randint(5, np.power(1048576, 1. / num_dims), size=num_dims)
        image = torch.randint(256, size=tuple(shape))
        grad = compute_gradient(image, np.random.randint(num_dims))
        assert image.shape == grad.shape, 'Gradient shape does not match input shape!'


def test_gradient_1D():
    arr_len = 5
    x = torch.arange(5)
    expected = torch.tensor([0.5] + [1.] * (arr_len - 2) + [0.5])

    assert (expected == compute_gradient(x, 0)).all(), 'Flat gradient incorrect.'

    x = x.unsqueeze(0)
    expected = expected.unsqueeze(0)
    assert (expected == compute_gradient(x, 1)).all(), 'Gradient incorrect when empty dimension added before.'

    x = x.unsqueeze(-1)
    expected = expected.unsqueeze(-1)
    assert (expected == compute_gradient(x, 1)).all(), 'Gradient incorrect when empty dimension added before and after.'


def test_gradient_2D():
    arr_len = 5
    x = torch.arange(5)
    x = x.repeat(5, 1)
    expected = torch.tensor([0.5] + [1.] * (arr_len - 2) + [0.5])
    expected = expected.repeat(5, 1)

    assert (expected == compute_gradient(x, 1)).all(), 'Non-zero gradient incorrect.'
    assert (torch.zeros_like(x) == compute_gradient(x, 0)).all(), 'Zero gradient incorrect.'
