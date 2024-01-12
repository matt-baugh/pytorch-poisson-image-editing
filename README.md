# PIE-torch: Poisson Image Editing in Pytorch

Fast, n-dimensional Poisson image editing.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matt-baugh/pytorch-poisson-image-editing/blob/master/examples.ipynb)

2 implementations, including:
 - using Green Function Convolution, as described in [Fast and Optimal Laplacian Solver for Gradient-Domain Image Editing using Green Function Convolution](https://arxiv.org/abs/1902.00176)
 - using a Discrete Sine Transform, following [OpenCV's implementation](https://github.com/opencv/opencv/blob/3f4ffe7844cead4e406bfb0067e9dae2ff9247f3/modules/photo/src/seamless_cloning_impl.cpp#L323)

Recommendations:
 - For blending images with consistent boundaries, use `blend`, the Green Function Convolution implementation. 
 - For images with inconsistent boundaries, use `blend_dst_numpy`.

Main interface:
 - `blend`: primary entrypoint, blends source image into target image at specified coordinates.
 - `blend_dst_numpy`: entrypoint for DST-based blending (currently only available in NumPy).
 - `CachedPoissonBlender`: calls `blend` but caches the Green function, so should be faster if you're repeatedly
blending source patches of equal size, as you will only need to construct the Green function once.
 - `blend_numpy`: A NumPy implementation of `blend`.
   - `blend_wide[_numpy]`: Wrappers of `blend` methods which allow for blending over entire image to more smoothly integrate the source region.
   
Why use it?
 - It's faster than any available alternative (OpenCV's `seamlessClone`, or manual solvers using iterative methods).
 - It's flexible, working on n-dimensional images, with no explicit limitations on data types
   (unlike `seamlessClone`, which only operates on 8-bit 3-channel images).
 - You using it makes me feel like I'm contributing to the world.

## Installation

### Using pip

```bash
pip install pie-torch
```

### Manually
Clone PIE-torch repository and install the package locally:

```bash
git clone https://github.com/matt-baugh/pytorch-poisson-image-editing.git
pip install -e ./pytorch-poisson-image-editing
```

You can check the installation by running the tests:

```bash
cd pytorch-poisson-image-editing/test
pytest
```

## Usage

Example of blending normal images using mixed gradients:
```python
from pietorch import blend

target : torch.Tensor = ... # 3 x N x M image to be blended into
source : torch.Tensor = ... # 3 x H x W image to be blended
mask : torch.Tensor = ... # H x W mask of which pixels from source to be included
corner : torch.Tensor = ... # [y, x] coordinate of location in target for source to be blended

result = blend(target, source, mask, corner, True, channels_dim=0)
```

## Examples
A wide variety of full examples (including how to create the two below) are given in [the examples notebook](https://github.com/matt-baugh/pytorch-poisson-image-editing/blob/master/examples.ipynb).

### 2D mixed gradient comparison with OpenCV's `seamlessClone`:

Target | Source 
:-------------------------:|:-------------------------:
![](./example_images/mug.png)  | <img src="./example_images/brick_texture.jpg" alt="drawing" width="512"/>

**Results**:

![](./example_images/mixed_gradients_blending_comparison.png)


### 3D mixed gradient blending over spatial and temporal dimensions:

Target | Source
:-------------------------:|:-------------------------:
![](./example_images/wave.gif)  | <img src="./example_images/splat.gif" alt="drawing" width="300" height="300"/>

**Result**:

![](./example_images/wave_splat.gif)