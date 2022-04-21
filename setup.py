from pathlib import Path
from setuptools import setup

with open(Path(__file__).parent / 'README.md') as f:
    long_description = f.read()

setup(
    name='pie-torch',
    version='0.0',
    packages=['pietorch'],
    url='https://github.com/matt-baugh/pytorch-poisson-image-editing',
    license='',
    author='Matthew Baugh',
    author_email='matthew.baugh17@imperial.ac.uk',
    description='N-dimensional Poisson image editing implemented with Pytorch',
    long_description=long_description,
    install_requirements=[
        'torch>=1.7',  # torch.fft.fftn was added in 1.7, haven't actually tested lol.
        'scipy>=1.4',  # similar, scipy changed it's fft interface in version 1.4, haven't tested exact versions
        'numpy'
    ]
)
