try:
    import torch
    import torch_geometric
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "PyTorch and/or PyG not installed. "
        "Please appropriately install PyTorch and PyG, "
        "see https://pytorch.org/ and https://pytorch-geometric.readthedocs.io/en/latest/ for installation."
    )

try:
    import tensorflow
except ModuleNotFoundError:
    # print(
    raise ModuleNotFoundError(
        "TensorFlow not installed. "
        "Poly2Graph will be significantly slow without TensorFlow. "
        "See https://www.tensorflow.org/ for installation."
    )

from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

if __name__ == '__main__':
    setup(
        name='poly2graph',
        version='0.0.1',
        author='Xianquan (Sarinstein) Yan',
        author_email='xianquanyan@gmail.com',
        url='https://github.com/sarinstein-yan/spectral-topology'
        description='Automated Non-Hermitian Spectral Graph Extraction',
        long_description=long_description,
        long_description_content_type='text/markdown',
        license='MIT',
        license_files='LICENSE',
        packages=find_packages(),
        python_requires='~=3.9',
        install_requires=[
            'numpy',
            'numba',
            'networkx',
            'scikit-image',
            'torchmetrics'
        ],
        classifiers=[
            "Development Status :: 1 - Beta",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.9"
        ]
    )