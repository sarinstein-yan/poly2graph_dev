from setuptools import setup

VERSION = '0.0.1'
LONG_DESCRIPTION = """ TBD """

if __name__ == '__main__':
    setup(
        name='poly2graph',
        version=VERSION,
        url=''
        description='Automated Non-Hermitian Spectral Graph Extraction',
        long_description=LONG_DESCRIPTION,
        author='Xianquan Yan',
        author_email='xianquanyan@gmail.com',
        license='MIT',
        packages=['poly2graph', 'gnl_transformer'],
        install_requires=[
            'numpy',
            'numba',
            'networkx',
            'tensorflow',
            'torch',
            'torch-geometric',
            'torch-sparse',
            'torch-scatter',
            'torch-cluster',
            'torchmetrics',
            'scikit-image'
        ],
    )