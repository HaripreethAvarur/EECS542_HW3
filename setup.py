from setuptools import setup, find_packages

print('Found packages:', find_packages())
setup(
    description='HMR2 as a package',
    name='hmr2',
    packages=find_packages(),
    install_requires=[
        'gdown',
        'numpy==1.26.4',
        'torch==1.13.1',
        'torchvision==0.14.1',
        'pytorch-lightning==1.9.5',
        'smplx==0.1.28',
        'pyrender',
        'opencv-python',
        'yacs',
        'scikit-image',
        'einops',
        'timm==0.6.12',
        'webdataset',
        'dill',
        'pandas',
        'chumpy @ git+https://github.com/mattloper/chumpy',
    ],
    extras_require={
        'all': [
            'detectron2 @ git+https://github.com/facebookresearch/detectron2',
        ],
    },
)
