from setuptools import setup, find_packages

setup(
    name='RamanPL_2D',
    version='1.0.0',
    packages=find_packages(),
    install_requires=['pandas', 
                      'matplotlib', 
                      'scipy', 
                      'numpy'
        # List any dependencies your package may have
    ],
)
