from setuptools import setup, find_packages

setup(
    name='RamanPL_2D',
    version='0.1.5',
    packages=find_packages(),
    install_requires=[  'numpy>=1.24.4',
                        'matplotlib>=3.5.2',
                        'scipy>=1.11.0',
                        'Pillow>=11.2.1',
                        'renishawWiRE>=0.1.16'
                        ],
)
