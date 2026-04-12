from setuptools import setup, find_packages
from pathlib import Path

_readme = (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")

setup(
    name='RamanPL_2D',
    version='0.4.2',
    author='Hao Yu',
    description='Raman and PL spectral fitting and 2D mapping utilities for 2D materials',
    long_description=_readme,
    long_description_content_type='text/markdown',
    url='https://github.com/barry063/RamanPL_2D',
    python_requires='>=3.9',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.4',
        'matplotlib>=3.5.2',
        'scipy>=1.11.0',
        'Pillow>=11.2.1',
        'renishawWiRE>=0.1.16',
        'pandas>=2.0.0',
    ],
    extras_require={
        "ramanspy": [
            "ramanspy>=0.2.10",
        ],
    },
    include_package_data=True,
    package_data={
        "ramanpl": ["raman_materials.json"],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
    ],
)
