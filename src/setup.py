from setuptools import setup, find_packages
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="RamanPL_2D",
    version="0.4.6",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.4",
        "matplotlib>=3.5.2",
        "scipy>=1.11.0",
        "Pillow>=11.2.1",
        "renishawWiRE>=0.1.16",
        "pandas>=2.0.0",
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
    author="Hao Yu",
    description="Python toolkit for Raman and photoluminescence spectroscopy analysis in 2D materials.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/barry063/RamanPL_2D",
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
)