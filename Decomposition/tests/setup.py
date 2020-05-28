import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="poldecomp",
    version="0.0.1",
    author="Raktim Ghosh",
    author_email="raktim.ghosh@ieee.org",
    description="This is poldecomp package",
    long_description="This program is written to implement the miscellaneous target decomposition "
                     "theorems (coherent and incoherent) in the domain of polarimetric synthetic aperture radar remote "
                     "sensing by utilizing the full polarimetric datasets (QuadPol)",
    long_description_content_type="text/markdown",
    url="https://github.com/raktim-ghosh",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)