# Installation

For most use cases, you can just install `flacarray` from pre-built python
wheels or conda packages. For specialized use cases or development it is
straightforward to build the package from source using either a conda
environment for dependencies or with those obtained through your OS package
manager.

## Python Wheels

You can install pre-built wheels from PyPI using pip within a virtualenv:

    pip install flacarray

Or, if you are using a shared python environment you can install to a user
location with:

    pip install --user flacarray

## Conda Packages

If you are using a conda environment you can install the conda package for
`flacarray` from the conda-forge channels:

    conda install -c conda-forge flacarray

!!! note "To-Do"

    `flacarray` is not yet on conda-forge

## Building From Source

In order to build from source, you will need a C compiler and the FLAC
development libraries installed.

### Building Within a Conda Environment

If you have conda available, you can create an environment will all the
dependencies you need to build flacarray from source. For this example, we
create an environment called "flacarray". First create the env with all
dependencies and activate it (FIXME, add a requirements file for dev):

    conda create -n flacarray \
        c_compiler numpy libflac cython meson-python

    conda activate flacarray

Now you can go into your local git checkout of the flacarray source and do:

    pip install .

To build and install the package.

### Other Ways of Building

!!! note "To-Do"

    Discuss OS packages, document apt, rpm, homebrew options.
