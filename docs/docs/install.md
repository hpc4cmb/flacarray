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

## Building From Source

In order to build from source, you will need a C compiler and the FLAC
development libraries installed.

### Building Within a Conda Environment

If you have conda available, you can create an environment will all the
dependencies you need to build flacarray from source. For this example, we
create an environment called "flacarray". First create the env with all
dependencies and activate it.  There is a list of conda requirements
provided in the source:

    conda create -n flacarray \
        --file packaging/conda_build_requirements.txt

    conda activate flacarray

Now you can go into your local git checkout of the flacarray source and do:

    pip install .

To build and install the package.

To also work on docs, install additional packages:

    conda install mkdocs mkdocstrings mkdocstrings-python mkdocs-jupyter
    pip install mkdocs-print-site-plugin

### Other Ways of Building

If you have a relatively recent system python3 and libFLAC provided by your
operating system, you can build flacarray using only OS tools. Flacarray
**requires libFLAC >= 1.4.0**. For example, on Debian-based systems, install
these packages:

    sudo apt update
    sudo apt install build-essential libflac-dev python3-dev python3-venv

Now create a virtualenv and activate it:

    python3 -m venv /path/to/env
    source /path/to/env/bin/activate

Next, go into the flacarray git checkout and install to the virtualenv:

    cd flacarray
    pip install .

### Running Tests

When building from source, you should definitely run the unit test suite after
installation. The tests are bundled in the package:

    python -c 'import flacarray.tests; flacarray.tests.run()'

## Using From Compiled Software

Flacarray ships with a low-level C library which can be linked against from compiled software.  First, install flacarray as described above.  Note that depending on how you build your compiled software, you may want to choose carefully how flacarray is installed.  For example, if you are compiling your software in a conda environment using the conda compiler toolchain, it is easiest if you install flacarray through the conda package.  If you are installing your software with the OS provided compiler, you may want to build flacarray from source with the same compiler.

### Linking to Flacarray from CMake

If you are using CMake to build your software, you can use the included FindFlacarray.cmake file in the top of the source tree.  This will set several environment variables.



### Other Build Systems

You can use the included flacarray_config script to print out the CFLAGS / LDFLAGS / LIBS needed to link to libflacarray and find the flacarray.h header.
