
python_sources = [
    '__init__.py',
    'array.py',
    'compress.py',
    'decompress.py',
    'utils.py',
    'hdf5.py',
    'hdf5_utils.py',
    'hdf5_load_v0.py',
    'hdf5_load_v1.py',
    'mpi.py',
    'demo.py',
    'zarr.py',
    'zarr_load_v0.py',
    'zarr_load_v1.py',
    'io_common.py',
]

py.install_sources(
    python_sources,
    subdir: 'flacarray'
)

subdir('libflacarray')
subdir('tests')
subdir('scripts')
