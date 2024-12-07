
incdir_numpy = run_command(py,
    [
      '-c',
      'import numpy as np; print(np.get_include())',
    ],
    check: true
).stdout().strip()

ext_sources = [
    'libflacarray.pyx',
    'utils.c',
    'compress.c',
    'decompress.c',
]

py.extension_module(
  'libflacarray',
  ext_sources,
  dependencies: [openmp, libflac],
  include_directories: [incdir_numpy],
  install: true,
  subdir: 'flacarray',
)
