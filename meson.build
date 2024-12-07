project(
  'flacarray',
  'c', 'cython',
  license: 'BSD-2-Clause',
  meson_version: '>= 1.0.0',
  default_options: [
    'buildtype=release',
    'c_std=c11',
  ]
)

openmp = dependency('openmp')
libflac = dependency('flac', version: '>= 1.4.0', static: false)

py = import('python').find_installation(pure: false)

subdir('flacarray')
