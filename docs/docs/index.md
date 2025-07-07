# FLACArray

This package provides a set of tools for compressing multi-dimensional arrays
where the last array dimension consists of "streams" of data. These streams are
compressed with the FLAC algorithm and can be written to different file formats
as well as decompressed back into numpy arrays.

[FLAC compression](https://xiph.org/flac/) is particularly suited to "noisy"
timestreams that do not compress well with DEFLATE algorithms used by zip /
gzip. This type of data is found in audio signals, scientific timestreams, etc.

In the `flacarray` package we use only a small subset of features found in the
libFLAC library. In particular, each data stream is compressed as either one or
two 32 bit "channels". Stream data consisting of 32 or 64 bit integers are
compressed in a loss-less fashion. Floating point data is converted to either 32
or 64 bit integers with a user-specified precision or quantization.

If you are specifically working with audio data and want to write flac format
audio files, you should look at other software tools such as
[pyflac](https://github.com/sonos/pyFLAC).
