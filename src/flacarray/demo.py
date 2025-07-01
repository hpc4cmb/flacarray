# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""Helper functions for use in unit tests and interactive sessions."""

import numpy as np

from .mpi import global_array_properties


def create_fake_data(
    local_shape, sigma=1.0, dtype=np.float64, seed=123456789, comm=None, dc_sigma=5,
):
    """Create fake random data for testing.

    This is a helper function to generate some random data for testing.
    if `sigma` is None, uniform randoms are return.  If sigma is not None,
    samples drawn from a Gaussian distribution are returned.

    If `comm` is not None, the data is created on one process and then pieces are
    distributed among the processes.

    Args:
        local_shape (tuple):  The local shape of the data on this process.
        sigma (float):  The width of the distribution or None.
        dtype (np.dtype):  The data type of the returned array.
        seed (int):  The optional seed for np.random.
        comm (MPI.Comm):  The MPI communicator or None.

    Returns:
        (tuple):  (The random data on the local process, MPI distribution).

    """
    if comm is None:
        rank = 0
    else:
        rank = comm.rank

    # Get the global array properties
    gprops = global_array_properties(local_shape, comm)
    shape = gprops["shape"]
    mpi_dist = gprops["dist"]

    flatshape = np.prod(shape)
    stream_size = shape[-1]
    leading_shape = shape[:-1]
    leading_shape_ext = leading_shape + (1,)

    rng = np.random.default_rng(seed=seed)
    global_data = None
    if rank == 0:
        if sigma is None:
            # Uniform randoms. Verify that we can fully encode the high / low
            # values by setting a few samples to those extremes.
            if dtype == np.dtype(np.int64) or dtype == np.dtype(np.int32):
                low = np.iinfo(dtype).min
                high = np.iinfo(dtype).max
                flat_data = rng.integers(
                    low=low, high=high, size=flatshape, dtype=np.int64
                ).astype(dtype)
            else:
                low = np.finfo(dtype).min
                high = np.finfo(dtype).max
                flat_data = rng.uniform(
                    low=low, high=high, size=flatshape, dtype=np.float64
                ).astype(dtype)
            flat_data[0] = low
            flat_data[1] = high
            global_data = flat_data.reshape(shape)
        else:
            # Construct a random DC level for each stream.
            if dc_sigma is None:
                dc = 0
            else:
                dc = dc_sigma * sigma * (rng.random(size=leading_shape_ext) - 0.5)

            # Construct a simple low frequency waveform (assume 1Hz sampling)
            wave = np.zeros(stream_size, dtype=dtype)
            t = np.arange(stream_size)
            minf = 5 / stream_size
            for freq, amp in zip([3 * minf, minf], [2 * sigma, 6 * sigma]):
                wave[:] += amp * np.sin(2 * np.pi * freq * t)

            # Initialize all streams to a scaled version of this waveform plus
            # the DC level
            scale = rng.random(size=leading_shape_ext)
            global_data = np.empty(shape, dtype=dtype)
            if len(leading_shape) == 0:
                global_data[:] = dc
                global_data[:] += scale * wave
            else:
                leading_slc = tuple([slice(None) for x in leading_shape])
                global_data[leading_slc] = dc
                global_data[leading_slc] += scale * wave

            # Add some Gaussian random noise to each stream
            global_data[:] += rng.normal(0.0, sigma, flatshape).reshape(shape)
    if comm is not None:
        global_data = comm.bcast(global_data, root=0)

    # Extract our local piece of the global data
    if len(leading_shape) == 0 or (len(leading_shape) == 1 and leading_shape[0] == 1):
        data = global_data
    else:
        local_start = mpi_dist[rank][0]
        local_stop = mpi_dist[rank][1]
        local_slice = [slice(local_start, local_stop, 1)]
        local_slice.extend([slice(None) for x in shape[1:]])
        local_slice = tuple(local_slice)
        data = global_data[local_slice]
    if len(data.shape) == 2 and data.shape[0] == 1:
        data = data.reshape((-1))

    return data, mpi_dist


def plot_data(data, keep=None, stream_slc=slice(None), file=None):
    # We only import matplotlib if we are actually going to make some plots.
    # This is not a required package.
    import matplotlib.pyplot as plt

    if len(data.shape) > 3:
        raise NotImplementedError("Can only plot 1D and 2D arrays of streams")

    if len(data.shape) == 1:
        plot_rows = 1
        plot_cols = 1
    elif len(data.shape) == 2:
        plot_rows = data.shape[0]
        plot_cols = 1
    else:
        plot_rows = data.shape[1]
        plot_cols = data.shape[0]

    fig_dpi = 100
    fig_width = 6 * plot_cols
    fig_height = 4 * plot_rows
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)
    if len(data.shape) == 1:
        # Single stream
        ax = fig.add_subplot(1, 1, 1, aspect="auto")
        ax.plot(data[stream_slc])
    elif len(data.shape) == 2:
        # 1-D array of streams, plot vertically
        for iplot in range(data.shape[0]):
            ax = fig.add_subplot(plot_rows, 1, iplot + 1, aspect="auto")
            ax.plot(data[iplot, stream_slc])
    else:
        # 2-D array of streams, plot in a grid
        for row in range(plot_rows):
            for col in range(plot_cols):
                slc = (col, row, stream_slc)
                ax = fig.add_subplot(
                    plot_rows, plot_cols, row * plot_cols + col + 1, aspect="auto"
                )
                ax.plot(data[slc], color="black")
    if file is None:
        plt.show()
    else:
        plt.savefig(file)
        plt.close()
