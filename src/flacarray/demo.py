# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""Helper functions for use in unit tests and interactive sessions."""

import numpy as np


def create_fake_data(shape, sigma=1.0, dtype=np.float64):
    flatshape = np.prod(shape)
    stream_size = shape[-1]
    leading_shape = shape[:-1]
    leading_shape_ext = leading_shape + (1,)

    rng = np.random.default_rng(seed=123456789)

    # Construct a random DC level for each stream that is +/- 5 sigma
    dc = 5 * sigma * (rng.random(size=leading_shape_ext) - 0.5)

    # Construct a simple low frequency waveform (assume 1Hz sampling)
    wave = np.zeros(stream_size, dtype=dtype)
    t = np.arange(stream_size)
    minf = 5 / stream_size
    for freq, amp in zip([3 * minf, minf], [2 * sigma, 6 * sigma]):
        wave[:] += amp * np.sin(2 * np.pi * freq * t)

    # Initialize all streams to a scaled version of this waveform plus the DC level
    scale = rng.random(size=leading_shape_ext)
    leading_slc = tuple([slice(None) for x in leading_shape])
    data = np.empty(shape, dtype=dtype)
    data[leading_slc] = dc
    data[leading_slc] += scale * wave

    # Add some Gaussian random noise to each stream
    data[:] += rng.normal(0.0, sigma, flatshape).reshape(shape)

    return data


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
