# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# FIXME: single-sourcing the package version from git without using
# setuptools build backend in pyproject.toml seems difficult...

import os
import logging

# Get the log level from the environment
log_level = os.getenv("FLACARRAY_LOG_LEVEL", "INFO")

# Set the log level
logging.basicConfig(level=log_level)

__version__ = "0.2.2"

from .array import FlacArray
