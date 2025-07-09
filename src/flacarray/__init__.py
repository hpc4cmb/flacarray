# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# FIXME: single-sourcing the package version from git without using
# setuptools build backend in pyproject.toml seems difficult...

import os


__version__ = "0.3.2"

from .array import FlacArray
