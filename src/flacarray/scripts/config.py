# Copyright (c) 2026-2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import os
import sysconfig

# We use absolute imports to find the location of the installed
# package.
import flacarray


def cli():
    parser = argparse.ArgumentParser(description="Print configuration of flacarray")
    parser.add_argument(
        "--version",
        required=False,
        default=False,
        action="store_true",
        help="Print the package version",
    )
    parser.add_argument(
        "--package",
        required=False,
        default=False,
        action="store_true",
        help="Print the package install location",
    )
    parser.add_argument(
        "--cflags",
        required=False,
        default=False,
        action="store_true",
        help="Print the include CFLAGS (-I...)",
    )
    parser.add_argument(
        "--include",
        required=False,
        default=False,
        action="store_true",
        help="Print the directory containing flacarray.h",
    )
    parser.add_argument(
        "--ldflags",
        required=False,
        default=False,
        action="store_true",
        help="Print the linking LDFLAGS (-L...)",
    )
    parser.add_argument(
        "--libs",
        required=False,
        default=False,
        action="store_true",
        help="Print the linking libraries (-lflacarray)",
    )
    parser.add_argument(
        "--lib",
        required=False,
        default=False,
        action="store_true",
        help="Print the path to libflacarray",
    )

    args = parser.parse_args()

    version = flacarray.__version__
    pkg_root = os.path.dirname(flacarray.__file__)
    pkg_incdir = os.path.join(pkg_root, "include")
    pkg_libdir = os.path.join(pkg_root, "lib")

    suffix = sysconfig.get_config_var("SHLIB_SUFFIX")

    if args.version:
        print(version)
    elif args.package:
        print(pkg_root)
    elif args.include:
        print(pkg_incdir)
    elif args.cflags:
        print(f"-I{pkg_incdir}")
    elif args.ldflags:
        print(f"-L{pkg_libdir}")
    elif args.libs:
        print("-lflacarray")
    elif args.lib:
        print(os.path.join(pkg_libdir, f"libflacarray{suffix}"))
    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
