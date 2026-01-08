#!/usr/bin/env python3

################################################################################
# Copyright (c) 2023, 2025, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""Update the autotuning database."""

import argparse
import logging
import os
import pathlib
import tempfile

import katsdpsigproc.accel

from katgpucbf.fgpu.ddc import DDCTemplate


def main() -> None:  # noqa: D103
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=pathlib.Path, help="Database filename")
    args = parser.parse_args()

    # Do all the autotuning in a temporary database, so that the
    # original is preserved in the event of a failure. Put it in
    # the same directory so that it can be moved atomicly.
    with (
        katsdpsigproc.accel.create_some_context(
            interactive=False,
            device_filter=lambda device: device.is_cuda,
        ) as context,
        tempfile.NamedTemporaryFile(
            prefix="tuning",
            suffix=".db",
            dir=args.database.parent,
            delete=False,
        ) as temp_db,
    ):
        should_delete = True
        try:
            os.environ["KATSDPSIGPROC_TUNE_DB"] = temp_db.name
            for taps_ratio in range(8, 33):
                for subsampling in [2, 4, 8, 16, 32, 64, 128]:
                    for input_sample_bits in range(2, 17):
                        DDCTemplate(context, taps_ratio * subsampling, subsampling, input_sample_bits)
            os.rename(temp_db.name, args.database)
            should_delete = False
        finally:
            # It would be more robust to use delete=True on
            # NamedTemporaryFile, but it doesn't have a way to indicate that
            # we've moved the file out from underneath.
            if should_delete:
                os.unlink(temp_db.name)


if __name__ == "__main__":
    main()
