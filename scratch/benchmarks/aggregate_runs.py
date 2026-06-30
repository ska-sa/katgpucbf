#!/usr/bin/env python3

"""Aggregate the output of multiple runs of the benchmark.

Reads all .txt.n<n> files from the xbgpu_calibration results directory,
sums the last 3 columns per frequency, and writes an output file.

Example command to rename existing files for each run:
```ls ./ | xargs -I file sh -c "mv file file.n0"```
Then they can all be in the same directory and be aggregated:
    calibrate-n2-1.txt.n0
    calibrate-n2-1.txt.n1
"""

import argparse
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "directory",
    help="Directory containing .txt.n result files",
)
parser.add_argument(
    "-o",
    "--output",
    default="aggregated.txt",
    help="Output file path (default: aggregated.txt)",
)
args = parser.parse_args()

files = sorted(glob.glob(os.path.join(args.directory, "*")))
if not files:
    raise SystemExit(f"No .txt.n files found in {args.directory}")

sums: dict[float, list[int]] = {}
freq_order: list[float] = []
seen: set[float] = set()

for path in files:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            freq = float(parts[0])
            vals = [int(v) for v in parts[-3:]]
            if freq not in seen:
                freq_order.append(freq)
                seen.add(freq)
                sums[freq] = [0, 0, 0]
            for i in range(3):
                sums[freq][i] += vals[i]

with open(args.output, "w") as out:
    for freq in freq_order:
        vals = sums[freq]
        out.write(f"{freq} {vals[0]} {vals[1]} {vals[2]}\n")

print(f"Aggregated {len(files)} files, {len(freq_order)} frequencies -> {args.output}")
