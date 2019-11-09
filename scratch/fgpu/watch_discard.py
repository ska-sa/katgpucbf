#!/usr/bin/env python3

import argparse
import re
import subprocess
import time
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('interface')
args = parser.parse_args()

last = 0
while True:
    result = subprocess.run(['ethtool', '-S', args.interface],
                            check=True, encoding='utf-8',
                            stdout=subprocess.PIPE,
                            stdin=subprocess.DEVNULL)
    discards = None
    for line in result.stdout.splitlines():
        match = re.search(r'rx_discards_phy: (\d+)', line)
        if match:
            discards = int(match.group(1))
            break
    if discards is None:
        print("WARNING: couldn't find discard stats")
        continue
    if discards != last:
        ts = datetime.now().isoformat()
        delta = discards - last
        print(f'{ts}: discards: {discards} (+{delta})')
        last = discards
    time.sleep(1)
