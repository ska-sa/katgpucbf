#!/usr/bin/env python3
import json
from collections import defaultdict
from typing import Any

import pandas

columns: dict[str, dict[str, Any]] = defaultdict(dict)
with open("dropped2.json") as f:
    for line in f:
        data = json.loads(line)["performanceCounters"]
        for counter_obj in data["counters"]:
            counter = counter_obj["counter"]
            columns[counter["name"]][counter["timestamp"]] = counter["value"]
        for analysis_obj in data["analysis"]:
            counter = analysis_obj["analysisAttribute"]
            columns[counter["name"]][counter["timestamp"]] = counter["value"]

df = pandas.DataFrame(columns)
corr = df.corrwith(df["RX Buffer Full Port 1"])
print(corr.dropna().sort_values())
