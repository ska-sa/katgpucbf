#!/usr/bin/env python3

import argparse
import json
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


class Queue:
    def __init__(self, name, maxsize):
        self.name = name
        self.maxsize = maxsize
        self.x = []
        self.y = []

    def add(self, x, y):
        self.x.append(x)
        self.y.append(y)


class StateMachine:
    def __init__(self, name):
        self.name = name
        self.x = []
        self.y = []

    def add(self, x, y):
        self.x.append(x)
        self.y.append(y)


parser = argparse.ArgumentParser()
parser.add_argument("filename")
args = parser.parse_args()

events = []
with open(args.filename, "r") as f:
    for line in f:
        if not line.endswith("\n"):
            break
        events.append(json.loads(line))

events.sort(key=lambda event: event["time"])
queues = {}
queue: Optional[Queue]
state_machines = {}
for data in events:
    if data["type"] == "qsize":
        name = data["name"]
        if name not in queues:
            queues[name] = Queue(name, data["maxsize"])
        queue = queues[name]
        if data["maxsize"] != queue.maxsize:
            raise RuntimeError(f'Queue {name} changed ({queue.maxsize} -> {data["maxsize"]}')
        queues[name].add(data["time"], data["qsize"])
    elif data["type"] == "qsize-delta":
        name = data["name"]
        queue = queues.get(name)
        if not queue:
            raise KeyError(f"qsize-delta on {name} requires the queue to be defined first")
        queue.add(data["time"], queue.y[-1] + data["delta"])
    elif data["type"] == "state":
        name = data["name"]
        if name not in state_machines:
            state_machines[name] = StateMachine(name)
        state_machines[name].add(data["time"], data["state"])

fig, axes = plt.subplots(len(queues) + len(state_machines), 1, sharex=True)
queue_axes = axes[: len(queues)]
state_machine_axes = axes[len(queues) :]
for ax, queue in zip(queue_axes, queues.values()):
    ax.axhline(queue.maxsize, color="r", alpha=0.3)
    ax.plot(queue.x, queue.y, drawstyle="steps-post", label=queue.name)
    ax.set_xticks(np.arange(0, queue.x[-1], 4096 * 16384 / 1712000000))
    ax.legend(loc="upper left")
for ax, state_machine in zip(state_machine_axes, state_machines.values()):
    ax.plot(state_machine.x, state_machine.y, drawstyle="steps-post", label=state_machine.name)
    ax.set_xticks(np.arange(0, queue.x[-1], 4096 * 16384 / 1712000000))
    ax.legend(loc="upper left")
plt.show()
