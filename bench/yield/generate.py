#!/bin/python

import os

MAX_BITWIDTH=@BITWIDTH@

benchmark_sets = [
        {
            "bitwidth": MAX_BITWIDTH,
            "name":  "b1-8192",
            "chain": False,
            "b1": 8192,
            "stage2": False,
            "b2": 0
        },
        {
            "bitwidth": MAX_BITWIDTH,
            "name":  "b1-5k-b2-5m",
            "chain": False,
            "b1": 50000,
            "stage2": True,
            "b2": 5000000
        },
        {
            "bitwidth": MAX_BITWIDTH,
            "name":  "b1-5kchain-b2-5m",
            "chain": True,
            "b1": "trpl-batch-50000",
            "stage2": True,
            "b2": 5000000
        },
        {
            "bitwidth": MAX_BITWIDTH,
            "name":  "miele",
            "chain": False,
            "b1": 256,
            "stage2": True,
            "b2": 16384
        },
        {
            "bitwidth": MAX_BITWIDTH,
            "name":  "bos-960",
            "chain": False,
            "b1": 960,
            "stage2": False,
            "b2": 0
        },
        {
            "bitwidth": MAX_BITWIDTH,
            "name":  "bos-8192",
            "chain": False,
            "b1": 8192,
            "stage2": False,
            "b2": 0
        }
        ]


def generate_config(filename, input, output, logfile,
                    b1, b2, stage2,
                    streams=8, threads_per_block=256, stage2_window = 2310):
    with open('./base.ini', "r") as baseconf, open(filename, "w") as conf:
        conf.write(baseconf.read().format(
            b1 = b1,
            b2 = b2,
            input = input,
            output = output,
            logfile = logfile,
            stage2 = "true" if stage2 else "false",
            stage2window = stage2_window,
            streams = streams,
            threads_per_block = threads_per_block))



prefix = os.path.abspath(os.path.join(os.path.dirname( __file__ )))
confprefix = os.path.join(prefix, "configs/")
try:
    os.makedirs(os.path.join(prefix, "output"))
except FileExistsError:
    pass
try:
    os.makedirs(os.path.join(prefix, "input"))
except FileExistsError:
    pass
try:
    os.makedirs(os.path.join(prefix, "log"))
except FileExistsError:
    pass


for param_set in benchmark_sets:
    try:
        os.makedirs(os.path.join(prefix, "configs", param_set["name"]))
    except FileExistsError:
        pass
    try:
        os.makedirs(os.path.join(prefix, "output", param_set["name"]))
    except FileExistsError:
        pass
    try:
        os.makedirs(os.path.join(prefix, "log", param_set["name"]))
    except FileExistsError:
        pass
    bwstr = "bw" + str(param_set["bitwidth"])
    if param_set["chain"]:
        b1str = "b1chain = " + param_set["b1"]
    else:
        b1str = "b1 = " + str(param_set["b1"])

    generate_config(os.path.join(confprefix, param_set["name"], bwstr + ".ini"),
                    os.path.join(prefix, "input", "bw" + str(param_set['bitwidth']) + ".input"),
                    os.path.join(prefix, "output", param_set["name"], bwstr),
                    os.path.join(prefix, "log", param_set["name"], bwstr),
                    b1str, param_set["b2"], param_set["stage2"],
                    )



