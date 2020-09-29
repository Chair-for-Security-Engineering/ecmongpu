import sys
import re
import os.path

stage = sys.argv[1]
logfile = sys.argv[2]
output = sys.argv[3]
report = sys.argv[4]

with open(logfile, "r") as log:
    lines = log.readlines()
    total = [line for line in lines if "Total" in line]
    totalre = r"(?P<date>[0-9:\s\.-]*).*\[Total\].*\:\D*(?P<num>\d*)\D*(?P<time>\d*)ms .*\((?P<throughput>\d*) c\/s\)"
    if not total:
        exit(-1)
    else:
        m = re.match(totalre, total[0])

with open(output, "r") as output:
    tasks = 0
    taskssolved = 0
    for line in output.readlines():
        ls = line.split(" ")
        if ls[0].isdigit():
            tasks += 1
        if len(ls) > 1 and int(ls[1]) > 1:
            taskssolved += 1



with open(report, "a") as out:
    out.write(", ".join([m.group('date').strip(), stage, os.path.basename(logfile),
               m.group('throughput'), m.group('num'), m.group('time'), str(tasks), str(taskssolved)]) + "\n")

