import sys
import csv

report = sys.argv[1]

class Table:

    def add_row(self, *args):
        print("|", " | ".join([a.rjust(10) for a in args]), "|")

    def add_header(self, *args):
        print("|", " | ".join([a.rjust(10) for a in args]), "|")
        print("="*(13*len(args)+1))

with open(report, "r") as rep:
    print("\n")
    read = csv.reader(rep, delimiter=',')
    tbl = Table()
    tbl.add_header("Benchmark", "Bitwidth", "Throughput", "Curves", "Time", "#Input", "#Solved", "Yield")
    for row in read:
        if int(row[6]) != 0:
            y = "{:1.02f}".format(int(row[7])/int(row[6]))
        else:
            y = "--"
        tbl.add_row(row[1], row[2], row[3], row[4], row[5], row[6], row[7], y)
    print("\n")

