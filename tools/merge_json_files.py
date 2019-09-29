#! /usr/bin/env python3

import json
import glob
import sys


def main():
    if len(sys.argv) < 3:
        print("usage: merge_json file1 file2 ...")
        sys.exit(1)

    accum = None
    files = sys.argv[1:]
    for filepath in files:
        try:
            with open(filepath, "rb") as f:
                j = json.loads(f.read().decode("utf-8"))
                if accum == None:
                    accum = j
                else:
                    accum["benchmarks"] += j["benchmarks"]
        except:
            None

    with open(dir + ".json", 'w') as fp:
        json.dump(accum, fp, indent=4)


if __name__ == "__main__":
    main()
