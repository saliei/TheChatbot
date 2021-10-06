#! /usr/bin/env python3
# read bz2, xz, zst compressed reddit data line by line, and process json files
import pandas as pd
import numpy as np
import bz2
import json
import matplotlib.pyplot as plt

BUFF_SIZE = 10

filename = "RC_2005-12.bz2"

with bz2.open(filename, "rt") as mdata:
    lines = []
    for i, line in enumerate(mdata):
        if i == BUFF_SIZE: break
        comments = json.loads(line)
        lines.append(comments)




