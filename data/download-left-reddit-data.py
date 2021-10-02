#! /bin/env python3

import pandas as pd
import subprocess

url = "https://files.pushshift.io/reddit/comments/"
dfs = pd.read_html(url)
df = dfs[0]["Filename"]

# missing reddit data from torrent download
df = df.drop(df.index[25:109])
df = df.reset_index(drop=True)

df = df.drop(df.index[104:])
df = df.reset_index(drop=True)



url_dl_str = "https://files.pushshift.io/reddit/comments/"
df = df.apply(lambda file: url_dl_str + file)

for file in df:
    download = subprocess.run(["aria2c", "-x16", "{}".format(file)], \
            check=True, stdout=subprocess.PIPE)
    print(download.stdout.decode('utf-8'))

