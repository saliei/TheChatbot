#!/usr/bin/env python3
# Preparing and testing the the data

import sqlite3
import json
from datetime import datetime

timeframe = '2005-12'
spl_transaction = []

connection = sqlite3.connect('{}.db'.format(timeframe))
print(connection.total_changes)
query = "CREATE TABLE IF NOT EXISTS parent_reply(parent_id TEXT PRIMARY KEY,\
        comment_id TEXT UNIQUE, parent TEXT, comment TEXT, subreddit TEXT,\
        unix INT, score INT)"
c = connection.cursor()

def create_table():
    c.execute(query)

def format_data(data):
    data = data.replace('\n', ' newlinechar ').replace('\r', ' newlinechar ').replace('"', "'")
    return data

def find_parent(pid):
    try:
        query = "SELECT comment FROM parent_reply WHERE comment_id = '{}' LIMIT 1".format(pid)
        c.execute(query)
        result = c.fetchone()
        if result != None:
            return result[0]
        else:
            return False
    except Exception as err:
            return False

if __name__ == '__main__':
    create_table()
    row_counter = 0
    paired_rows = 0
    
    data_path = "../data/RC_{}".format(timeframe.split('-')[0], timeframe)
    with open()
