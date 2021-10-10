#!/usr/bin/env python3
# Preparing and testing the the data

import sqlite3
import json
from datetime import datetime

timeframe = '2005-12'
spl_transaction = []
buffering = 1000

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

def find_existing_score(pid):
    try:
        query = "SELECT score from parent_reply WHERE parent_id = '{}' LIMIT 1".format(pid)
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
    with open(data_path, buffering=buffering) as data:
        for row in data:
            row_counter += 1
            row = json.loads(row)
            parent_id = row["parent_id"]
            body = format_data(row["body"])
            created_utc = row["created_utc"]
            score = row["score"]
            comment_id = row["name"]
            subreddit = row["subreddit"]
            parent_data = find_parent(parent_id)

            if score >= 2:
                existing_comment_score = find_existing_score(parent_id)
                # TODO: replace with comment that has the higher score
                # if existing_comment_score and score > existing_comment_score:

