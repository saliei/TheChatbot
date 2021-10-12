#!/usr/bin/env python3
# Preparing and testing the the data

import sqlite3
import json
from datetime import datetime

timeframe = '2009-06'
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
        print("During handlig `find_parent` exceptioin occured: {}".format(err))
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
        print("During handling `find_existing_score` exception occured: {}".format(err))
        return False

def acceptable(data):
    if len(data.split(' ')) > 50 or len(data) < 1:
        return False
    elif len(data) > 1000:
        return False
    elif data == '[deleted]' or data == '[removed]':
        return False
    else:
        return True

def sql_insert_replace_comment(commentid, parentid, parent, comment, subreddit, time, score):
    try:
        query = """UPDATE parent_reply SET parent_id = ?, comment_id = ?, parent = ?, comment = ?, subreddit = ?, unix = ? WHERE parent_id =?;""".format(parentid, commentid, parent, comment, subreddit, int(time), score, parentid)
        transcation_bldr(query)
    except Exception as err:
        print("s0 insertion", str(err))

def sql_insert_has_parent(commentid, parentid, parent, comment, subreddit, time, score):
    try:
        query = """INSERT INTO parent_reply (parent_id, comment_id, parent, comment, subreddit, unix, score) VALUES ("{}", "{}", "{}", "{}", "{}", {}, {})""".format(parentid, commentid, parent, comment, subreddit, int(time), score)
        transcation_bldr(query)
    except Exception as err:
        print("s0 insertion", str(err))

def sql_insert_no_parent(commentid, parentid, comment, subreddit, time, score):
    try:
        query = """INSERT INTO parent_reply (parent_id, comment_id, comment, subreddit, unix, score) VALUES ("{}", "{}", "{}", "{}", {}, {});""".format(parentid, commentid, comment, subreddit, int(time), score)
        transcation_bldr(query)
    except Exception as err:
        print("s0 insertion", str(err))

def transcation_bldr(query):
    global sql_transaction
    sql_transaction.append(query)
    if len(sql_transaction) > 1000:
        c.execute('BEGIN TRANSACTION')
        for q in sql_transaction:
            try:
                c.execute(q)
            except:
                pass
            connection.commit()
            sql_transaction = []

if __name__ == '__main__':
    create_table()
    row_counter = 0
    paired_rows = 0
    
    # data_path = "../data/RC_{}".format(timeframe.split('-')[0], timeframe)
    data_path = "../data/RC_{}".format('-'.join(timeframe.split('-')), timeframe)
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

            # if score >= 2:
                # existing_comment_score = find_existing_score(parent_id)
                # if existing_comment_score:
                    # if score > existing_comment_score:
                        # if acceptable(body):
                            # sql_insert_replace_comment(comment_id, parent_id, body, subreddit, created_utc, score)
                # else:
                    # if acceptable(body):
                        # if parent_data:
                            # sql_insert_has_parent(comment_id, parent_id, parent_data, body, subreddit, created_utc, score)
                            # paired_rows += 1
                        # else:
                            # sql_insert_no_parent(comment_id, parent_id, body, subreddit, created_utc, score)

            if row_counter % 100000 == 0:
                print("Total row read: {}, Paired row: {}, Time: {}".format(row_counter, paired_rows, str(datetime.now())))


