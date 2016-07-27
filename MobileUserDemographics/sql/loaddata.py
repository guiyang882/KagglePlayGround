import MySQLdb
import numpy as np
import pandas as pd
import csv
from DBUtils.PooledDB import PooledDB

pool = PooledDB(MySQLdb, 5, host='10.2.3.119', user='fighter', passwd='fighter', db='db_mobileuser', port=3306)
db = pool.connection()
cur = db.cursor()

with open('../data/table_prepare_app.csv', 'r') as handle:
    reader = csv.reader(handle)
    index = 0
    for line in reader:
        index += 1
        if index == 1:
            continue
        str_sql = '''insert into t_app_type values('%s', %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d);'''\
                  % (line[0],
                     int(float(line[1])), int(float(line[2])), int(float(line[3])),
                     int(float(line[4])), int(float(line[5])), int(float(line[6])),
                     int(float(line[7])), int(float(line[8])), int(float(line[9])),
                     int(float(line[10])), int(float(line[11])), int(float(line[12])),
                     int(float(line[13])), int(float(line[14])), int(float(line[15])),
                     int(float(line[16])), int(float(line[17])), int(float(line[18])),
                     int(float(line[19])))
        print str_sql
        cur.execute(str_sql)
    db.commit()
cur.close()
db.close()
