import datetime
import sys
import time
import pandas as pd

import krakenex
import pykrakenapi as pka
import mysql.connector
import pymysql
from sqlalchemy import create_engine, text

import cryptos

# config
KEY = ''
SECRET = ''

SQL_NAME = 'trader'
SQL_PW = 'trade_good'
SQL_SERVER = 'vserver'
SQL_DATABASE = 'trading_data'

PAIRS = cryptos.PAIRS

def update_db(engine, df, cur):
    with engine.connect() as connection:
        # get last date
        last_time = pd.read_sql(text(f'SELECT time FROM ohcl WHERE currency="{cur}" ORDER BY time DESC LIMIT 1'), connection)
        if len(last_time) != 0:
            last_time = last_time.iloc[0]['time']
            df = df[df['time'] > last_time]
        df.to_sql('ohcl', connection, if_exists='append', index=False)
        connection.commit()
        return len(df)


# init api
api = krakenex.API(KEY, SECRET)
k = pka.KrakenAPI(api)


while True:
    time_format = '%d.%m.%Y %H:%M:%S'
    print('start update cycle', time.strftime(time_format))

    print('reading pairs.txt...')
    file = open('pairs.txt', 'r')
    currencies = [cur[:6] for cur in file.readlines()]
    file.close()

    print("currencies to be updated: ", currencies)

    # establish sql connection
    engine = create_engine(f'mysql+pymysql://{SQL_NAME}:{SQL_PW}@{SQL_SERVER}/{SQL_DATABASE}')

    for cur in currencies:
        # load last half day
        ohlc, last = k.get_ohlc_data(cur)
        ohlc['currency'] = cur
        ohlc['time'] = pd.to_datetime(ohlc['time'], unit='s')

        # save new entries to db
        num = update_db(engine, ohlc, cur)

        print('updated ' + cur + ' entries (' + str(num) + ' new entries)')

        time.sleep(1)

    sleepy_time_s = 60*60*10
    print('hibernating till', (datetime.datetime.now() + datetime.timedelta(seconds=sleepy_time_s)).strftime(time_format))
    time.sleep(sleepy_time_s)
    print('rise and shine')

