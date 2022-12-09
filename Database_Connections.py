
import mysql.connector
from mysql.connector.constants import ClientFlag
import pandas as pd 
import numpy as np  


from Data_Retrieval import *


from pprint import pprint
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials


# from pandas.io import sql
# import MySQLdb

def connect_to_service():

    credentials = GoogleCredentials.get_application_default()

    service = discovery.build('sqladmin', 'v1beta4', credentials=credentials)

    return service



def print_GCP_DB_info(project, instance, database):
    
    service = connect_to_service()

    request = service.databases().get(project=project, instance=instance, database=database)
    response = request.execute()

    pprint(response)


def connect_to_GCP(password, host, database):
    
    config = {
        'user': 'root',
        'password': "{}".format(password), 
        'host': "{}".format(host), 
        "database": "{}".format(database)
    }

    # 'client_flags': [ClientFlag.SSL],
    #         'ssl_ca': 'ssl/server-ca.pem',
    #         'ssl_cert': 'ssl/client-cert.pem',
    #         'ssl_key': 'ssl/client-key.pem'

    # now we establish our connection
    cnxn = mysql.connector.connect(**config)

    return cnxn


def insert_data_into_GCP_table(dataframe, query, password, host, database, ):

    # https://towardsdatascience.com/sql-on-the-cloud-with-python-c08a30807661

    cnxn = connect_to_GCP(password, host, database)
    
    # sql.write_frame(dataframe, con=cnxn, 
    #             if_exists='replace', flavor='mysql')

    cursor = cnxn.cursor()
    if len(dataframe.columns) > 8:
        for index, row in dataframe.iterrows():
            cursor.execute(query,[row.predicted_price_change_rank,row.ticker,row.company_name,row.predicted_price_change,row.logo,row.sector,row.industry,row.description,row.ceo_name,row.exchange])
            cnxn.commit()
    else:
        for index, row in dataframe.iterrows():
            cursor.execute(query,[row.predicted_price_change_rank,row.one_day,row.three_day,row.five_day])
            cnxn.commit()
            
    # data_list = dataframe.to_records(index=False)
    # data_list = dataframe.values.tolist()

    # # then we execute with every row in our dataframe
    # cursor.executemany(query, data_list)
    #cursor.execute(query)
    # cnxn.commit()
    cnxn.close()



def delete_from_table(table_name, password, host, database):
    query = "DELETE FROM {}".format(table_name)

    cnxn = connect_to_GCP(password, host, database)
    cursor = cnxn.cursor()
    cursor.execute(query)
    cnxn.commit()
    cnxn.close()



from datetime import date

def clock(daily_update_queries):
    power_to_computer = True

    curr_day = date.today().strftime("%d")

    while power_to_computer:
        curr_day_checker = date.today().strftime("%d")
        
        if curr_day != curr_day_checker:
            
            for query in daily_update_queries:
                
                
                #data = pull_data_from_past()

                insert_data_into_GCP_table(query=query, dataframe=dataframe)

            curr_day = date.today().strftime("%d")