# Connect to Snowflake
from abc import ABCMeta, abstractmethod
from oauth2client.service_account import ServiceAccountCredentials
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

from pandas import DataFrame as DataFrame
import io
import datetime
import pandas as pd
import sys
import time

root_path = sys.argv[1]  # a list of the arguments provided (str)
print(root_path)

class Credentials(metaclass=ABCMeta):
    pass
    
class SSMPSCredentials(Credentials):
    def __init__(self, secretid: str):
        self._secretid = secretid
        self._secrets = {}
        
    def get_keys(self):
        """
        credential fetching 
        """
        #_aws_sm_args = {'service_name': 'secretsmanager', 'region_name': 'us-east-1'}
        #secrets_client = boto3.client(**_aws_sm_args)
        #get_secret_value_response = secrets_client.get_secret_value(SecretId=self._secretid)
        #return get_secret_value_response
    
    
class BaseConnector(metaclass=ABCMeta):
    @abstractmethod
    def connect(self):
        raise NotImplementedError
        
class SnowflakeConnector(BaseConnector):
    def __init__(self, credentials: Credentials):
        keys = credentials.get_keys()
        #self._secrets = json.loads(keys.get('SecretString', "{}"))
    def connect(self, dbname: str, schema: str = 'DEFAULT'):
        ctx = snowflake.connector.connect(
            user='max.glue.dev',
            password='2p6R3teB8wm@9G',
            account='hbomax.us-east-1',
            #warehouse='MAX_DATASCIENCE_DEV',
            warehouse='max_analytics_user',
            database=dbname,
            schema=schema
        )
        return ctx
    
def execute_query(query: str):
    cur = con.cursor()
    try:
        cur.execute(query)
    finally:
        cur.close()
        
def truncate_table(sf_table_name):
    query = "TRUNCATE TABLE {}".format(sf_table_name)
    execute_query(query)

def save_df_to_sf(df, sf_table_name, date_columns = ['DATE','Date'], date_format = '%Y-%m-%d'):
    for c in date_columns:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.strftime(date_format)
    string='('
    for i in range(len(df.columns)):
        if i == len(df.columns) - 1:
            if df.dtypes[i] == 'int64':
                string=string+df.columns[i]+' INT)'
            elif df.dtypes[i] == 'float64':
                string=string+df.columns[i]+' FLOAT)'
            elif df.dtypes[i] == 'datetime64':
                string=string+df.columns[i]+' DATE)'
            else:
                string=string+df.columns[i]+' VARCHAR(255))'
        else:
            if df.dtypes[i] == 'int64':
                string=string+df.columns[i]+' INT,'
            elif df.dtypes[i] == 'float64':
                string=string+df.columns[i]+' FLOAT,'
            elif df.dtypes[i] == 'datetime64':
                string=string+df.columns[i]+' DATE,'
            else:
                string=string+df.columns[i]+' VARCHAR(255),'
    query = "CREATE OR REPLACE TABLE {}".format(sf_table_name) +string+";"
    execute_query(query)
    
    df.columns = map(str.upper, df.columns)
    write_pandas(con, df, sf_table_name)

def create_task(q, task_name, hour, minute):
    out = f'''
    CREATE OR REPLACE TASK {task_name}
    WAREHOUSE = MAX_ANALYTICS_USER
    SCHEDULE = 'USING CRON {minute} {hour} * * * America/Los_Angeles' as
    
    {q}
    
    alter task {task_name} resume;
    '''
    execute_query(out)

class DataDimensionException(Exception):
    def __init__(self, dataframe_name=None):
        self.msg = f"The {dataframe_name} table does not have matching number of columns with source" if dataframe_name else ""    
    def __str__(self):
        return self.msg

def run_sql(file_path):
    file = open(file_path, "r")
    sql = file.read()
    file.close()
    queries = sql.split(';')
    n_queries = len([q for q in queries if len(q)>0 ])
    tic = time.perf_counter()
    for q in queries:
        #execute_query(q)
        if len(q)>0:
            pd.read_sql(q, con)
    toc = time.perf_counter()
    print(f"Executed {file_path} | {n_queries} queries | {toc - tic:0.4f} seconds")


DB_NAME = "MAX_DEV"
SCHEMA = "WORKSPACE"


## Credentials
SF_CREDS = 'datascience-max-dev-sagemaker-notebooks'
connector = SnowflakeConnector(SSMPSCredentials(SF_CREDS))
#connector = SnowflakeConnector()
con = connector.connect(dbname=DB_NAME, schema=SCHEMA)

print ("Snowflake Connection Successful")