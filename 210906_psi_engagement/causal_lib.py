import pandas as pd
import numpy as np
from typing import Any, Dict, List
import pickle
import time


def save_pickle(data:pd.DataFrame, file_name:str):
    """
    Saves dataframe as a pickle file in specified path
    Consider adding path as a parameter
    :param data: dataframe that is being saved
    :param file_name: name to be given to the saved file
    """
    with open("/home/ec2-user/SageMaker/ad_hoc/psi/psi_saved_data/"
                          +str(file_name)+".p",'wb') as f:
        pickle.dump(data,f)
    return 

def import_pickle(file_name):
    """
    Imports previously pickled file
    Consider adding path as a parameter
    :param file_name: name of file to be imported
    """
    with open("/home/ec2-user/SageMaker/ad_hoc/psi/psi_saved_data/"
                          +str(file_name)+".p",'rb') as f:
        df = pickle.load(f)
    return df
       
def create_snowflake_query(col_list:str, table:str) -> str:
    """
    Generates query with flexible select, from and where clause
    :param col_list: list of columns to pull from snowflake
    :param table: the name of referenced table
    :param random_number: random number assigned in snowflake for sampling
    """
    query = f"""
    select {", ".join(col_list)}
    from {table}"""
    return query

def bucket_continuous_features(continuous_feature:List[float], buckets:int) -> List[int]:
    return pd.qcut(continuous_feature, buckets, labels=np.arange(1,buckets+1))


