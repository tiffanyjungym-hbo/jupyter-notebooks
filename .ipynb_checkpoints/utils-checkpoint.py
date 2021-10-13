import numpy as np
import math
import pandas as pd
import re
from datetime import date, datetime, timedelta

## Snowflake
import json
from abc import ABCMeta, abstractmethod
import boto3

#Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
import plotly as py
import plotly.graph_objs as go
py.offline.init_notebook_mode(connected=True)
from IPython.display import display, HTML
import plotly.io as pio


## Other
from functools import partial, reduce
from collections import Counter
import yaml 
import shap
import pickle
shap.initjs()
#import scipy.stats as stats
import scipy.stats as stats
import matplotlib.cm as cm
import itertools
import random
from collections import Counter 
from collections import defaultdict, namedtuple
from collections import OrderedDict

## ML Algorithms 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import *
#import xgboost as xgb
# import lightgbm as lgbm
# import category_encoders as cecc

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 1000)
# Rename columns 
percents=[0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,1]


##https://htmlcolorcodes.com/color-chart/
colorlist = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf',  # blue-teal
    '#ff4500',  # orangered
    '#228b22',  # forest green
    '#db7093',  # palevioletred
    '#800080',  # purple
    '#CD853F',  # peru
    '#FF1493',  # deeppink
    '#696969',  # dimgray
    '#BDB76B',  # darkkhaki
    '#7FFFD4',  # aquamarine
    '#708090',  # slategray
    '#6B8E23',  # olivedrab
    '#20B2AA',  # lightseagreen
    '#FF7F50',  # coral
    '#00FF7F',  # springgreen
    '#8B4513',  # saddlebrown
    '#8B008B',  # darkmagenta
    '#FFE4B5'   # moccasin    
    
]
# Paths 


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
        _aws_sm_args = {'service_name': 'secretsmanager', 'region_name': 'us-east-1'}
        secrets_client = boto3.client(**_aws_sm_args)
        get_secret_value_response = secrets_client.get_secret_value(SecretId=self._secretid)
        return get_secret_value_response
    
    
class BaseConnector(metaclass=ABCMeta):
    @abstractmethod
    def connect(self):
        raise NotImplementedError
        

class SnowflakeConnector(BaseConnector):
    def __init__(self, credentials: Credentials):
        keys = credentials.get_keys()
        self._secrets = json.loads(keys.get('SecretString', "{}"))

    def connect(self, dbname: str, schema: str = 'DEFAULT'):
        ctx = snowflake.connector.connect(
            user=self._secrets['login_name'],
            password=self._secrets['login_password'],
            account=self._secrets['account'],
            warehouse=self._secrets['warehouse'],
            database=dbname,
            schema=schema
        )

        return ctx
    

## share variables across notebooks 
# %store

def get_plot(dflist, varlist, labellist=None,  ## Old
              title=None, config={}, x_var='order_date', mode='lines'):
    """ 
    How to use:
        kwargs={'dflist':[df_1, df_2]  ## List of dataframes 
                ,'varlist':['var1', 'var2'] ## Variables to be plotted
                ,'title':'var1 & var2 for df_1 and df_2'  ## Plot title
                ,'labellist':['df_1_var_1', 'df_1_var_2, 'df_2_var_1, 'df_2_var_2'] ## Labels to be shown in legend
                } ## x_var set to order_date by default
        fig = get_plot(**kwargs)
    1. Creates dataframe x variable combination
    2. For each data trace, creates scatter-line plot
    3. Plots and returns fig object for interactive plot edit
    
    To edit on chart editor: 
    import plotly.io as pio
    pio.write_json(fig, 'chart_1.json') Right click on json file, open with plotly chart editor. 
    """

    ## Create dataframe x var combinations (plot all variables for each dataframe)
    dfvarlist=itertools.product(*[dflist,varlist])

    ## Get list of labels
    if labellist==None:
        labellist=varlist
        
    ## For every dataframe x variable, create a data trace for plotting
    data=[]    
    for dfvar,name in zip(dfvarlist, labellist):
        dfplt,var=dfvar[0], dfvar[1]
        if x_var=='order_date':
            try:
                dfplt.reset_index(inplace=True)
            except:
                pass
            dfplt.set_index(pd.DatetimeIndex(dfplt['order_date']), inplace=True)
        data = data + [go.Scatter(x=dfplt[x_var], y=dfplt[var], mode=mode, name=name)]  

    ## Initiate offline plot and create plot
    py.offline.init_notebook_mode(connected=True) 
    layout = dict(title = title,
                  xaxis = dict(title = x_var), #, range=['2017-09-01','2017-02-01']
                  yaxis = dict(title = title),
                  autosize=False,
                  width=(900 if len(config)==0 else 600),
                  height=(450 if len(config)==0 else 300),
                  showlegend=True
                 )
    fig = dict(data=data, layout=layout)
    py.offline.iplot(fig, config=config)
    return fig



def get_plot_small(dflist, varlist, labellist=None,  ## Old
              title=None, config={}, x_var='order_date', y_name=None,  mode='lines'):
    ## Create dataframe x var combinations (plot all variables for each dataframe)
    dfvarlist=itertools.product(*[dflist,varlist])

    ## Get list of labels
    if labellist==None:
        labellist=varlist
        
    ## For every dataframe x variable, create a data trace for plotting
    data=[]    
    for dfvar,name in zip(dfvarlist, labellist):
        dfplt,var=dfvar[0], dfvar[1]
        if x_var=='order_date':
            try:
                dfplt.reset_index(inplace=True)
            except:
                pass
            dfplt.set_index(pd.DatetimeIndex(dfplt['order_date']), inplace=True)
        data = data + [go.Scatter(x=dfplt[x_var], y=dfplt[var], mode=mode, name=name)]  

    ## Initiate offline plot and create plot
    py.offline.init_notebook_mode(connected=True) 
    
    layout = {"height":300, "showlegend":True,
           "title":{"text":title,"font":{"size":14}},
           "width":500,
           "xaxis":{"title":{"text":"Order Date","font":{"size":12}},
                    "type":"date",
                    "tickfont":{"size":10}},
           "yaxis":{"title":{"text":y_name,"font":{"size":12}},
                    "type":"linear",
                    "tickfont":{"size":10}},
           "margin":{"t":40,"b":40,"l":60,"r":0},
           "legend":{'yanchor':"top",
                     "xanchor":"left",
                     "x":0.9,"y":1}}

    fig = dict(data=data, layout=layout)
    py.offline.iplot(fig, config=config)
    return fig

def get_plot_daxes(dflist, varlist, varlist_2=None, labellist=None, labellist_2=None,
              title=None, config={}, x_var='order_date', mode='lines'):
    """ 
    How to use:
        kwargs={'dflist':[df_1, df_2]  ## List of dataframes 
                ,'varlist':['var1', 'var2'] ## Variables to be plotted
                ,'varlist_2':['var1', 'var2'] ## Variables on second exis, bar plot
                ,'title':'var1 & var2 for df_1 and df_2'  ## Plot title
                ,'labellist':['df_1_var_1, df_1_var_2, df_2_var_1, df_2_var_2'] ## Labels in legend
                , labellist_2: []
                , 'x_var':''} ## x_var set to order_date by default
        fig = get_plot(**kwargs)
    1. Creates dataframe x variable combination
    2. For each data trace, creates scatter-line plot
    3. Plots and returns fig object for interactive plot edit
    
    To edit on chart editor: 
    
    pio.write_json(fig, 'chart_1.json') Right click on json file, open with plotly chart editor. 
    """

    ## Create dataframe x var combinations (plot all variables for each dataframe)
    dfvarlist=itertools.product(*[dflist,varlist])
    dfvarlist_2=itertools.product(*[dflist,varlist_2])
    
    ## Get list of labels
    if labellist==None:
        labellist=varlist+varlist2
        
    ## For every dataframe x variable, create a data trace for plotting
    data=[]
    
    # primary axis
    for dfvar,name in zip(dfvarlist, labellist):
        dfplt,var=dfvar[0], dfvar[1]
        if x_var=='order_date':
            try:
                dfplt.reset_index(inplace=True)
            except:
                pass
            dfplt.set_index(pd.DatetimeIndex(dfplt['order_date']), inplace=True)
        data = data + [go.Scatter(x=dfplt.index, y=dfplt[var], mode=mode, name=name)]

    ## plot secondary axis
    for dfvar,name in zip(dfvarlist_2, labellist_2):
        dfplt,var=dfvar[0], dfvar[1]
        if x_var=='order_date':
            try:
                dfplt.reset_index(inplace=True)
            except:
                pass
            dfplt.set_index(pd.DatetimeIndex(dfplt['order_date']), inplace=True)
        data = data+[go.Bar(x=dfplt.index, y=dfplt[var], 
                            marker=dict(color='grey'), name=name, yaxis="y2", opacity=0.4)]


    py.offline.init_notebook_mode(connected=True)
    layout = dict(title = title,
                  xaxis = dict(title = x_var), 
                  yaxis = dict(title = title),
                  yaxis2 = dict(title=", ".join(varlist_2), side="right", overlaying="y", automargin=True),
                  autosize=False,
                  width=(1000 if len(config)==0 else 600),
                  height=(450 if len(config)==0 else 300),
                  showlegend=True,
                  legend=dict(x=1.1)
                 )

    fig = dict(data=data, layout=layout)
    py.offline.iplot(fig, config=config)
    return fig


def histogram_plot(dflist, varlist, labellist=None, 
              title=None, config={}, option={}):
    """ 
    How to use:
        kwargs={'dflist':[df_1, df_2]  ## List of dataframes 
                ,'varlist':['var1', 'var2'] ## Variables to be plotted
                ,'title':'var1 & var2 for df_1 and df_2'  ## Plot title
                ,'labellist':['df_1_var_1, df_1_var_2, df_2_var_1, df_2_var_2']} ## Labels to be shown in legend
        fig = histogram_plot(**kwargs)
        'option': {'histnorm':'probability','cumulative':dict(enabled=False), 'xbins':{'size':0.005}}
    1. Creates dataframe x variable combination
    2. For each data trace, creates scatter-line plot
    3. Plots and returns fig object for interactive plot edit
    
    To edit on chart editor: 
    import plotly.io as pio
    pio.write_json(fig, 'chart_1.json') Right click on json file, open with plotly chart editor. 
    """

    ## Create dataframe x var combinations (plot all variables for each dataframe)
    dfvarlist=itertools.product(*[dflist,varlist])  
    data=[]
    
    for dfvar,name in zip(dfvarlist, labellist):
        dfplt,var=dfvar[0], dfvar[1]
        data = data + [go.Histogram(x=dfplt[var], name=name, opacity=0.75,  **option)]       #nbinsx=200,

    py.offline.init_notebook_mode(connected=True)
    layout = dict(title = title,
                  xaxis = dict(title = title), #, range=['2017-09-01','2017-02-01']
                  autosize=False,
                  width=600, # if len(config)==0 else 600),
                  height=300, # if len(config)==0 else 300),
                  showlegend=True,
                  barmode='overlay'
                 )
    fig = dict(data=data, layout=layout)
    py.offline.iplot(fig, config=config)
    return 


def get_box_plot(df, y_var):
    fig, ax = plt.subplots(figsize = (20, 10))
    sns.boxplot(x='category',y='first_views', data=df, hue='tier', ax=ax, showfliers = False)
#     ax.set_ylabel(('first_views'))
    plt.show()
    
def get_hist_plot(df, y_var):
    fig, ax = plt.subplots(figsize = (10, 5))
    plt.hist(df[[y_var]], bins=100)
    ax.set_xlabel(y_var)
    plt.show()


def get_shap(df, model_path, model, resolution, value=100):
    """
    ev, shap, x_ohe = get_shap(df_pred, model_path, model, 'return')
    shap.force_plot(ev, shap, x_ohe)
    """
    with open(model_path, 'rb') as f:
        ms = pickle.load(f)['{}_{}'.format(resolution, model)]
    pre_days = [7, 10, 12, 14, 16, 18, 21]
    features = {'prob': ['{}_frac_pre_{}'.format(resolution,j) for j in pre_days],
                'prob_pop':['mean_days_to_deliver_pre_14', 'mean_days_to_deliver_pre_21'],
                'prob_sparse': ['mean_days_to_deliver_pre_14', 'mean_days_to_deliver_pre_21'],
                'prob_categoricals_pop':['order_date_moy','order_date_woy', 'sku_grp'],
                'prob_categoricals_sparse': ['order_date_moy', 'order_date_woy', 'sku_grp']}
    grpdic = {'pop':['sku', 'order_date'],'sparse':['sku_grp', 'order_date']}

    x = features['prob'] +features['prob_{}'.format(model)]+ features['prob_categoricals_{}'.format(model)]
    df_x = df[x]

    shap.initjs()
    model_prob = ms['bst']
    x_ohe = ms['ohe'].transform(df_x)
    explainer = shap.TreeExplainer(model_prob)
    shap_values = explainer.shap_values(x_ohe)
    
    shap.summary_plot(shap_values, x_ohe)
    shap.summary_plot(shap_values, x_ohe, plot_type="bar")
    return explainer.expected_value, shap_values[value,:], x_ohe.iloc[value,:]

