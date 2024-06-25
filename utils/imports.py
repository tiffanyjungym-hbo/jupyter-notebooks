# Standard packages
import pandas as pd
import numpy as np
import csv
import plotly.express as px

# Notebook viewing
pd.set_option('display.max_rows', 75)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:95% !important; }</style>"))

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (20,7)

# Dates and times
from pandas.tseries.offsets import MonthEnd, Week
from datetime import timedelta
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
import pytz

# Statistics
import statsmodels.formula.api as sm  
import statsmodels
from statsmodels.tsa.arima.model import ARIMA as ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Convert to datetime
def d(date_string):
    return pd.to_datetime(date_string)

## update last Sunday to capture the actual date fall on Sunday, and projection is run on a Monday. Updated on Mar. 20, 2023, TS
def last_sunday(date_column):
    ## if last actual is Sunday, then reporting date should be Sunday the same date, but not previous sunday
    if d(date_column).weekday()==6:
        return d(date_column)
    return d(date_column) + Week(weekday=0) - timedelta(days=8)

def week_end(date_column):
    return d(date_column) + Week(weekday=0) - timedelta(days=1)

def month_end(date_column):
    return d(date_column) + MonthEnd(0)

def calculate_smape(actual, predicted) :
  
    # Convert actual and predicted to numpy
    # array data type if not already
    if not all([isinstance(actual, np.ndarray), 
                isinstance(predicted, np.ndarray)]):
        actual, predicted = np.array(actual),
        np.array(predicted)
  
    return round(
        np.mean(
            np.abs(predicted - actual) / 
            ((np.abs(predicted) + np.abs(actual))/2)
        )*100, 2)