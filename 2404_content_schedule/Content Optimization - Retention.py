# Databricks notebook source
import re
import io
import warnings
import os

repo = 'jupyter-notebooks'
root_path = os.getcwd()[:re.search(repo, os.getcwd()).start()] + repo + '/'

utils_imports = root_path + 'utils/imports.py'
utils_import_data = root_path + 'utils/import_data.py'
utils_sf =  root_path + 'utils/utils_sf.py'
# utils_gs =  root_path + 'utils/utils_gs.py'
# utils_od =  root_p ath + 'utils/utils_od.py'
utils_models =  root_path + 'utils/utils_models.py'

for file in (utils_imports, utils_sf, utils_import_data, utils_models):
    with open(file, "r") as file:
        python_code = file.read()
    exec(python_code)

pd.options.display.float_format = '{:,.3f}'.format

# path=!pwd
# sys.path.append(os.path.join(path[0], '../utils'))

# from imports import *
# # from import_data import *
# from utils_sf import *
# from utils_gs import *
# from utils_od import *


# COMMAND ----------

# MAGIC %md
# MAGIC # Churn

# COMMAND ----------

# M1 Churn
df_churn = pd.read_sql("""
select a.region, a.title_name, a.imdb_id, a.forecast_category, a.season_number, a.premiere_date, a.finale_date,
c.fv_post_7, c.fv_post_30,
b.lob,
sum(b.num_subs) as fv, 
sum(b.eligible_churn * b.churn_rate)/sum(b.eligible_churn) as churn_rate
from forecasting_fv_summary a
left join max_ltv_content_past b using (title_name, region, season_number)
left join forecasting_fv_pre_post c using (title_name, region, season_number)
where 
period_number = 1
and provider = 'Direct'
and sub_region = 'US'
and region = 'NORTH AMERICA'
and ad_strategy = 'ad_free'
and payment_period = 'PERIOD_MONTH'
and signup_offer = 'none'
and b.paid_start_date between a.premiere_date and a.premiere_date + 30
and imdB_id in (select imdb_id from forecasting_premieres)
and a.premiere_date between '2022-01-01' and '2023-12-31'
group by 1,2,3,4,5,6,7,8,9,10
                       """, con)

# COMMAND ----------

df_plot = df_churn.query("FV>=1000 & FORECAST_CATEGORY=='series' & LOB=='HBO Originals'")
df_plot['lnFV'] = np.log(df_plot.FV_POST_30)
sns.scatterplot(x='lnFV',y='CHURN_RATE',data=df_plot)

# COMMAND ----------

import plotly.express as px
fig = px.scatter(df_plot, x = 'lnFV', y='CHURN_RATE', hover_data=['TITLE_NAME','SEASON_NUMBER','PREMIERE_DATE'])
fig.show()

# COMMAND ----------

df_plot = df_churn.query("FV>=1000 & FORECAST_CATEGORY=='wb pay1'")
df_plot['lnFV'] = np.log(df_plot.FV_POST_7)

sns.scatterplot(x='lnFV',y='CHURN_RATE',data=df_plot)

# COMMAND ----------

# MAGIC %md
# MAGIC # FV

# COMMAND ----------

df_raw = pd.read_sql("""
    select * from forecasting_feature_space
    where (future_title = 1) or (medal in ('Platinum','Gold','Silver','A','B'))
    """, con)
df_raw.head()

# COMMAND ----------

df_raw.columns

# COMMAND ----------

df_content_strength = pd.read_sql("select * from forecasting_fv_pre_post", con)

# COMMAND ----------

df_content_strength.head()

# COMMAND ----------

df_reg = df_raw.merge(df_content_strength[['REGION','IMDB_ID','SEASON_NUMBER','FV_PRE_30']], on = ['REGION','IMDB_ID','SEASON_NUMBER']).query("REGION=='NORTH AMERICA' & SEASON_NUMBER>1.1")

# COMMAND ----------

df_reg['lnFV7'] = np.log(df_reg.RFV7)
df_reg['lnFV90'] = np.log(df_reg.RFV90)
# df_reg['lnCatchupFV'] = np.log(df_reg.CATCHUP_RFV)
# df_reg['lnStrength'] = np.log(df_reg.FV_PRE_PREMIERE)
# df_reg['lnSpend'] = np.log(df_reg.SPEND_US+1)
df_reg['content_strength'] = 'Low'
#df_reg.loc[df_reg.lnStrength<12.2,'content_strength'] = 'Low'
# df_reg.loc[df_reg.lnStrength>12.4,'content_strength'] = 'High'

# COMMAND ----------

import pandas as pd
import seaborn
import matplotlib.pyplot as plt

# create the figure and axes
fig, ax = plt.subplots(figsize=(6, 6))

# add the plots for each dataframe
sns.regplot(x='lnCatchupFV', y='lnFV7', data=df_reg.query("content_strength=='High'"), fit_reg=True, ci=None, ax=ax, label='df1')
sns.regplot(x='lnCatchupFV', y='lnFV7', data=df_reg.query("content_strength=='Low'"), fit_reg=True, ci=None, ax=ax, label='df2')
ax.set(ylabel='y', xlabel='x')
ax.legend()
plt.show()

# COMMAND ----------

df_reg.lnStrength.describe()

# COMMAND ----------

sns.scatterplot(x='lnCatchupFV',y='lnFV7',data=df_reg,hue='content_strength')

# COMMAND ----------

df_reg[['TITLE_NAME','SEASON_NUMBER','PREMIERE_DATE','RFV7','CATCHUP_RFV','lnStrength','content_strength']].sort_values('CATCHUP_RFV')

# COMMAND ----------


