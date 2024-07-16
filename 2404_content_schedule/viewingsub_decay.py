# Databricks notebook source
import re
import io
import warnings
import os
from scipy.optimize import curve_fit

repo = 'jupyter-notebooks'
root_path = os.getcwd()[:re.search(repo, os.getcwd()).start()] + repo + '/'

utils_imports = root_path + 'utils/imports.py'
utils_import_data = root_path + 'utils/import_data.py'
utils_sf =  root_path + 'utils/utils_sf.py'
utils_models =  root_path + 'utils/utils_models.py'

for file in (utils_imports, utils_sf, utils_import_data, utils_models):
    with open(file, "r") as file:
        python_code = file.read()
    exec(python_code)

pd.options.display.float_format = '{:,.3f}'.format


# COMMAND ----------


query_metric = """
select 
    imdb_series_id as imdb_id,
    title_series as title_name,
    geo_value as region,
    days_on_max,
    percent_cumulative_viewing_subs as vs_pct
from bolt_cus_dev.gold.delphi_title_metrics_platform p
left join bolt_cus_dev.silver.forecasting_feature_space_test f
    on f.imdb_id=p.imdb_series_id 
    and f.title_name = p.title_series
    and f.season_number= p.season_number
    and f.region = p.geo_value 
where p.geo_level='REGION'
and p.geo_value='NORTH AMERICA'
and forecast_category='wb pay1'
and medal='Platinum'
"""

##  metrics data cleaning 
df_m = spark.sql(query_metric)
df_m = df_m.toPandas()
df_m.columns=df_m.columns.str.lower()


# COMMAND ----------

df_m = df_m[df_m.days_on_max<=28]

def convert_to_percentage(group):
    denominator = group.loc[group['days_on_max'] == 28, 'vs_pct'].values[0]
    group['vs_pct_decay'] = (group['vs_pct'] / denominator) * 100
    return group

# Apply the function to each group
df_m_pct = df_m.groupby('title_name').apply(convert_to_percentage)

df_m_pct['vs_pct_decay'] = df_m_pct['vs_pct_decay']/100
df_decay =  df_m_pct.groupby(by=['days_on_max']).mean().reset_index()
df_decay = df_decay[['days_on_max','vs_pct_decay']]

# COMMAND ----------

df_decay

# COMMAND ----------

# from pyspark.sql import SparkSession
# spark = SparkSession.builder.appName("DataFrame to Table").getOrCreate()
sdf = spark.createDataFrame(df_decay)
full_table_name = f"bolt_cus_dev.bronze.cso_pct_vs_decay_pay1_platinum"
sdf.write.saveAsTable(full_table_name)

# COMMAND ----------


