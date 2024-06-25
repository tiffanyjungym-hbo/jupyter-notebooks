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
utils_gs =  root_path + 'utils/utils_gs.py'
utils_od =  root_path + 'utils/utils_od.py'
utils_models =  root_path + 'utils/utils_models.py'

for file in (utils_imports, utils_sf, utils_models):
    print(file)
    with open(file, "r") as file:
        python_code = file.read()
    exec(python_code)

# pd.options.display.float_format = '{:,.2f}'.format


# path=!pwd
# sys.path.append(os.path.join(path[0], '../utils'))

# from imports import *
# # from import_data import *
# from utils_sf import *
# from utils_gs import *
# from utils_od import *

# COMMAND ----------

# MAGIC %md
# MAGIC Features to include
# MAGIC - title meta data
# MAGIC - days from theatrical (pay1)
# MAGIC - days from previous platinum
# MAGIC - total first views of previous platinum 
# MAGIC
# MAGIC
# MAGIC Model definition 
# MAGIC - FV = a(title_metadata) + b(days_from_theatrical) + c(days_from_previous_platinum/gold) + d(previous_platinum)/FV_predicted_0
# MAGIC
# MAGIC Optimization
# MAGIC - Maximize sum of FV across title t=0...n,
# MAGIC - constraint:  Sum of days_from_previous + days_running <= target time period 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## All

# COMMAND ----------

# S1
forecast_category = ['series', 'wb pay1']

df = pd.read_sql("""
    select *
    from forecasting_feature_space_test a
     where premiere_date>='2022-01-01'
     and region='NORTH AMERICA'
    """, 
    con)

df.head()

# COMMAND ----------

model_list_s1 = {
    '2wbp_search':'ln_SEARCH_2WBP + ln_WIKI_2WBP + ln_NUM_EPISODES',
    '2wbp_search_fv7':'ln_SEARCH_2WBP + ln_WIKI_2WBP + ln_NUM_EPISODES + ln_FV_PRE_7',
    '2wbp_search_fv30':'ln_SEARCH_2WBP + ln_WIKI_2WBP + ln_NUM_EPISODES + ln_FV_PRE_30',
    '2wbp_search_fv7_series':'ln_SEARCH_2WBP + ln_WIKI_2WBP + ln_NUM_EPISODES + ln_FV_PRE_7_SERIES',
    '2wbp_search_fv30_series':'ln_SEARCH_2WBP + ln_WIKI_2WBP + ln_NUM_EPISODES + ln_FV_PRE_30_SERIES',
    '2wbp_search_fv7_pay1':'ln_SEARCH_2WBP + ln_WIKI_2WBP + ln_NUM_EPISODES + ln_FV_PRE_7_PAY1',
    '2wbp_search_fv30_pay1':'ln_SEARCH_2WBP + ln_WIKI_2WBP + ln_NUM_EPISODES + ln_FV_PRE_30_PAY1',
}

# Transformations
features_log = ['RFV30','MARKETING_SPEND','VIEWS_OWNED_TEASER','VIEWS_OWNED_TRAILER','NUM_EPISODES','SEARCH_TEASER','WIKI_TEASER','SEARCH_TRAILER','WIKI_TRAILER','SEARCH_2WBP','WIKI_2WBP','AWARENESS_2WBP',
                'FV_PRE_7','FV_PRE_30','FV_PRE_7_PAY1','FV_PRE_30_PAY1','FV_PRE_7_SERIES','FV_PRE_30_SERIES']

# Outliers
titles_exclude = ['The Idol','And Just Like That...','Pause with Sam Jay','The Hype']

medal = ['Gold','Platinum','Silver']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Series

# COMMAND ----------

# S1
forecast_category = 'series'

df = pd.read_sql("""
    select *
    from forecasting_feature_space_test a
    where a.forecast_category = 'series'
     and season_number = 1
     and premiere_date>='2021-01-01'
    """, 
    con)


df.head()

# COMMAND ----------

model_list_s1 = {
    '2wbp_search':'ln_SEARCH_2WBP + ln_WIKI_2WBP + ln_NUM_EPISODES',
    '2wbp_search_fv7':'ln_SEARCH_2WBP + ln_WIKI_2WBP + ln_NUM_EPISODES + ln_FV_PRE_7',
    '2wbp_search_fv30':'ln_SEARCH_2WBP + ln_WIKI_2WBP + ln_NUM_EPISODES + ln_FV_PRE_30',
    '2wbp_search_fv7_series':'ln_SEARCH_2WBP + ln_WIKI_2WBP + ln_NUM_EPISODES + ln_FV_PRE_7_SERIES',
    '2wbp_search_fv30_series':'ln_SEARCH_2WBP + ln_WIKI_2WBP + ln_NUM_EPISODES + ln_FV_PRE_30_SERIES',
    '2wbp_search_fv7_pay1':'ln_SEARCH_2WBP + ln_WIKI_2WBP + ln_NUM_EPISODES + ln_FV_PRE_7_PAY1',
    '2wbp_search_fv30_pay1':'ln_SEARCH_2WBP + ln_WIKI_2WBP + ln_NUM_EPISODES + ln_FV_PRE_30_PAY1',
}

# Transformations
features_log = ['RFV90','MARKETING_SPEND','VIEWS_OWNED_TEASER','VIEWS_OWNED_TRAILER','NUM_EPISODES','SEARCH_TEASER','WIKI_TEASER','SEARCH_TRAILER','WIKI_TRAILER','SEARCH_2WBP','WIKI_2WBP','AWARENESS_2WBP',
                'FV_PRE_7','FV_PRE_30','FV_PRE_7_PAY1','FV_PRE_30_PAY1','FV_PRE_7_SERIES','FV_PRE_30_SERIES']

# Outliers
titles_exclude = ['The Idol','And Just Like That...','Pause with Sam Jay','The Hype']

medal = ['Gold','Platinum','Silver']

# COMMAND ----------

models_s1 = fv_model_set(df, 'series', model_list_s1, features_log, medal, titles_exclude = titles_exclude, target = 'RFV90')
models_s1.train_all()

# COMMAND ----------

df.columns

# COMMAND ----------

df_train = models_s1.get_df('NORTH AMERICA')
df_train.columns=[col.lower() for col in df_train.columns]
df_train.shape

# COMMAND ----------

models_s1.get_summary()

# COMMAND ----------

models_s1.get_coeff_summary('NORTH AMERICA')

# COMMAND ----------

models_s1.get_tstat('NORTH AMERICA')

# COMMAND ----------

models_s1.get_coef('NORTH AMERICA')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Returning

# COMMAND ----------

# Returning
forecast_category = 'series'

df = pd.read_sql("""
    select *
    from forecasting_feature_space_test a
     where a.forecast_category = 'series'
     and season_number > 1
     and season_number not in (1.1, 2.1, 3.1, 4.1)
     and premiere_date>='2021-01-01'
    """, 
    con)

# COMMAND ----------

model_list_returning = {
    '2wbp_cfv':'ln_CFV_2WBP + ln_NUM_EPISODES',
    '2wbp_cfv_fv7':'ln_CFV_2WBP + ln_NUM_EPISODES + ln_FV_PRE_7',
    '2wbp_cfv_fv30':'ln_CFV_2WBP + ln_NUM_EPISODES + ln_FV_PRE_30',
    '2wbp_cfv_fv7_series':'ln_CFV_2WBP + ln_NUM_EPISODES + ln_FV_PRE_7_SERIES',
    '2wbp_cfv_fv30_series':'ln_CFV_2WBP + ln_NUM_EPISODES + ln_FV_PRE_30_SERIES',
    '2wbp_cfv_fv7_pay1':'ln_CFV_2WBP + ln_NUM_EPISODES + ln_FV_PRE_7_PAY1',
    '2wbp_cfv_fv30_pay1':'ln_CFV_2WBP + ln_NUM_EPISODES + ln_FV_PRE_30_PAY1',
}

# Transformations
features_log = ['RFV7','MARKETING_SPEND','VIEWS_OWNED_TEASER','VIEWS_OWNED_TRAILER','NUM_EPISODES','SEARCH_TEASER','WIKI_TEASER','SEARCH_TRAILER','WIKI_TRAILER','SEARCH_2WBP','WIKI_2WBP','AWARENESS_2WBP','CFV_2WBP',
                'FV_PRE_7','FV_PRE_30','FV_PRE_7_PAY1','FV_PRE_30_PAY1','FV_PRE_7_SERIES','FV_PRE_30_SERIES']

# Outliers
titles_exclude = ['The Idol','And Just Like That...','Pause with Sam Jay','The Hype','South Park','Rick and Morty']

medal = ['Gold','Platinum','Silver']

# COMMAND ----------

models_returning = fv_model_set(df, 'series', model_list_returning, features_log, medal, titles_exclude = titles_exclude, target = 'RFV7')
models_returning.train_all()

# COMMAND ----------

df_train = models_returning.get_df('NORTH AMERICA')
df_train.columns=[col.lower() for col in df_train.columns]
df_train.shape

# COMMAND ----------

models_returning.get_summary()

# COMMAND ----------

models_returning.get_tstat('NORTH AMERICA')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pay1

# COMMAND ----------

# WB Pay 1
forecast_category = 'wb pay1'

df = pd.read_sql("""
    select *
    from forecasting_feature_space_test a
     where a.forecast_category = 'wb pay1'
     and premiere_date>='2022-01-01'
    """, 
    con)

display(df.head())
print(df.shape)

# COMMAND ----------

model_list_wbpay1 = {
    'bo_opening':'ln_BO_OPENING_IMPUTED + ln_GAP + ln_PRODUCTION_COST',
    'bo_opening_fv7':'ln_BO_OPENING_IMPUTED + ln_GAP + ln_PRODUCTION_COST + ln_FV_PRE_7',
    'bo_opening_fv30':'ln_BO_OPENING_IMPUTED + ln_GAP + ln_PRODUCTION_COST + ln_FV_PRE_30',
    'bo_opening_fv7_series':'ln_BO_OPENING_IMPUTED + ln_GAP + ln_PRODUCTION_COST + ln_FV_PRE_7_SERIES',
    'bo_opening_fv30_series':'ln_BO_OPENING_IMPUTED + ln_GAP + ln_PRODUCTION_COST + ln_FV_PRE_30_SERIES',
    'bo_opening_fv7_pay1':'ln_BO_OPENING_IMPUTED + ln_GAP + ln_PRODUCTION_COST + ln_FV_PRE_7_PAY1',
    'bo_opening_fv30_pay1':'ln_BO_OPENING_IMPUTED + ln_GAP + ln_PRODUCTION_COST + ln_FV_PRE_30_PAY1',
}

# Transformations
features_log = ['RFV7','BO_OPENING_IMPUTED','GAP','PRODUCTION_COST',
                'FV_PRE_7','FV_PRE_30','FV_PRE_7_PAY1','FV_PRE_30_PAY1','FV_PRE_7_SERIES','FV_PRE_30_SERIES']

# Outliers
titles_exclude = ['The Idol','And Just Like That...','Pause with Sam Jay','The Hype','South Park','Rick and Morty']

medal = ['Gold','Platinum','Silver','Other']

# COMMAND ----------

models_wbpay1 = fv_model_set(df, 'wb pay1', model_list_wbpay1, features_log, medal, titles_exclude = titles_exclude, target = 'RFV7')
models_wbpay1.train_all()

# COMMAND ----------

df_train = models_wbpay1.get_df('NORTH AMERICA')
df_train.columns=[col.lower() for col in df_train.columns]
df_train.shape

# COMMAND ----------

models_wbpay1.get_tstat('NORTH AMERICA')

# COMMAND ----------


