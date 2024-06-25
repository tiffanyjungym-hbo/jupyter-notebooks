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

## titles viewed/month : num:  # title view @ request_date_month,   denom:  # active sub cumulative request_date_month 
## %VA/month:  

# COMMAND ----------

query = """
select 
date_trunc('month', request_date_pst) as date_month,
sub_tenure_months,
count(distinct wbd_max_or_hbomax_user_id) as user_count,
sum(CONTENT_MINUTES_WATCHED)/60 as hours_viewed 
from  bolt_dai_ce_prod.gold.combined_video_stream hb 
where region='NORTH AMERICA'
and  request_date_pst between '2022-01-01' and '2024-03-01'
    and PROGRAM_ID_OR_VIEWABLE_ID IS NOT NULL 
    and CONTENT_MINUTES_WATCHED >= 2
    and video_type = 'main'
    and category='retail'
    and sub_tenure_months is not null
group by all
"""

df = spark.sql(query)
df = df.toPandas()
df.to_csv('combined_video_stream_monthly.csv')

# COMMAND ----------

df = pd.read_csv('title_hours_viewed_2024_06_18.csv')
# df['title_per_activesub'] = df['n_title_viewed']/df['n_viewing_sub']
df.date_month= pd.to_datetime(df.date_month)
df = df[(df.date_month>='2022-01-01') & (df.date_month<'2024-03-01')]
df=df[df.date_month!='2023-05-01']
# df['average_title_view_count'] = df['mean(n_title_viewed)']
df.sort_values(by=['date_month'])



# COMMAND ----------

fig = px.scatter(df, x='date_month', y='average_title_view_count', hover_data=['date_month',],template='plotly_white') #color='region',

fig.update_layout(
height=400,
width=600,
yaxis_range =[4,7],
xaxis=dict(
    title=dict(font=dict(size=12)),   
    linecolor='black',
    ticks='outside'
),
yaxis=dict(
    title=dict(font=dict(size=12)),   
    linecolor='black',
    ticks='outside'
),
font=dict(size=12),  # Font size for the rest of the text elements
title_font=dict(size=12),

)

fig.show()

# COMMAND ----------

fig = px.scatter(df, x='date_month', y='average_hours_viewed', hover_data=['date_month',],template='plotly_white') #color='region',

fig.update_layout(
height=400,
width=600,
# yaxis_range =[4,7],
xaxis=dict(
    title=dict(font=dict(size=12)),   
    linecolor='black',
    ticks='outside'
),
yaxis=dict(
    title=dict(font=dict(size=12)),   
    linecolor='black',
    ticks='outside'
),
font=dict(size=12),  # Font size for the rest of the text elements
title_font=dict(size=12),

)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tenure
# MAGIC HOTD 0821-1023
# MAGIC TLOU  0115-0312
# MAGIC

# COMMAND ----------

df.head()

# COMMAND ----------

df = pd.read_csv('title_hours_viewed_tenure_2024_06_20.csv')
df.date_month= pd.to_datetime(df.date_month)
df = df[(df.date_month>='2022-01-01') & (df.date_month<'2024-03-01')]
df=df[df.date_month!='2023-05-01']
df.sort_values(by=['date_month'])

min_tenure_mth = 4
max_tenure_mth = 12
df['tenure_grp'] = df.sub_month
df.loc[(df.tenure_grp>=min_tenure_mth) & (df.tenure_grp<=max_tenure_mth), 'tenure_grp']=f'>={min_tenure_mth} <={max_tenure_mth}'


total_sum = df.groupby(['date_month','tenure_grp'])['viewing_sub_count'].sum()  
group_sums = df.groupby(['date_month','tenure_grp','sub_month'])['viewing_sub_count'].sum()
weights = group_sums / total_sum

df['weight'] = df.apply(lambda x: weights[(x['date_month'], x['tenure_grp'],  x['sub_month'])], axis=1)
df['weighted_average_title_view_count'] = df['average_title_view_count'] * df['weight']

df_plt = df[df.sub_month<=max_tenure_mth].copy()
df_plt = df_plt.groupby(by=['date_month','tenure_grp']).sum().reset_index()
df_plt['average_hours_viewed'] = df_plt['total_hours_viewed']/df_plt['viewing_sub_count']

# COMMAND ----------


fig = px.scatter(df_plt, x='date_month', y='average_hours_viewed', hover_data=['date_month',],template='plotly_white', color='tenure_grp') #color='region',

fig.update_layout(
height=400,
width=600,
# yaxis_range =[4,7],
xaxis=dict(
    title=dict(font=dict(size=12)),   
    linecolor='black',
    ticks='outside'
),
yaxis=dict(
    title=dict(font=dict(size=12)),   
    linecolor='black',
    ticks='outside'
),
font=dict(size=12),  # Font size for the rest of the text elements
title_font=dict(size=12),

)

fig.show()

# COMMAND ----------


fig = px.scatter(df_plt, x='date_month', y='weighted_average_title_view_count', hover_data=['date_month',],template='plotly_white', color='tenure_grp') #color='region',

fig.update_layout(
height=400,
width=600,
# yaxis_range =[4,7],
xaxis=dict(
    title=dict(font=dict(size=12)),   
    linecolor='black',
    ticks='outside'
),
yaxis=dict(
    title=dict(font=dict(size=12)),   
    linecolor='black',
    ticks='outside'
),
font=dict(size=12),  # Font size for the rest of the text elements
title_font=dict(size=12),

)

fig.show()

# COMMAND ----------

query_metric = """
select 
    p.imdb_series_id as imdb_id,
    p.title_series as title_name,
    p.season_number,
    p.offering_start_date as premiere_date,
    d.last_release_date,
    p.geo_value as region,
    p.days_on_max,
    p.percent_cumulative_viewing_subs as vs_pct
from bolt_cus_dev.gold.delphi_title_metrics_platform p
left join bolt_cus_dev.gold.delphi_title_metadata d
    on p.imdb_series_id=d.imdb_series_id 
    and p.title_series = d.title_series
    and p.season_number= d.season_number
    and p.geo_value = d.geo_value 
where p.geo_level='REGION'
and d.geo_level='REGION'
and p.geo_value='NORTH AMERICA'
and p.days_on_max in (28)
and p.offering_start_date>='2022-01-01'
"""
##  metrics data cleaning 
df_m = spark.sql(query_metric)
df_m = df_m.toPandas()
df_m.columns=df_m.columns.str.lower()

df_m[df_m.vs_pct>20].sort_values(by=['premiere_date'])

# COMMAND ----------

## mean_title_viewed query 
'''
with user_title_count as (
SELECT   date_trunc('month',h.request_date) as date_month
    , h.user_id
    , count(distinct h.ckg_series_id) as n_title_viewed
FROM bolt_cus_dev.bronze.user_title_hours_watched_subs_wbd h 
WHERE 1=1
AND h.request_date::DATE >= '2023-05-23'
group by 1, 2),

user_title_count_legacy as (
SELECT   date_trunc('month',h.request_date) as date_month
    , u_map.user_id
    , count(distinct h.ckg_series_id) as n_title_viewed
FROM bolt_cus_dev.bronze.user_title_hours_watched_subs_legacy h 
LEFT JOIN bolt_dai_subs_prod.gold.max_legacy_user_mapping_global u_map
    ON h.hurley_user_id::STRING = u_map.hurley_user_id::STRING
WHERE 1=1
AND h.request_date::DATE < '2023-05-23'
group by 1, 2)


SELECT
    date_month
    ,count(distinct user_id) as viewing_sub_count
    ,PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY n_title_viewed) AS median_unique_title_count
    ,mean(n_title_viewed) as average_title_view_count
FROM
    user_title_count
group by date_month

UNION

SELECT
    date_month
    ,count(distinct user_id) as viewing_sub_count
    ,PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY n_title_viewed) AS median_unique_title_count
    ,mean(n_title_viewed) as average_title_view_count
FROM
    user_title_count_legacy
group by date_month
'''


# COMMAND ----------

## mean_title_viewed query 
'''
with user_title_count as (
SELECT   date_trunc('month',h.request_date) as date_month
    , h.user_id
    , count(distinct h.ckg_series_id) as n_title_viewed
    , sum(hours_viewed) as hours_viewed
FROM bolt_cus_dev.bronze.user_title_hours_watched_subs_wbd h 
WHERE 1=1
AND h.request_date::DATE >= '2023-05-23'
group by 1, 2),

user_title_count_legacy as (
SELECT   date_trunc('month',h.request_date) as date_month
    , u_map.user_id
    , count(distinct h.ckg_series_id) as n_title_viewed
    , sum(hours_viewed) as hours_viewed
FROM bolt_cus_dev.bronze.user_title_hours_watched_subs_legacy h 
LEFT JOIN bolt_dai_subs_prod.gold.max_legacy_user_mapping_global u_map
    ON h.hurley_user_id::STRING = u_map.hurley_user_id::STRING
WHERE 1=1
AND h.request_date::DATE < '2023-05-23'
group by 1, 2)


SELECT
    date_month
    ,count(distinct user_id) as viewing_sub_count
    ,PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY n_title_viewed) AS median_unique_title_count
    ,mean(n_title_viewed) as average_title_view_count
    , mean(hours_viewed) as average_hours_viewed
FROM
    user_title_count
group by date_month

UNION

SELECT
    date_month
    ,count(distinct user_id) as viewing_sub_count
    ,PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY n_title_viewed) AS median_unique_title_count
    ,mean(n_title_viewed) as average_title_view_count
    ,mean(hours_viewed) as average_hours_viewed
FROM
    user_title_count_legacy
group by date_month


'''


# COMMAND ----------

## Tenure level 

query = """
with user_title_count as (
SELECT   date_trunc('month',h.request_date) as date_month
    , sub_month
    , h.user_id
    , count(distinct h.ckg_series_id) as n_title_viewed
    , sum(hours_viewed) as hours_viewed
FROM bolt_cus_dev.bronze.cip_user_title_hours_watched_season_wbd_avod_svod h 
WHERE 1=1
AND h.request_date::DATE >= '2023-05-23'
group by all),

user_title_count_legacy as (
SELECT   date_trunc('month',h.request_date) as date_month
    , sub_month
    , u_map.user_id
    , count(distinct h.ckg_series_id) as n_title_viewed
    , sum(hours_viewed) as hours_viewed
FROM bolt_cus_dev.bronze.cip_user_title_hours_watched_season_legacy_avod_svod h 
LEFT JOIN bolt_dai_subs_prod.gold.max_legacy_user_mapping_global u_map
    ON h.hurley_user_id::STRING = u_map.hurley_user_id::STRING
WHERE 1=1
AND h.request_date::DATE < '2023-05-23'
group by all)


SELECT
    date_month
    , sub_month
    ,count(distinct user_id) as viewing_sub_count
    , sum(hours_viewed) as total_hours_viewed
    ,PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY n_title_viewed) AS median_unique_title_count
    ,mean(n_title_viewed) as average_title_view_count
    , mean(hours_viewed) as average_hours_viewed
FROM
    user_title_count
group by all

UNION

SELECT
    date_month
    , sub_month
    ,count(distinct user_id) as viewing_sub_count
    ,sum(hours_viewed) as total_hours_viewed
    ,PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY n_title_viewed) AS median_unique_title_count
    ,mean(n_title_viewed) as average_title_view_count
    ,mean(hours_viewed) as average_hours_viewed
FROM
    user_title_count_legacy
group by all

"""

