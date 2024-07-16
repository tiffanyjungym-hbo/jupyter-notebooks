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

query=''' select * from bolt_cus_dev.bronze.cso_viewing_sub_count_tenure_us where contains(title_series, 'House of the Dragon')'''
df = spark.sql(query)

display(df)

# COMMAND ----------

query='''
CREATE OR REPLACE TABLE bolt_cus_dev.bronze.cso_viewing_sub_count_tenure_us
USING DELTA
PARTITIONED BY (days_since_season_premiere_date_pst)

select 
	base.title_series
	,base.season_number
	,base.ckg_match_id
    ,base.region
    ,to_date(base.first_release_date, 'yyyy-MM-dd') as first_release_date
    ,to_date(base.last_release_date, 'yyyy-MM-dd') as last_release_date
    ,case 
        when sub_tenure_months > 12 then '13+'
        when sub_tenure_months between 7 and 12 then '7-12'
        when sub_tenure_months between 4 and 6 then '4-6'
        else sub_tenure_months 
        end as tenure_grp
    ,metric.request_date_pst
    ,metric.days_since_season_premiere_date_pst 
    ,count(distinct WBD_MAX_OR_HBOMAX_USER_ID) as sub_count
from bolt_cus_dev.gold.delphi_title_metadata base
inner join bolt_dai_ce_prod.gold.combined_video_stream metric 
    on coalesce(metric.WBD_MAX_SHOW_SEASON_ID,HBOMAX_SHOW_SEASON_ID, 0) = 
        concat(base.ckg_series_id, '-', coalesce(base.season_number, '0')) 
        and base.region=metric.region
where  metric.PROGRAM_ID_OR_VIEWABLE_ID IS NOT NULL 
    AND metric.CONTENT_MINUTES_WATCHED >= 2
    AND metric.video_type = 'main' 
    and metric.days_since_season_premiere_date_pst <=28
    and base.geo_level='REGION'
group by 
	base.title_series
	,base.season_number
	,base.ckg_match_id
    ,base.region
    ,to_date(base.first_release_date, 'yyyy-MM-dd')
    ,to_date(base.last_release_date, 'yyyy-MM-dd')
    ,case 
    when sub_tenure_months > 12 then '13+'
    when sub_tenure_months between 7 and 12 then '7-12'
    when sub_tenure_months between 4 and 6 then '4-6'
    else sub_tenure_months 
    end
    ,metric.request_date_pst
    ,metric.days_since_season_premiere_date_pst 
'''
  
run = spark.sql(query)
run


# COMMAND ----------

a='''
select * from bolt_cus_dev.gold.delphi_title_metadata 
where title_series=='Barbie'
and geo_level='REGION'
and content_category='movie'
'''
df=spark.sql(a)
df= df.toPandas()
display(df)

# COMMAND ----------



# COMMAND ----------

## Jeni's data 
'''
select 
 title_name
, ckg_series_id
, season_number
, medal
, season_window_start
, request_date
,case 
    when tenure > 12 then '13+'
    when tenure between 6 and 12 then '6-12'
    when tenure between 3 and 6 then '4-6'
    else tenure 
    end as tenure_grp
, DATEDIFF(request_date, season_window_start) AS days_since_premiere
, count(distinct user_id) as user_count
from bolt_cus_dev.bronze.cip_user_stream_subscription_metric_us base
left join bolt_cus_dev.gold.delphi_title_metadata delphi
on base.ckg_series_id = 
where DATEDIFF(request_date, season_window_start) <=28
group by 
 title_name
, ckg_series_id
, season_number
, medal
, season_window_start
, request_date
, case 
    when tenure > 12 then '13+'
    when tenure between 6 and 12 then '6-12'
    when tenure between 3 and 6 then '4-6'
    else tenure 
    end 
, DATEDIFF(request_date, season_window_start)

'''
