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

for file in (utils_imports,  utils_import_data, utils_models):#utils_sf,
    with open(file, "r") as file:
        python_code = file.read()
    exec(python_code)

pd.options.display.float_format = '{:,.3f}'.format


# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE bolt_cus_dev.bronze.cso_cumulative_platform_viewership_tenure

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE OR REPLACE TEMP VIEW user_video_stream_filtered_ AS
# MAGIC SELECT 
# MAGIC wbd_max_or_hbomax_user_id,
# MAGIC request_date_pst, 
# MAGIC region,
# MAGIC sub_tenure_months
# MAGIC FROM bolt_dai_ce_prod.gold.combined_video_stream 
# MAGIC where PROGRAM_ID_OR_VIEWABLE_ID IS NOT NULL 
# MAGIC     AND CONTENT_MINUTES_WATCHED >= 2
# MAGIC     AND video_type = 'main' 
# MAGIC     and region in ('NORTH AMERICA','EMEA','LATAM');
# MAGIC
# MAGIC -- Create a table of dates
# MAGIC CREATE OR REPLACE TEMP VIEW date_range_ AS
# MAGIC SELECT explode(sequence(
# MAGIC   to_date('2022-01-01'), 
# MAGIC   to_date('2024-07-05'), 
# MAGIC   interval 1 day
# MAGIC )) AS date;
# MAGIC
# MAGIC -- Join the date range with user requests and calculate the unique user count
# MAGIC CREATE OR REPLACE TEMP VIEW unique_user_counts_ AS
# MAGIC SELECT
# MAGIC   d.date as start_date,
# MAGIC   v.region,
# MAGIC   case 
# MAGIC       when v.sub_tenure_months > 12 then '13+'
# MAGIC       when v.sub_tenure_months between 7 and 12 then '7-12'
# MAGIC       when v.sub_tenure_months between 4 and 6 then '4-6'
# MAGIC       else v.sub_tenure_months 
# MAGIC       end as tenure_grp,
# MAGIC   COUNT(DISTINCT v.wbd_max_or_hbomax_user_id) AS viewing_subscribers_28d
# MAGIC FROM
# MAGIC   date_range_ d
# MAGIC LEFT JOIN
# MAGIC   user_video_stream_filtered_ v
# MAGIC ON
# MAGIC   v.request_date_pst BETWEEN d.date AND date_add(d.date, 28)
# MAGIC GROUP BY
# MAGIC   d.date,
# MAGIC   v.region,
# MAGIC   case 
# MAGIC     when v.sub_tenure_months > 12 then '13+'
# MAGIC     when v.sub_tenure_months between 7 and 12 then '7-12'
# MAGIC     when v.sub_tenure_months between 4 and 6 then '4-6'
# MAGIC     else v.sub_tenure_months 
# MAGIC     end ;
# MAGIC
# MAGIC
# MAGIC -- Create Delta table partitioned by region
# MAGIC CREATE OR REPLACE TABLE bolt_cus_dev.bronze.cso_cumulative_platform_viewership_tenure_
# MAGIC USING DELTA
# MAGIC PARTITIONED BY (tenure_grp)
# MAGIC AS
# MAGIC SELECT 
# MAGIC * 
# MAGIC FROM unique_user_counts_;
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE bolt_cus_dev.bronze.cso_cumulative_platform_viewership_tenure
# MAGIC USING DELTA
# MAGIC PARTITIONED BY (start_date)
# MAGIC
# MAGIC WITH filtered_metadata AS (
# MAGIC     SELECT distinct cast(first_release_date as DATE) as first_release_date
# MAGIC     FROM bolt_cus_dev.gold.delphi_title_metadata
# MAGIC ),
# MAGIC date_range AS (
# MAGIC     SELECT
# MAGIC         first_release_date,
# MAGIC         date_add(first_release_date, 28) AS end_date
# MAGIC     FROM
# MAGIC         filtered_metadata
# MAGIC )
# MAGIC
# MAGIC SELECT
# MAGIC     dr.first_release_date as start_date,
# MAGIC     t.sub_tenure_months,
# MAGIC     t.region,
# MAGIC     COUNT(DISTINCT t.WBD_MAX_OR_HBOMAX_USER_ID) AS viewing_subscribers_28d
# MAGIC FROM date_range dr
# MAGIC JOIN bolt_dai_ce_prod.gold.combined_video_stream t 
# MAGIC     ON t.request_date_pst >= dr.first_release_date 
# MAGIC     AND t.request_date_pst < dr.end_date
# MAGIC where  t.PROGRAM_ID_OR_VIEWABLE_ID IS NOT NULL 
# MAGIC     AND t.CONTENT_MINUTES_WATCHED >= 2
# MAGIC     AND t.video_type = 'main' 
# MAGIC     and t.region in ('NORTH AMERICA','EMEA','LATAM')
# MAGIC GROUP BY dr.first_release_date,
# MAGIC     t.sub_tenure_months,
# MAGIC     t.region
# MAGIC ORDER BY dr.first_release_date,
# MAGIC     t.sub_tenure_months,
# MAGIC     t.region;
# MAGIC

# COMMAND ----------

df = spark.sql('''select * from bolt_cus_dev.bronze.cso_cumulative_platform_viewership_tenure''')
df = df.toPandas()
display(df)

# COMMAND ----------

df[df.region.isin(['NORTH AMERICA','EMEA','LATAM'])]
