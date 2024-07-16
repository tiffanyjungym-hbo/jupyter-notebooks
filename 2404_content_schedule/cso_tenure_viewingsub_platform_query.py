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

current_shuffle_partitions = spark.conf.get("spark.sql.shuffle.partitions")
print(f"Current shuffle partitions: {current_shuffle_partitions}")
a =spark.conf.set("spark.sql.shuffle.partitions", 2000)

print(f"Current shuffle partitions: {a}")

# COMMAND ----------

## partition by startdate and tenuregrp, for US only.   

# COMMAND ----------

# MAGIC
# MAGIC %sql
# MAGIC -- Enable Delta Lake optimizations
# MAGIC SET spark.databricks.delta.optimizeWrite.enabled = true;
# MAGIC SET spark.databricks.delta.autoOptimize.optimizeWrite = true;
# MAGIC SET spark.databricks.delta.autoOptimize.autoCompact = true;
# MAGIC
# MAGIC -- Filter and create a temporary view
# MAGIC CREATE OR REPLACE TEMP VIEW user_video_stream_filtered AS
# MAGIC SELECT 
# MAGIC   wbd_max_or_hbomax_user_id,
# MAGIC   request_date_pst, 
# MAGIC   region,
# MAGIC   sub_tenure_months
# MAGIC FROM bolt_dai_ce_prod.gold.combined_video_stream 
# MAGIC WHERE PROGRAM_ID_OR_VIEWABLE_ID IS NOT NULL 
# MAGIC   AND CONTENT_MINUTES_WATCHED >= 2
# MAGIC   AND video_type = 'main' 
# MAGIC   AND region IN ('NORTH AMERICA');
# MAGIC
# MAGIC -- Create a date range table
# MAGIC CREATE OR REPLACE TEMP VIEW date_range AS
# MAGIC SELECT explode(sequence(
# MAGIC   to_date('2022-01-01'), 
# MAGIC   to_date('2024-07-05'), 
# MAGIC   interval 1 day
# MAGIC )) AS date;
# MAGIC
# MAGIC -- Join the date range with user requests and calculate the unique user count
# MAGIC CREATE OR REPLACE TEMP VIEW unique_user_counts AS
# MAGIC SELECT
# MAGIC   d.date AS start_date,
# MAGIC   CASE 
# MAGIC     WHEN v.sub_tenure_months > 12 THEN '13+'
# MAGIC     WHEN v.sub_tenure_months BETWEEN 7 AND 12 THEN '7-12'
# MAGIC     WHEN v.sub_tenure_months BETWEEN 4 AND 6 THEN '4-6'
# MAGIC     ELSE CAST(v.sub_tenure_months AS STRING)
# MAGIC   END AS tenure_grp,
# MAGIC   COUNT(DISTINCT v.wbd_max_or_hbomax_user_id) AS viewing_subscribers_28d
# MAGIC FROM date_range d
# MAGIC LEFT JOIN user_video_stream_filtered v
# MAGIC ON v.request_date_pst BETWEEN d.date AND date_add(d.date, 28)
# MAGIC GROUP BY d.date, 
# MAGIC   CASE 
# MAGIC     WHEN v.sub_tenure_months > 12 THEN '13+'
# MAGIC     WHEN v.sub_tenure_months BETWEEN 7 AND 12 THEN '7-12'
# MAGIC     WHEN v.sub_tenure_months BETWEEN 4 AND 6 THEN '4-6'
# MAGIC     ELSE CAST(v.sub_tenure_months AS STRING)
# MAGIC   END;
# MAGIC
# MAGIC -- Create Delta table partitioned by tenure_grp
# MAGIC CREATE OR REPLACE TABLE bolt_cus_dev.bronze.cso_cumulative_platform_viewership_tenure_us
# MAGIC USING DELTA
# MAGIC PARTITIONED BY (start_date, tenure_grp)
# MAGIC AS
# MAGIC SELECT * FROM unique_user_counts;

# COMMAND ----------

# MAGIC
# MAGIC %sql
# MAGIC -- Enable Delta Lake optimizations
# MAGIC SET spark.databricks.delta.optimizeWrite.enabled = true;
# MAGIC SET spark.databricks.delta.autoOptimize.optimizeWrite = true;
# MAGIC SET spark.databricks.delta.autoOptimize.autoCompact = true;
# MAGIC
# MAGIC -- Filter and create a temporary view
# MAGIC CREATE OR REPLACE TEMP VIEW user_video_stream_filtered AS
# MAGIC SELECT 
# MAGIC   wbd_max_or_hbomax_user_id,
# MAGIC   request_date_pst, 
# MAGIC   region,
# MAGIC   sub_tenure_months
# MAGIC FROM bolt_dai_ce_prod.gold.combined_video_stream 
# MAGIC WHERE PROGRAM_ID_OR_VIEWABLE_ID IS NOT NULL 
# MAGIC   AND CONTENT_MINUTES_WATCHED >= 2
# MAGIC   AND video_type = 'main' 
# MAGIC   AND region IN ('NORTH AMERICA','EMEA','LATAM');
# MAGIC
# MAGIC -- Create a date range table
# MAGIC CREATE OR REPLACE TEMP VIEW date_range AS
# MAGIC SELECT explode(sequence(
# MAGIC   to_date('2022-01-01'), 
# MAGIC   to_date('2024-07-05'), 
# MAGIC   interval 1 day
# MAGIC )) AS date;
# MAGIC
# MAGIC -- Join the date range with user requests and calculate the unique user count
# MAGIC CREATE OR REPLACE TEMP VIEW unique_user_counts AS
# MAGIC SELECT
# MAGIC   d.date AS start_date,
# MAGIC   v.region,
# MAGIC   CASE 
# MAGIC     WHEN v.sub_tenure_months > 12 THEN '13+'
# MAGIC     WHEN v.sub_tenure_months BETWEEN 7 AND 12 THEN '7-12'
# MAGIC     WHEN v.sub_tenure_months BETWEEN 4 AND 6 THEN '4-6'
# MAGIC     ELSE CAST(v.sub_tenure_months AS STRING)
# MAGIC   END AS tenure_grp,
# MAGIC   COUNT(DISTINCT v.wbd_max_or_hbomax_user_id) AS viewing_subscribers_28d
# MAGIC FROM date_range d
# MAGIC LEFT JOIN user_video_stream_filtered v
# MAGIC ON v.request_date_pst BETWEEN d.date AND date_add(d.date, 28)
# MAGIC GROUP BY d.date, v.region, 
# MAGIC   CASE 
# MAGIC     WHEN v.sub_tenure_months > 12 THEN '13+'
# MAGIC     WHEN v.sub_tenure_months BETWEEN 7 AND 12 THEN '7-12'
# MAGIC     WHEN v.sub_tenure_months BETWEEN 4 AND 6 THEN '4-6'
# MAGIC     ELSE CAST(v.sub_tenure_months AS STRING)
# MAGIC   END;
# MAGIC
# MAGIC -- Create Delta table partitioned by tenure_grp
# MAGIC CREATE OR REPLACE TABLE bolt_cus_dev.bronze.cso_cumulative_platform_viewership_tenure
# MAGIC USING DELTA
# MAGIC PARTITIONED BY (start_date)
# MAGIC AS
# MAGIC SELECT * FROM unique_user_counts;

# COMMAND ----------

# %sql
# -- Enable Delta Lake optimizations
# SET spark.databricks.delta.optimizeWrite.enabled = true;
# SET spark.databricks.delta.autoOptimize.optimizeWrite = true;
# SET spark.databricks.delta.autoOptimize.autoCompact = true;

# -- Filter and create a temporary view
# CREATE OR REPLACE TEMP VIEW user_video_stream_filtered AS
# SELECT 
#   wbd_max_or_hbomax_user_id,
#   request_date_pst, 
#   region,
#   sub_tenure_months
# FROM bolt_dai_ce_prod.gold.combined_video_stream 
# WHERE PROGRAM_ID_OR_VIEWABLE_ID IS NOT NULL 
#   AND CONTENT_MINUTES_WATCHED >= 2
#   AND video_type = 'main' 
#   AND region IN ('NORTH AMERICA','EMEA','LATAM');

# -- Create a date range table
# CREATE OR REPLACE TEMP VIEW date_range AS
# SELECT explode(sequence(
#   to_date('2022-01-01'), 
#   to_date('2024-07-05'), 
#   interval 1 day
# )) AS date;

# -- Join the date range with user requests and calculate the unique user count
# CREATE OR REPLACE TEMP VIEW unique_user_counts AS
# SELECT
#   d.date AS start_date,
#   v.region,
#   CASE 
#     WHEN v.sub_tenure_months > 12 THEN '13+'
#     WHEN v.sub_tenure_months BETWEEN 7 AND 12 THEN '7-12'
#     WHEN v.sub_tenure_months BETWEEN 4 AND 6 THEN '4-6'
#     ELSE CAST(v.sub_tenure_months AS STRING)
#   END AS tenure_grp,
#   COUNT(DISTINCT v.wbd_max_or_hbomax_user_id) AS viewing_subscribers_28d
# FROM date_range d
# LEFT JOIN user_video_stream_filtered v
# ON v.request_date_pst BETWEEN d.date AND date_add(d.date, 28)
# GROUP BY d.date, v.region, 
#   CASE 
#     WHEN v.sub_tenure_months > 12 THEN '13+'
#     WHEN v.sub_tenure_months BETWEEN 7 AND 12 THEN '7-12'
#     WHEN v.sub_tenure_months BETWEEN 4 AND 6 THEN '4-6'
#     ELSE CAST(v.sub_tenure_months AS STRING)
#   END;

# -- Create Delta table partitioned by tenure_grp
# CREATE OR REPLACE TABLE bolt_cus_dev.bronze.cso_cumulative_platform_viewership_tenure
# USING DELTA
# PARTITIONED BY (start_date)
# AS
# SELECT * FROM unique_user_counts;

# COMMAND ----------

# %sql
# -- Enable Delta Lake optimizations
# SET spark.databricks.delta.optimizeWrite.enabled = true;
# SET spark.databricks.delta.autoOptimize.optimizeWrite = true;
# SET spark.databricks.delta.autoOptimize.autoCompact = true;

# -- Filter and create a temporary view
# CREATE OR REPLACE TEMP VIEW user_video_stream_filtered_ AS
# SELECT 
#   wbd_max_or_hbomax_user_id,
#   request_date_pst, 
#   region,
#   sub_tenure_months
# FROM bolt_dai_ce_prod.gold.combined_video_stream 
# WHERE PROGRAM_ID_OR_VIEWABLE_ID IS NOT NULL 
#   AND CONTENT_MINUTES_WATCHED >= 2
#   AND video_type = 'main' 
#   AND region IN ('NORTH AMERICA','EMEA','LATAM');

# -- Optimize the source table by ZORDER on frequently queried columns
# OPTIMIZE bolt_dai_ce_prod.gold.combined_video_stream 
# ZORDER BY (request_date_pst, region);

# -- Create a date range table
# CREATE OR REPLACE TEMP VIEW date_range_ AS
# SELECT explode(sequence(
#   to_date('2022-01-01'), 
#   to_date('2024-07-05'), 
#   interval 1 day
# )) AS date;

# -- Join the date range with user requests and calculate the unique user count
# CREATE OR REPLACE TEMP VIEW unique_user_counts_ AS
# SELECT
#   d.date AS start_date,
#   v.region,
#   CASE 
#     WHEN v.sub_tenure_months > 12 THEN '13+'
#     WHEN v.sub_tenure_months BETWEEN 7 AND 12 THEN '7-12'
#     WHEN v.sub_tenure_months BETWEEN 4 AND 6 THEN '4-6'
#     ELSE CAST(v.sub_tenure_months AS STRING)
#   END AS tenure_grp,
#   COUNT(DISTINCT v.wbd_max_or_hbomax_user_id) AS viewing_subscribers_28d
# FROM date_range_ d
# LEFT JOIN user_video_stream_filtered_ v
# ON v.request_date_pst BETWEEN d.date AND date_add(d.date, 28)
# GROUP BY d.date, v.region, 
#   CASE 
#     WHEN v.sub_tenure_months > 12 THEN '13+'
#     WHEN v.sub_tenure_months BETWEEN 7 AND 12 THEN '7-12'
#     WHEN v.sub_tenure_months BETWEEN 4 AND 6 THEN '4-6'
#     ELSE CAST(v.sub_tenure_months AS STRING)
#   END;

# -- Create Delta table partitioned by tenure_grp
# CREATE OR REPLACE TABLE bolt_cus_dev.bronze.cso_cumulative_platform_viewership_tenure_
# USING DELTA
# PARTITIONED BY (tenure_grp)
# AS
# SELECT * FROM unique_user_counts_;

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
