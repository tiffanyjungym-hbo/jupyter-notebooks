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
# utils_od =  root_path + 'utils/utils_od.py'
utils_models =  root_path + 'utils/utils_models.py'

for file in (utils_imports, utils_sf, utils_import_data, utils_models):
    with open(file, "r") as file:
        python_code = file.read()
    exec(python_code)

pd.options.display.float_format = '{:,.2f}'.format
# path=!pwd
# sys.path.append(os.path.join(path[0], '../utils'))

# from imports import *
# # from import_data import *
# from utils_sf import *
# from utils_gs import *
# from utils_od import *


# COMMAND ----------

# %sql
# select * from bolt_cus_dev.gold.delphi_titles limit 10



# COMMAND ----------

## Nielson data 

# %sql
# select * from prod_ted_tbstnt_max_share.title_evaluation.nielsen_svod_programs_weekly limit 10

## Nielson
df = pd.read_sql("""
    select *
    from prod_ted_tbstnt_max_share.title_evaluation.nielsen_svod_programs_weekly
    limit 100
    """, 
    con)

df.head()

# COMMAND ----------

## Writing databricks table from pandas 

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("DataFrame to Table").getOrCreate()
sdf = spark.createDataFrame(df_decay)
full_table_name = f"bolt_cus_dev.bronze.cso_pct_vs_decay_pay1_platinum"
sdf.write.saveAsTable(full_table_name)

# COMMAND ----------

# df = pd.read_sql("""
#     select *
#     from max_prod.content_datascience.viewingsubs_metrics_train limit 100
#     """, 
#     con)

# df.head()

df = pd.read_sql("""
    select *
    from max_prod.content_datascience.title_season_metrics_platform 
    limit 100
    """, 
    con)

df.head()

# COMMAND ----------

# df = pd.read_sql("""
#     select *
#     from max_prod.content_datascience.viewingsubs_metrics_train limit 100
#     """, 
#     con)

# df.head()

df = pd.read_sql("""
    select *
    from max_prod.content_datascience.title_season_metrics_platform limit 100
    """, 
    con)

df.head()

# COMMAND ----------

df = pd.read_sql("""
    select *
    from max_prod.delphi.delphi_title_metadata limit 100
    """, 
    con)

df.MEDAL_US.unique()

# COMMAND ----------

df = pd.read_sql("""
    select 
    churn_type,
    termination_type,
    count(SUB_ID)
    from max_prod.bi_analytics.wbd_max_period_events
    where subscription_event_type='churn_event'
    group by (churn_type, termination_type)
    """, 
    con)

df 

# COMMAND ----------

# ## interval, no overlap 
# df_s['days_diff'] = None
# for index, row in df_s.iterrows():
#     # Filter previous rows with 'gold' or 'platinum' medals and 'last_release_date' before the current 'first_release_date'
#     eligible_rows = df_s[(df_s['medal'].isin(['Gold', 'Platinum'])) & (df_s['last_release_date'] < row['first_release_date'])]
    
#     if not eligible_rows.empty:
#         # Get the latest 'last_release_date' from the filtered rows
#         latest_date = eligible_rows['last_release_date'].max()
#         # Calculate the difference in days and update the column
#         df_s.at[index, 'days_diff'] = (row['first_release_date'] - latest_date).days

# df_s[['title_name','medal','first_release_date','last_release_date','days_diff']][df_s.first_release_date>='2024-01-03']

# ## interval, allow 28 day overlap 
# df_s['days_diff'] = None
# for index, row in df_s.iterrows():

#     # Filter previous rows with 'gold' or 'platinum' medals and 'last_release_date' before the current 'first_release_date'
#     eligible_rows = df_s[(df_s['medal'].isin(['Gold', 'Platinum'])) & (df_s['last_release_date'] < row['first_release_date'])]

#     if not eligible_rows.empty:
#         # Get the latest 'last_release_date' from the filtered rows
#         latest_date = eligible_rows['last_release_date'].max()
#         # Calculate the difference in days and update the column
#         df_s.at[index, 'days_diff'] = (row['first_release_date'] - latest_date).days

# COMMAND ----------

#v0 
def get_days_diff(df_in):
    df_in['last_release_date_ref'] = df_in['first_release_date'] + pd.DateOffset(days=14)
    df_in['first_release_date_ref'] = df_in['first_release_date'] - pd.DateOffset(days=28)
    df_in['days_diff'] = None
    for index, row in df_in.iterrows():
        ## Filter titles that was released before the title, and last release date is within 28d after title's release date
        eligible_rows = df_in[((df_in['medal'].isin(['Gold', 'Platinum'])) 
                            & (df_in['last_release_date'] < row['last_release_date_ref']) 
                            & (df_in['first_release_date'] < row['first_release_date']))]

        eligible_rows_plat = df_in[((df_in['medal'].isin(['Platinum'])) 
                    # & (df_in['last_release_date'] < row['last_release_date_ref']) 
                    & (df_in['first_release_date'] < row['first_release_date'])
                    & (df_in['last_release_date'] <= row['last_release_date_ref']))]
        if not eligible_rows_plat.empty:
            latest_date = eligible_rows_plat['last_release_date'].max()
            latest_title = eligible_rows_plat.loc[eligible_rows_plat['last_release_date'] == latest_date, 'title_name'].iloc[0]
            latest_title_fv = eligible_rows_plat.loc[eligible_rows_plat['last_release_date'] == latest_date, 'rfv7'].iloc[0]
            df_in.at[index, 'days_diff'] = (row['first_release_date'] - latest_date).days
            df_in.at[index, 'prev_title'] = latest_title
            df_in.at[index, 'prev_title_fv'] = latest_title_fv
        elif not eligible_rows.empty:
            latest_date = eligible_rows['last_release_date'].max()
            latest_title = eligible_rows.loc[eligible_rows['last_release_date'] == latest_date, 'title_name'].iloc[0]
            latest_title_fv = eligible_rows.loc[eligible_rows['last_release_date'] == latest_date, 'rfv7'].iloc[0]
            df_in.at[index, 'days_diff'] = (row['first_release_date'] - latest_date).days
            df_in.at[index, 'prev_title'] = latest_title
            df_in.at[index, 'prev_title_fv'] = latest_title_fv
        else:
            pass

    df_in = df_in.fillna(0)
    df_in.days_diff = df_in.days_diff.astype('int')
    # df_in = df_in.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df_in


# COMMAND ----------


def get_days_diff(df_in):
    df_in['last_release_date_ref'] = df_in['first_release_date'] + pd.DateOffset(days=28)
    # df_in['first_release_date_ref'] = df_in['first_release_date'] - pd.DateOffset(days=28)
    df_in['days_diff'] = None
    for index, row in df_in.iterrows():
        ## Eligibility: 1) released before current title, 2)last episode is within 28d after title's premiere date
        eligible_rows = df_in[((df_in['medal'].isin(['Gold', 'Platinum'])) 
                            & (df_in['last_release_date'] <= row['last_release_date_ref']) 
                            & (df_in['first_release_date'] < row['first_release_date']))]

        eligible_rows_plat = df_in[((df_in['medal'].isin(['Platinum'])) 
                    & (df_in['first_release_date'] < row['first_release_date'])
                    & (df_in['last_release_date'] <= row['last_release_date_ref']))]
        if not eligible_rows_plat.empty:
            latest_date = eligible_rows_plat['last_release_date'].max()
            latest_title = eligible_rows_plat.loc[eligible_rows_plat['last_release_date'] == latest_date, 'title_name'].iloc[0]
            latest_title_fv = eligible_rows_plat.loc[eligible_rows_plat['last_release_date'] == latest_date, 'rfv7'].iloc[0]
            df_in.at[index, 'days_diff'] = (row['first_release_date'] - latest_date).days
            df_in.at[index, 'prev_title'] = latest_title
            df_in.at[index, 'prev_title_fv'] = latest_title_fv
        elif not eligible_rows.empty:
            latest_date = eligible_rows['last_release_date'].max()
            latest_title = eligible_rows.loc[eligible_rows['last_release_date'] == latest_date, 'title_name'].iloc[0]
            latest_title_fv = eligible_rows.loc[eligible_rows['last_release_date'] == latest_date, 'rfv7'].iloc[0]
            df_in.at[index, 'days_diff'] = (row['first_release_date'] - latest_date).days
            df_in.at[index, 'prev_title'] = latest_title
            df_in.at[index, 'prev_title_fv'] = latest_title_fv
        else:
            pass

    df_in = df_in.fillna(0)
    df_in.days_diff = df_in.days_diff.astype('int')
    # df_in = df_in.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df_in

# COMMAND ----------


def get_days_diff(df_in):
    df_in['last_release_date_ref'] = df_in['first_release_date'] - pd.DateOffset(days=200) #titles that ended after 28d pre-premiere
    df_in['last_release_date_ref_'] = df_in['first_release_date'] + pd.DateOffset(days=28) #titles that ended before 28d post premiere
    df_in['days_diff'] = None
    for index, row in df_in.iterrows():
        for filter in [['Gold','Platinum'], ['Platinum']]: ## Prioritize Platinum titles that last aired after 28d ago by overwriting   
            eligible_rows = df_in[((df_in['medal'].isin(filter)) 
                            & (df_in['first_release_date'] < row['first_release_date'])
                            # & (df_in['last_release_date'] >= row['last_release_date_ref'])
                            & (df_in['last_release_date'] <= row['last_release_date_ref_']))]
            if not eligible_rows.empty: 
                # eligible_rows['days_diff'] = (row['first_release_date'] - eligible_rows['last_release_date']).dt.days
                # min_days_diff = eligible_rows['days_diff'].min()
                # latest_title = eligible_rows.loc[eligible_rows['days_diff'] == min_days_diff, 'title_name'].iloc[0]
                # latest_date = eligible_rows.loc[eligible_rows['days_diff'] == min_days_diff, 'last_release_date'].iloc[0]
                # latest_title_fv = eligible_rows.loc[eligible_rows['days_diff'] == min_days_diff, 'rfv7'].iloc[0]
                # df_in.at[index, 'days_diff'] = (row['first_release_date'] - latest_date).days
                # df_in.at[index, 'prev_title'] = latest_title
                # df_in.at[index, 'prev_title_fv'] = latest_title_fv
                latest_date = eligible_rows['last_release_date'].max()
                latest_title = eligible_rows.loc[eligible_rows['last_release_date'] == latest_date, 'title_name'].iloc[0]
                latest_title_fv = eligible_rows.loc[eligible_rows['last_release_date'] == latest_date, 'rfv7'].iloc[0]
                df_in.at[index, 'days_diff'] = (row['first_release_date'] - latest_date).days
                df_in.at[index, 'prev_title'] = latest_title
                df_in.at[index, 'prev_title_fv'] = latest_title_fv

            # elif eligible_rows.empty and filter==['Gold','Platinum']:
            #     eligible_rows = df_in[((df_in['medal'].isin(filter)) 
            #                            & (df_in['first_release_date'] < row['first_release_date'])
            #                            & (df_in['last_release_date'] <= row['last_release_date_ref_']))]
            else:
                pass
    df_in = df_in.fillna(0)
    df_in.days_diff = df_in.days_diff.astype('int')
    # df_in.loc[df_in.days_diff<-28, 'days_diff'] = -28
    return df_in
