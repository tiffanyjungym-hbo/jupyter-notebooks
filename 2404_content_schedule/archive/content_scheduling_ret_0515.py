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

query = """
select 
* 
from bolt_cus_dev.silver.forecasting_feature_space_test f
left join bolt_cus_dev.gold.delphi_title_metadata d
on f.imdb_id=d.imdb_series_id 
and f.season_number= d.season_number
and f.region = d.geo_value 
where f.premiere_date>='2021-01-01'
    and f.premiere_date<'2024-04-01'
    and d.geo_level='REGION'"""

# query_ref = """
# select 
# f.title_name,
# f.season_number,
# f.imdb_id,
# i.reference_title_id,
# i.reference_reference_title_id
# from bolt_cus_dev.silver.forecasting_feature_space_test f
# left join bolt_cus_dev.silver.delphi_title_imdb_refs i
# on f.imdb_id=i.imdb_series_id and f.season_number= i.season_number
# where f.premiere_date>='2021-01-01'
#     and f.premiere_date<'2024-04-01'
#     and f.region='NORTH AMERICA'
#     and i.geo_value='UNITED STATES'
# """

query_ref = """
select 
    f.title_name,
    f.season_number,
    f.imdb_id,
 	  itr.title_id as reference_title_id,
 	  itrr.title_id as reference_reference_title_id
from bolt_cus_dev.silver.forecasting_feature_space_test f
left join bolt_dai_ckg_prod.gold.imdb_title it
 	on f.imdb_id=it.title_id
left join bolt_dai_ckg_prod.gold.imdb_movie_connection as imc
 	on it.title_id = imc.title_id
left join bolt_dai_ckg_prod.gold.imdb_title itr
 	on itr.title_id = imc.reference_title_id
left join bolt_dai_ckg_prod.gold.imdb_movie_connection as imcr
 	on itr.title_id = imcr.title_id
 	 	and imcr.reference_type in ('featured_in')
left join bolt_dai_ckg_prod.gold.imdb_title itrr
 	on itrr.title_id = imcr.reference_title_id
where imc.reference_type in ('follows', 'spin_off_from', 'remake_of', 'version_of', 'featured_in')
and it.production_release_date > itr.production_release_date
and it.production_release_date > itrr.production_release_date
"""

df_raw = spark.sql(query)
df_raw = df_raw.toPandas()
df_raw.rename(columns={'season_number': 'season_number_'}, inplace=True)
df_raw.columns=df_raw.columns.str.lower()

## Clean data 
df_raw = df_raw[(df_raw.rfv7.notnull()) & (df_raw.rfv7>0)]
df_raw = df_raw.drop_duplicates(subset=['title_name','season_number','rfv7'])
df_raw['medal'] = np.where(df_raw['medal']=='Other', df_raw['predicted_medal_us'],df_raw['medal'])
print(df_raw.shape)

# ## get ref feature 
df_ref = spark.sql(query_ref)
df_ref = df_ref.toPandas()

df_ref = df_ref.groupby(['title_name','season_number','imdb_id']).agg({'reference_title_id':'nunique', 
                                           'reference_reference_title_id':'nunique'}).reset_index()
df_ref = df_ref.rename(columns={'reference_title_id':'ref_count',
                        'reference_reference_title_id':'ref_popularity'}) 

## Merge
df_raw = df_raw.merge(df_ref, 
                      on=['title_name', 'season_number', 'imdb_id'], 
                      how='left')


## Data cleaning 
df_raw['premiere_date'] = pd.to_datetime(df_raw['premiere_date'])
df_raw['first_release_date'] = pd.to_datetime(df_raw['first_release_date'])
df_raw['last_release_date'] = pd.to_datetime(df_raw['last_release_date'])
df_raw.loc[(df_raw.content_category=='movie') & (df_raw.region=='EMEA'), 'last_release_date'] = df_raw['first_release_date'] + pd.Timedelta(days=1)
df_raw.gap = np.where(df_raw.forecast_category=='popcorn', 0, df_raw.gap)

df_raw = df_raw.sort_values(by=['region','first_release_date','last_release_date']) 

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

def get_df(df_m, medal, forecast_category,col_id, features, target):
    df_in = df_m.query("forecast_category==@forecast_category & medal==@medal")[col_id + features + [target]]
    return df_in

def log_features(df_in, features, target):
    for i in features+[target]:
        try:
            df_in['ln_'+i] = np.log(df_in[i].astype(float)+1)
        except:
            print(i)
            pass
    return df_in 

def transform_features(df_in, days_diff, features, target):
    for d in days_diff:
        min_days_diff = df_in[d].min()
        df_in[d] = df_in[d] + (-min_days_diff) + 1

    df_in = log_features(df_in, features, target)

    for d in days_diff:
        df_in[d] = df_in[d] + min_days_diff -1
    return df_in

def train(df_train,target, features):
    model = sm.quantreg('ln_' + target + ' ~ ' + features, data=df_train.query("future_title==0")).fit()
    df_train['fv_predict'] = np.exp(model.predict(df_train)) + 1
    df_train['error'] = df_train['ln_' + target] - model.predict(df_train)
    df_train = df_train.loc[~df_train.error.isnull()]
    return model, df_train

def get_coef(model, model_name, target, region):
    # Model coefficients
        df_coef = (model.params.reset_index()
                .merge(model.tvalues.reset_index(), on='index')
                .merge(model.pvalues.reset_index(), on='index')
                )
        df_coef.columns = ['feature','coef','tstat','pvalue']
        df_coef['model'] = model_name
        df_coef['target'] = target
        df_coef['region'] = region
        return df_coef[['region','model','feature','coef','pvalue']]

# Return model RSQ and MAPE
def get_summary(df_in, model, model_name, features, target,region):
    df_summary = pd.DataFrame({
                    'region':[region],
                    'model':[model_name],
                    'target':[target], 
                    'features':[features],
                    'rsquared':[model.prsquared],
                    'n_obs':[len(df_in.query("future_title==0 & error==error"))],
                    'med_ape':[np.abs(df_in.query("future_title==0").error).median()],
                    'mape':[np.abs(df_in.query("future_title==0").error).mean()]
                    })
    return df_summary

# COMMAND ----------

query = """
select 
* 
from bolt_cus_dev.silver.forecasting_feature_space_test f
left join bolt_cus_dev.gold.delphi_title_metadata d
on f.imdb_id=d.imdb_series_id 
and f.season_number= d.season_number
and f.region = d.geo_value 
where f.premiere_date>='2021-01-01'
    and f.premiere_date<'2024-04-01'
    and d.geo_level='REGION'"""

# query_ref = """
# select 
# f.title_name,
# f.season_number,
# f.imdb_id,
# i.reference_title_id,
# i.reference_reference_title_id
# from bolt_cus_dev.silver.forecasting_feature_space_test f
# left join bolt_cus_dev.silver.delphi_title_imdb_refs i
# on f.imdb_id=i.imdb_series_id and f.season_number= i.season_number
# where f.premiere_date>='2021-01-01'
#     and f.premiere_date<'2024-04-01'
#     and f.region='NORTH AMERICA'
#     and i.geo_value='UNITED STATES'
# """

query_ref = """
select 
    f.title_name,
    f.season_number,
    f.imdb_id,
 	  itr.title_id as reference_title_id,
 	  itrr.title_id as reference_reference_title_id
from bolt_cus_dev.silver.forecasting_feature_space_test f
left join bolt_dai_ckg_prod.gold.imdb_title it
 	on f.imdb_id=it.title_id
left join bolt_dai_ckg_prod.gold.imdb_movie_connection as imc
 	on it.title_id = imc.title_id
left join bolt_dai_ckg_prod.gold.imdb_title itr
 	on itr.title_id = imc.reference_title_id
left join bolt_dai_ckg_prod.gold.imdb_movie_connection as imcr
 	on itr.title_id = imcr.title_id
 	 	and imcr.reference_type in ('featured_in')
left join bolt_dai_ckg_prod.gold.imdb_title itrr
 	on itrr.title_id = imcr.reference_title_id
where imc.reference_type in ('follows', 'spin_off_from', 'remake_of', 'version_of', 'featured_in')
and it.production_release_date > itr.production_release_date
and it.production_release_date > itrr.production_release_date
"""

df_raw = spark.sql(query)
df_raw = df_raw.toPandas()
df_raw.rename(columns={'season_number': 'season_number_'}, inplace=True)
df_raw.columns=df_raw.columns.str.lower()

## Clean data 
df_raw = df_raw[(df_raw.rfv7.notnull()) & (df_raw.rfv7>0)]
df_raw = df_raw.drop_duplicates(subset=['title_name','season_number','rfv7'])
df_raw['medal'] = np.where(df_raw['medal']=='Other', df_raw['predicted_medal_us'],df_raw['medal'])
print(df_raw.shape)

# ## get ref feature 
df_ref = spark.sql(query_ref)
df_ref = df_ref.toPandas()

df_ref = df_ref.groupby(['title_name','season_number','imdb_id']).agg({'reference_title_id':'nunique', 
                                           'reference_reference_title_id':'nunique'}).reset_index()
df_ref = df_ref.rename(columns={'reference_title_id':'ref_count',
                        'reference_reference_title_id':'ref_popularity'}) 

## Merge
df_raw = df_raw.merge(df_ref, 
                      on=['title_name', 'season_number', 'imdb_id'], 
                      how='left')


## Data cleaning 
df_raw['premiere_date'] = pd.to_datetime(df_raw['premiere_date'])
df_raw['first_release_date'] = pd.to_datetime(df_raw['first_release_date'])
df_raw['last_release_date'] = pd.to_datetime(df_raw['last_release_date'])
df_raw.loc[(df_raw.content_category=='movie') & (df_raw.region=='EMEA'), 'last_release_date'] = df_raw['first_release_date'] + pd.Timedelta(days=1)
df_raw.gap = np.where(df_raw.forecast_category=='popcorn', 0, df_raw.gap)

df_raw = df_raw.sort_values(by=['region','first_release_date','last_release_date']) 

# COMMAND ----------

col = ['region', 'imdb_id', 'season_number', 'title_name', 'full_title',
       'num_episodes', 'medal', 'marketing_spend', 'production_cost',
       'forecast_category', 'segment', 'premiere_date', 'pvc', 'future_title',
       'rfv7', 'rfv90', 'tfv90', 'tfv90_prior', 'search_2wbp', 'wiki_2wbp', 'cfv_2wbp',
       'awareness_2wbp', 'itv_2wbp', 'search_1wbp', 'wiki_1wbp', 'cfv_1wbp',
       'awareness_1wbp', 'itv_1wbp', 
       'views_owned_teaser', 'views_owned_trailer', 'search_teaser', 'wiki_teaser', 'search_trailer', 'wiki_trailer', 'premiere_date_bo', 'gap', 'bo_opening',
       'bo_gross', 'bo_opening_imputed', 'bo_gross_imputed', 'fv_pre_30',
       'fv_pre_7', 'fv_pre_30_pay1', 'fv_pre_7_pay1', 'fv_pre_30_series',
       'fv_pre_7_series', 'content_category',
       'content_source', 'imdb_series_id',
       'first_release_date_utc', 'first_release_date', 'last_release_date',
       'reporting_net_studio', 'program_type', 'primary_genre',
       'reporting_primary_genre', 'scripted_flag',
       'ever_pay_1_title', 'lop_flag', 'release_year',
       'predicted_medal_us',  'asset_count',
       'asset_runtime',  'duplicate_season_ind',
       'runtime_in_mins', 'derived_genre',
       'ref_count', 'ref_popularity']

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

df_s=[]

for i in ['NORTH AMERICA','LATAM','EMEA']:
    print(i)
    df_r = get_days_diff(df_raw[df_raw.region==i])
    df_s.append(df_r)
    display(df_r[['region','title_name','medal','first_release_date','last_release_date','last_release_date_ref','days_diff', 'prev_title']].sort_values(by=['first_release_date']))

df_s = pd.concat(df_s)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Pay1

# COMMAND ----------

## Get DF 
target = 'rfv7'
medal = ['Gold','Platinum']
forecast_category = ['wb pay1']
region = ['NORTH AMERICA','LATAM','EMEA']
col_id = ['imdb_id','title_name','season_number','premiere_date','region','forecast_category','future_title']
features = ['bo_opening_imputed', 'gap', 'production_cost', 'medal', 'fv_pre_7','days_diff', 'prev_title', 'prev_title_fv', 'ref_count', 'ref_popularity', 'fv_pre_7_pay1','fv_pre_7_series']


df_in = get_df(df_s[df_s.premiere_date<='2024-05-01'], medal, forecast_category, col_id, features, target)

## data cleaning 
df_in.loc[df_in.production_cost==0, 'production_cost'] = df_in['production_cost'].median()
df_in.loc[(df_in.title_name=='House Party') & (df_in.region=='LATAM'), 'gap'] = 50
df_in.loc[(df_in.title_name=="Magic Mike's Last Dance") & (df_in.region=='LATAM'), 'gap'] = 112
df_in.loc[(df_in.title_name=="Magic Mike's Last Dance") & (df_in.region=='LATAM'), 'bo_opening_imputed'] = 5556243.301106308 #removing negative

df_in = transform_features(df_in, ['days_diff'], features, target)


## Remove outliers; Batman: 150K fv; Magic Mike: 3K fv 
titles_removed= {'NORTH AMERICA':['The Batman'],
                 'EMEA':['The Batman'],
                 'LATAM':['The Batman', 'House Party']} ##"Magic Mike's Last Dance", 'House Party',

df_in = df_in[((df_in.region=='NORTH AMERICA') & ~(df_in.title_name=='The Batman')) |
              ((df_in.region=='EMEA') & ~(df_in.title_name.isin(['The Batman', 'House Party']))) | 
              ((df_in.region=='LATAM') & ~(df_in.title_name.isin(['The Batman', 'House Party'])))]

# COMMAND ----------

## Check df 
display(df_in.sort_values(by=['region','title_name']))

import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from itertools import combinations

variables = list(df_in[['ln_rfv7','ln_days_diff', 'ln_fv_pre_7_pay1', 'ln_fv_pre_7', 'ln_gap','ln_bo_opening_imputed','ln_production_cost']])
variable_pairs = list(combinations(variables, 2))
num_plots = len(variable_pairs)
rows = int(num_plots**0.5)
cols = (num_plots + rows - 1) // rows  # Ensure all plots fit
fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f'{v1} vs {v2}' for v1, v2 in variable_pairs])


for index, (var1, var2) in enumerate(variable_pairs, 1):
    row = (index - 1) // cols + 1
    col = (index - 1) % cols + 1
    scatter = px.scatter(df_in, x=var2, y=var1, hover_data=['title_name','medal'], color='region')
    for trace in scatter.data:
        fig.add_trace(trace, row=row, col=col, )

for annot in fig['layout']['annotations']:
    annot['font'] = dict(size=10)  # Customize font size and color here

fig.update_layout(height=1000, width=1000, title_font=dict(size=10), font=dict(size=9))
fig.show()

# COMMAND ----------


model_list = {
    # 'pay_ogmd': 'ln_bo_opening_imputed + ln_gap + ln_days_diff  + medal',
    # 'pay_ogmdp': 'ln_bo_opening_imputed + ln_gap + ln_days_diff + ln_fv_pre_7_pay1 + medal',
    # 'pay_ogmpc': 'ln_bo_opening_imputed + ln_gap + ln_fv_pre_7_pay1 + medal + ln_production_cost',
    # 'pay_ogmdc': 'ln_bo_opening_imputed + ln_gap + ln_days_diff + medal + ln_production_cost',
    'pay_ogmdpc': 'ln_bo_opening_imputed + ln_gap + medal + ln_days_diff  + ln_fv_pre_7_pay1 + ln_production_cost',
    # 'pay_ogdmtc': 'ln_bo_opening_imputed + ln_gap + medal + ln_days_diff  + ln_fv_pre_7 + ln_production_cost',
}
df_coef=[]
df_summary =[]

for i in model_list:
    for r in region:
        model, df_in_ = train(df_in.query("region==@r"), target, model_list[i])
        df_coef.append(get_coef(model, i, target, r))
        df_summary.append(get_summary(df_in_, model, i, model_list[i], target, r))

df_summary = pd.concat(df_summary)
df_summary = df_summary.sort_values(by=['region','model'])
df_coef = pd.concat(df_coef).sort_values(by=['region','model','feature'])


# COMMAND ----------

display(df_summary)

# COMMAND ----------

df_coef[df_coef.region=='NORTH AMERICA']

# COMMAND ----------

df_coef[df_coef.region=='EMEA']

# COMMAND ----------

df_coef[df_coef.region=='LATAM']

# COMMAND ----------

df_coef[df_coef.feature=='ln_days_diff']

# COMMAND ----------

df_coef[df_coef.feature=='ln_fv_pre_7_pay1']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sensitivity analysis

# COMMAND ----------

query = """
select 
title_name,
f.season_number,
imdb_id,
premiere_date,
forecast_category,
rfv7,
medal,
num_episodes,
gap,
bo_opening_imputed,
production_cost,
last_release_date
from bolt_cus_dev.silver.forecasting_feature_space_test f
left join bolt_cus_dev.gold.delphi_title_metadata d
on f.imdb_id=d.imdb_series_id and f.season_number= d.season_number and f.region=d.geo_value
where  f.premiere_date>='2024-04-01'
    and f.premiere_date<'2024-12-01'
    and f.region='NORTH AMERICA'
    
     """

df_ss = spark.sql(query)
df_ss = df_ss.toPandas()
df_ss.columns = df_ss.columns.str.lower()
df_ss.premiere_date=pd.to_datetime(df_ss.premiere_date)
print(df_ss.premiere_date.max())

## Data cleaning 
df_ss['last_release_date'] = np.where(df_ss.forecast_category=='series',
                                      df_ss['premiere_date'] + pd.to_timedelta(49, unit='D'),
                                      df_ss['premiere_date'])
df_ss['first_release_date'] = df_ss['premiere_date']
df_ss.loc[df_ss.title_name=='Furiosa','gap'] = 77

df_ss2 = df_ss.copy()
## GxK
# df_ss2.loc[df_ss2['title_name']=='Godzilla x Kong: The New Empire', 'premiere_date'] = '2024-06-14'
# df_ss2.loc[df_ss2['title_name']=='Godzilla x Kong: The New Empire', 'first_release_date'] = '2024-06-14'
# df_ss2.loc[df_ss2['title_name']=='Godzilla x Kong: The New Empire', 'last_release_date'] = '2024-06-14'
# df_ss2.loc[df_ss2['title_name']=='Godzilla x Kong: The New Empire', 'gap'] = 77

# Furiosa
delta_days = 21
df_ss2.loc[df_ss2['title_name']=='Furiosa', 'premiere_date'] = '2024-08-30'
df_ss2.loc[df_ss2['title_name']=='Furiosa', 'first_release_date'] = '2024-08-30'
df_ss2.loc[df_ss2['title_name']=='Furiosa', 'last_release_date'] = '2024-08-30'
df_ss2.loc[df_ss2['title_name']=='Furiosa', 'gap'] = 98

df_ss = get_days_diff(df_ss)
df_ss2 = get_days_diff(df_ss2)

df_sc = df_ss.merge(df_ss2[['title_name','season_number','gap','first_release_date','last_release_date', 'prev_title','days_diff']], on=['title_name','season_number'], how='left', suffixes=('','_2'))

df_sc['delta_days_diff'] = df_sc['days_diff_2']- df_sc['days_diff'] 
df_sc['delta_gap'] = df_sc['gap_2']- df_sc['gap']                        

# COMMAND ----------

df_ss[(df_ss.medal=='Platinum') | (df_ss.forecast_category=='pay1_wb')]

# COMMAND ----------

df_fv = pd.read_sql("""
select 
    date, 
    sum(retail_fv) as fv_retail, 
    sum(retail_fv_series) as fv_series, 
    sum(retail_fv_pay1) as fv_pay1
from forecasting_fv_pacing_category a
left join forecasting_dates using (date)
where 
region = 'NORTH AMERICA' 
and year(date) = 2024
group by 1 order by 1""", con)

df_fv_title = pd.read_sql("""
select 
    date, 
    retail_fv as fv_title
from max_dev.workspace.forecasting_fv_pacing a
where 
region = 'NORTH AMERICA' 
and year(date) = 2024
-- and imdb_id='tt14539740' --gxk
and imdb_id = 'tt12037194' --Furiosa
""", con) 

def get_rolling_sum(df, columns):
    df.columns=df.columns.str.lower()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    display(df.head())
    rolling_sums = {}
    for col in columns:
        rolling_sums[col] = df[col].rolling(window=7).sum()
    df_out = pd.DataFrame(rolling_sums).reset_index()
    return df_out


#Clean data
df_fv = df_fv.reset_index()
df_fv.columns = df_fv.columns.str.lower()
df_fv.date = pd.to_datetime(df_fv.date)

## Get scenario 2 (gxk release 2 07/05-> 06/14, 21d shift; furiosa 8/9 ->8/30)
df_fv_title = df_fv_title.reset_index()
df_fv_title.columns=df_fv_title.columns.str.lower()
df_fv_title['date'] = pd.to_datetime(df_fv_title['date'])

df_fv_title_new=  df_fv_title.copy()
df_fv_title_new['date'] = df_fv_title_new['date'] + pd.to_timedelta(delta_days, unit='D')
df_fv_title_new['date'] = pd.to_datetime(df_fv_title_new['date'])

## merge sc1 + sc2
df_fv_ = df_fv.merge(df_fv_title, on=['date'],how='left').merge(df_fv_title_new, on=['date'],how='left', suffixes=('','_2'))
df_fv_ = df_fv_.fillna(0)
df_fv_['fv_pay1_2'] = df_fv_['fv_pay1'] - df_fv_['fv_title'] + df_fv_['fv_title_2']
df_fv_['fv_series_2'] = df_fv_['fv_series']
df_fv_['fv_retail_2'] = df_fv_['fv_retail']  - df_fv_['fv_title'] + df_fv_['fv_title_2']


## Get rolling first views by date, for scenario 1 vs 2 
df_fv_sc = get_rolling_sum(df_fv_, ['fv_retail','fv_series','fv_pay1','fv_retail_2','fv_series_2','fv_pay1_2'])
df_fv_sc['date'] = df_fv_sc['date'] - pd.to_timedelta(1, unit='D') ## shift date to exclude today from rolling

df_fv_sc['delta_fv_pre_7_pay1'] = df_fv_sc['fv_pay1_2'] - df_fv_sc['fv_pay1']
df_fv_sc['delta_fv_pre_7'] = df_fv_sc['fv_retail_2'] - df_fv_sc['fv_retail']

df_fv_sc = df_fv_sc.rename(columns={'fv_pay1': 'fv_pre_7_pay1',
                                    'fv_series': 'fv_pre_7_series',
                                    'fv_retail': 'fv_pre_7_retail',
                                    'fv_pay1_2': 'fv_pre_7_pay1_2',
                                    'fv_series_2': 'fv_pre_7_series_2',
                                    'fv_retail_2': 'fv_pre_7_retail_2',})



# COMMAND ----------

df_fv_sc.query('date > "2024-08-01" and date<"2024-09-15"')

# COMMAND ----------

## Merge dates and get delta_feature 
df_sc_fin = df_sc.merge(df_fv_sc[['date','fv_pre_7_pay1','fv_pre_7_series', 'fv_pre_7_retail','fv_pre_7_pay1_2','fv_pre_7_series_2', 'fv_pre_7_retail_2','delta_fv_pre_7_pay1']], left_on ='premiere_date', right_on='date', how='left')
df_sc_fin = df_sc_fin.sort_values(by='premiere_date')

df_sc_fin = df_sc_fin[['title_name', 'season_number', 'imdb_id', 'premiere_date', 'forecast_category', 'rfv7', 'medal', 'num_episodes','first_release_date', 'last_release_date', 'bo_opening_imputed','production_cost','prev_title', 'prev_title_fv', 'gap', 'days_diff', 'fv_pre_7_pay1', 'fv_pre_7_series', 'fv_pre_7_retail','first_release_date_2', 'last_release_date_2', 'prev_title_2', 'gap_2',  'days_diff_2', 'fv_pre_7_pay1_2','fv_pre_7_series_2', 'fv_pre_7_retail_2', 'delta_days_diff', 'delta_gap',   'delta_fv_pre_7_pay1']]

# df_sc_fin[df_sc_fin.title_name.str.contains('Godzilla')]
display(df_sc_fin[(df_sc_fin.medal=='Platinum') | (df_sc_fin.forecast_category=='wb pay1')])


# COMMAND ----------

coef = df_coef[(df_coef.model=='pay_ogmdpc') & (df_coef.region=='NORTH AMERICA')].copy()
list_val = ['days_diff','fv_pre_7_pay1','gap']
coeff_val={}
for i in list_val:
    coeff_val[i] = coef[coef.feature=='ln_'+i].at[coef[coef.feature=='ln_'+i].index[0],'coef']

df_test = df_sc_fin[((df_sc_fin.forecast_category=='wb pay1')) |
                    (df_sc_fin.medal=='Platinum')]
for i in list_val:
    df_test['fv_change_{}'.format(i)]= df_test[i] ** coeff_val[i]/df_test[i+'_2']**coeff_val[i]-1


df_test[['fv_pre_7_pay1','fv_pre_7_pay1_2']] = df_test[['fv_pre_7_pay1','fv_pre_7_pay1_2']].astype(int)
df_test.first_release_date =  pd.to_datetime(df_test.first_release_date).dt.date    
df_test.first_release_date_2 =  pd.to_datetime(df_test.first_release_date_2).dt.date       
display(df_test[['title_name','first_release_date','gap','days_diff','fv_pre_7_pay1']])
display(df_test[['title_name','first_release_date_2','gap_2','days_diff_2','fv_pre_7_pay1_2']])
coef

# COMMAND ----------

df_test.head()

# COMMAND ----------

coeff_days_diff = coef[coef.feature=='ln_days_diff'].at[coef[coef.feature=='ln_days_diff'].index[0],'coef']

a=df_sc_fin[df_sc_fin.title_name.str.contains('Godzilla')]
a['fv_ratio_days_diff']= (a.days_diff**coeff_days_diff)
a


# COMMAND ----------

a=df_sc_fin[df_sc_fin.title_name.str.contains('Godzilla')]
a['fv_ratio_days_diff']= (a.days_diff**coeff_days_diff)
a

# COMMAND ----------

 # future 
query ="""
select * 
from bolt_cus_dev.gold.delphi_title_metadata_future 
where premiere_date_us >='2024-04-01' 
and premiere_date_us <='2024-09-01' 
and geo_value='NORTH AMERICA'
and home_region_predicted_medal in ('Platinum', 'Gold')
order by first_release_date
"""

df_raw = spark.sql(query)
df_raw = df_raw.toPandas()
df_raw

# COMMAND ----------

# future 
query ="""
select * 
from bolt_cus_dev.gold.delphi_title_metadata_future 
where premiere_date_us >='2024-04-01' 
and premiere_date_us <='2024-09-01' 
and geo_value='NORTH AMERICA'
and home_region_predicted_medal in ('Platinum')
order by first_release_date"""

df_raw = spark.sql(query)
df_raw = df_raw.toPandas()

# COMMAND ----------

make dataframe for scenario 1
make df for sc2 w/ change in days_diff and .. 


# COMMAND ----------



# COMMAND ----------

df_s.premiere_date = pd.to_datetime(df_s.premiere_date)
df_sc1 = df_s[(df_s.medal.isin(['Gold','Platinum']))
               & (df_s.premiere_date>='2024-04-29')]
df_sc1               

# COMMAND ----------



# COMMAND ----------

mid-July premiere of GODZILLA (falling mid-season HOTD S2, and about a month or so before FURIOSA’s very tentative August Pay-1 start) vs. a mid/late June premiere (right on top of the HOTD S2 debut) could be more impactful) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Series

# COMMAND ----------

df_in.ref_popularity.isnull().sum()

# COMMAND ----------

medal = ['Gold','Platinum']
forecast_category = ['series']
features = ['search_2wbp', 'wiki_2wbp', 'num_episodes', 'medal', 'fv_pre_7','days_diff', 'prev_title', 'prev_title_fv','ref_count', 'ref_popularity']

target = 'rfv7'
model_list = {
    '2wbp_d':'ln_days_diff',
    '2wbp_dm':'ln_days_diff + medal',
    '2wbp_dmp':'ln_days_diff + medal  + ln_fv_pre_7',
    '2wbp_dmp':'ln_days_diff + medal  + ln_ref_popularity',
    '2wbp_dms':'ln_days_diff + medal  +  ln_search_2wbp',
    '2wbp_dmnf':'ln_days_diff + medal  + ln_prev_title_fv',
}

model_list = {
    '2wbp_m':'medal',
    '2wbp_ms':'medal + ln_search_2wbp',
    '2wbp_msd':'medal  +  ln_search_2wbp + ln_days_diff', 
    '2wbp_mr':'medal + ln_ref_count',
    '2wbp_mrd':'medal + ln_ref_count + ln_days_diff',
    '2wbp_mp':'medal + ln_ref_popularity',
    '2wbp_mpd':'medal + ln_ref_popularity + ln_days_diff',
}

df_coef=[]
df_summary =[]

df_in = get_df(df_s, medal, forecast_category, col_id, features, target)
# df_in['days_diff_1'] = np.where(df_in['days_diff_1']>=30, 30, df_in['days_diff_1'])
df_in = transform_features(df_in, ['days_diff'], features, target)
df_in = df_in.fillna(0)

for i in model_list:
    model, df_in_ = train(df_in, target, model_list[i] )
    df_coef.append(get_coef(model, i, target))
    df_summary.append(get_summary(df_in_, model, i, model_list[i], target))

df_summary = pd.concat(df_summary)
df_coef = pd.concat(df_coef).sort_values(by=['model','feature'])



# COMMAND ----------

display(df_summary)

# COMMAND ----------

df_coef.sort_values(by='feature')

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA

# COMMAND ----------

df_s[df_s.forecast_category=='wb pay1'].gap.describe()
# df_s[df_s.forecast_category=='popcorn'].gap.describe()

# COMMAND ----------

df_s.groupby(by=['forecast_category','medal']).count()

# COMMAND ----------

df_in.premiere_date = pd.to_datetime(df_in.premiere_date)
df_in[df_in.premiere_date>='2022-01-01'].shape

# COMMAND ----------

df_in[df_in.forecast_category=='series']

# COMMAND ----------

df_in.describe()

# COMMAND ----------

corr_matrix = df_in.query('forecast_category=="wb pay1" & region=="NORTH AMERICA"')[['rfv7','days_diff', 'fv_pre_7_pay1', 'gap','bo_opening_imputed','production_cost']].corr()

fig = px.imshow(corr_matrix,
                text_auto=True, # This will automatically add the correlation values as text on the cells
                labels=dict(x="Variable", y="Variable", color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                title="Correlation Matrix Heatmap")
fig.show()

# COMMAND ----------

corr_matrix = df_s[df_s.forecast_category=='wb pay1'][['bo_opening_imputed', 'gap', 'production_cost', 'medal', 'fv_pre_7','days_diff', 'prev_title', 'prev_title_fv']].corr()

fig = px.imshow(corr_matrix,
                text_auto=True, # This will automatically add the correlation values as text on the cells
                labels=dict(x="Variable", y="Variable", color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                title="Correlation Matrix Heatmap")
fig.show()

# COMMAND ----------

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

df_vif = df_s[df_s.forecast_category=='wb pay1'][['bo_opening_imputed','production_cost', 'gap', 'fv_pre_7','days_diff',  'prev_title_fv','ref_popularity']]
vif_data = pd.DataFrame()
vif_data["feature"] = df_vif.columns
vif_data["VIF"] = [variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])]

print(vif_data)

# COMMAND ----------

# df_vif = df_vif.fillna(0)
# if np.any(np.isinf(df_vif)) or np.any(np.isnan(df_vif)):
#     # Handle or remove infinite or NaN values
#     # For example, you can replace them with a suitable value or drop rows/columns containing them
#     df_vif = df_vif.replace([np.inf, -np.inf], np.nan).fillna(0)

display(df_in.groupby(by=['medal']).agg({'days_diff':'mean', 
                                 'num_episodes':'mean',
                                 'search_2wbp':'mean'}))
display(df_in.groupby(by=['medal']).count())

# COMMAND ----------



# COMMAND ----------

df_in.premiere_date = pd.to_datetime(df_in.premiere_date)
fig = px.scatter(df_in[(df_in.forecast_category=='wb pay1') & (df_in.premiere_date>='2021-01-01')], x='days_diff', y='rfv7', title='fv_7 vs. days_diff pay1',
                 hover_data=['title_name', 'medal','premiere_date','prev_title'], color='bo_opening_imputed')  

fig.show()

# COMMAND ----------



# COMMAND ----------

fig = px.scatter(df_in[(df_in.medal=='Gold') & (df_in.premiere_date>='2021-01-01')], x='ln_wiki_2wbp', y='rfv7', title='fv_7 vs. wii Gold',
                 hover_data=['title_name', 'medal','premiere_date','prev_title'], color='ln_days_diff')  

fig.show()

# COMMAND ----------

fig = px.scatter(df_in[(df_in.medal=='Gold') & (df_in.premiere_date>='2021-01-01')], x='days_diff', y='rfv7', title='fv_7 vs. days_diff Gold',
                 hover_data=['title_name', 'medal','premiere_date','prev_title'], color='ln_wiki_2wbp')  

fig.show()

# COMMAND ----------

fig = px.scatter(df_in[(df_in.medal=='Gold') & (df_in.premiere_date>='2021-01-01')], x='days_diff', y='rfv7', title='fv_7 vs. days_diff Gold',
                 hover_data=['title_name', 'medal','premiere_date','prev_title'], color='ln_wiki_2wbp')  

fig.show()

# COMMAND ----------

fig = px.scatter(df_in[(df_in.medal=='Platinum') & (df_in.premiere_date>='2021-01-01')], x='days_diff', y='rfv7', title='fv_7 vs. days_diff',labels={'Income': 'Annual Income (USD)', 'Age': 'Age (Years)'},
                 hover_data=['title_name', 'medal','premiere_date'], color='ln_search_2wbp')  

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Correlation

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# pip install category_encoders
import xgboost as xgb
from category_encoders import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder

class XGB():
    def __init__(self, df_in, mode, target_var, features_cat=None, features_cont=None):
        self.df_in = df_in    
        self.mode = mode
        self.target = target_var
        # self.date_run = date_run
        # self.territory = territory
        self.features_cat = features_cat
        self.features_cont = features_cont
        self.get_parameters()
        
        if mode=='train':
            self.train_xgb()
            # self.save_model()
        # elif mode=='predict':
        #     self.get_model()
        #     self.predict_xgb()
        #     self.return_df()
            
    @staticmethod
    def _encode(df, categoricals):
        
        """
        perform category encoding on the data
        :param df: dataframe to be encoded
        :param categoricals: list of name of categorical columns
        :return ohe, x_ohe: OHE object and OHE-encoded data
        """
        ohe = OneHotEncoder(cols=categoricals, 
                            handle_unknown='return_nan',
                           handle_missing='return_nan',  
                           use_cat_names=True) 
        x_ohe = ohe.fit_transform(df)
        return ohe, x_ohe
    
    def return_df(self):
        return self.df_pred
    
    def get_parameters(self):
        # if self.features_cat == None:
        #     print('No Cat Features specified, using default')
        #     self.features_cat = ['medal', 'derived_genre', 'program_type', 'content_source', 'country_of_origin', 'lop_flag', 'lop_origin_flag']
        # if self.features_cont == None:
        #     print('No Cont Features specified, using default')
        #     self.features_cont= ['count_refs','count_pop_refs', 'sum_1year_talent_page_views_per_territory', 'sum_1year_page_views_per_territory', 'cost'] 
            
        self.param_xgb = {"booster":"gbtree",
                    "gamma":1,
                    "objective": 'reg:squarederror',
                    "eval_metric": 'mae'}

    def train_xgb(self):
        x_train = self.df_in[self.features_cat + self.features_cont]
        y_train = self.df_in[self.target] 
        self.ohe, x_ohe = self._encode(x_train, self.features_cat)
        dm_train = xgb.DMatrix(x_ohe, label=y_train)

        ## train 
        self.model = xgb.train(params = self.param_xgb, dtrain = dm_train, num_boost_round = 30)
        
    # def save_model(self):
    #     dict_model = {'model': self.model, 'ohe': self.ohe}
    #     Utils.to_pkl_s3(dict_model, input_bucket, key_path, f'models/{self.target}_{self.territory}_{self.date_run}.pkl')
    #     logger.info(f'Done model training {self.date_run}')
    
    # def get_model(self):
    #     dict_model = Utils.read_pkl_s3(input_bucket, key_path, f'models/{self.target}_{self.territory}_{self.date_run}.pkl')
    #     self.ohe = dict_model['ohe']
    #     self.model = dict_model['model']
        
    # def predict_xgb(self):
    #     x_test = self.df_in[self.features_cat + self.features_cont]
    #     x_ohe_test = self.ohe.transform(x_test)
    #     dm_test = xgb.DMatrix(x_ohe_test)
    #     pred = self.model.predict(dm_test)

    #     self.df_pred = self.df_in[list(set(key_merge + self.features_cat + self.features_cont))]
    #     self.df_pred[self.target + '_log_pred'] = pred
    #     self.df_pred[self.target + '_pred'] = np.exp(self.df_pred[self.target + '_log_pred'])
    #     self.df_pred['pred_date'] = self.date_run

        features_cat = ['medal']
    features_cont= ['ln_search_2wbp','ln_wiki_2wbp', 'ln_num_episodes', 'ln_days_diff']
    target = ['ln_rfv7']
    #features_embed = [col for col in df_train_out.columns if 'logline_dim' in col]
    float_dict = {value: "float" for value in features_cont}
    
    # training and exporting model
    XGB(df_in, 'train', target, features_cat, features_cont)

# COMMAND ----------

import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Assuming you have your data in X and y
# X should be your features and y your target variable
df_encoded = pd.get_dummies(df_in, columns=['medal'])

# Define XGBoost model parameters
params = {
    'objective': 'reg:squarederror',  # For regression tasks
    'eval_metric': 'rmse',             # Evaluation metric to be used
    'eta': 0.1,                        # Learning rate
    'max_depth': 6,                    # Maximum depth of tree
    'subsample': 0.8,                  # Subsample ratio of the training instances
    'colsample_bytree': 0.8,           # Subsample ratio of columns when constructing each tree
    'seed': 42                         # Random seed for reproducibility
}

# Initialize an XGBoost regressor
regressor = xgb.XGBRegressor(**params)

# Define evaluation metrics
scoring = {
    'mae': make_scorer(mean_absolute_error),
    'mse': make_scorer(mean_squared_error),
    'r2': make_scorer(r2_score)
}

# Perform 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
results = cross_val_score(regressor, df_encoded[['search_2wbp', 'days_diff','medal_Gold', 'medal_Platinum']], df_encoded[['rfv7']], cv=kf, scoring=scoring)

# Print the evaluation metrics for each fold
for fold, scores in enumerate(results, 1):
    print(f"Fold {fold} - MAE: {scores['test_mae']:.4f}, MSE: {scores['test_mse']:.4f}, R2: {scores['test_r2']:.4f}")

# Calculate the mean and standard deviation of the evaluation metrics across all folds
mean_mae = np.mean([score['test_mae'] for score in results])
mean_mse = np.mean([score['test_mse'] for score in results])
mean_r2 = np.mean([score['test_r2'] for score in results])
std_mae = np.std([score['test_mae'] for score in results])
std_mse = np.std([score['test_mse'] for score in results])
std_r2 = np.std([score['test_r2'] for score in results])

print(f"\nMean MAE: {mean_mae:.4f} ± {std_mae:.4f}")
print(f"Mean MSE: {mean_mse:.4f} ± {std_mse:.4f}")
print(f"Mean R2: {mean_r2:.4f} ± {std_r2:.4f}")


