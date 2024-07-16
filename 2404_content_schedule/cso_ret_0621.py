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

## engagement model  (break out to tenure level later)
## tenure level churn curve 
## tenure level median calculation 


# Title level sensitivity (week x eligible x title-tenure)
# Total slate (week x eligible x sum(title-tenure)) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Engagement model

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Get data & clean 

# COMMAND ----------

## Pull raw data & clean 
query = """
select 
f.*,
d.first_release_date,
d.last_release_date,
d.content_category,
d.primary_genre,
d.predicted_medal_us,
d.predicted_medal_latam,
d.predicted_medal_emea,
d.predicted_medal
from bolt_cus_dev.silver.forecasting_feature_space_test f
left join bolt_cus_dev.gold.delphi_title_metadata d
    on f.imdb_id=d.imdb_series_id 
    and f.title_name = d.title_series
    and f.season_number= d.season_number
    and f.region = d.geo_value 
where f.premiere_date>='2021-01-01'
    and f.premiere_date<'2024-04-01'
    and d.geo_level='REGION'
"""

query_metric = """
select 
    imdb_series_id as imdb_id,
    title_series as title_name,
    season_number,
    offering_start_date as premiere_date,
    geo_value as region,
    days_on_max,
    percent_cumulative_viewing_subs as vs_pct
from bolt_cus_dev.gold.delphi_title_metrics_platform p
where geo_level='REGION'
and days_on_max in (7,28,60,90)
"""

## feature space data cleaning 
df_raw = spark.sql(query)
df_raw = df_raw.toPandas()
df_raw.columns=df_raw.columns.str.lower()
# drop duplicates 
df_raw = df_raw.sort_values(by=['region','premiere_date','title_name','season_number','tfv90','wiki_1wbp'])
df_raw = df_raw.drop_duplicates(subset=['region','premiere_date','title_name','season_number'], keep='last')
df_raw = df_raw.drop_duplicates(subset=['region','title_name','season_number'], keep='first') ## Keep first premiere date

##  metrics data cleaning 
df_m = spark.sql(query_metric)
df_m = df_m.toPandas()
df_m.columns=df_m.columns.str.lower()
# drop duplicates 
df_m = df_m.sort_values(by=['region','title_name','season_number', 'days_on_max','premiere_date','vs_pct'])
df_m = df_m.drop_duplicates(subset=['region','premiere_date','title_name','season_number','days_on_max'], keep='last')
df_m = df_m.drop_duplicates(subset=['region','title_name','season_number','days_on_max'], keep='first')

# create columns for viewingsub
df_m = df_m.pivot(index=['title_name','season_number','region'], columns="days_on_max", values="vs_pct")
df_m.columns = [f"vspct{col}" for col in df_m.columns]
df_m = df_m.reset_index()

## Merge 
df_raw = df_raw.merge(df_m, on=['title_name', 'season_number', 'region'], how='left')

## Clean date columns  
df_raw['premiere_date'] = pd.to_datetime(df_raw['premiere_date'])
df_raw['first_release_date'] = pd.to_datetime(df_raw['first_release_date'])
df_raw['last_release_date'] = pd.to_datetime(df_raw['last_release_date'])
df_raw.loc[(df_raw.content_category=='movie') & (df_raw.region=='EMEA'), 'last_release_date'] = df_raw['first_release_date'] + pd.Timedelta(days=1)
df_raw.gap = np.where(df_raw.forecast_category=='popcorn', 0, df_raw.gap)

## Replace medal column with predicted medal 
df_raw['medal'] = df_raw['predicted_medal']

df_raw = df_raw.sort_values(by=['region','first_release_date','last_release_date']) 
df_raw.head()



# COMMAND ----------

## Get features 

def get_count_medal(df_in, nday):
    df_in['first_release_date_min'] = df_in['first_release_date'] - pd.DateOffset(days=nday) 
    df_in['count_platinum'] = None
    for index, row in df_in.iterrows(): 
        for filter in [['Platinum']]: 
            eligible_rows = df_in[((df_in['medal'].isin(filter)) 
                & (df_in['first_release_date'] <= row['first_release_date'])
                & (df_in['first_release_date'] >= row['first_release_date_min']))]
            if not eligible_rows.empty: 
                count_platinum = (eligible_rows['medal'] == 'Platinum').sum()
                df_in.at[index, 'count_platinum'] = count_platinum
            else:
                pass
    df_in = df_in.fillna(0)
    df_in.loc[df_in.medal=='Platinum','count_platinum'] = df_in['count_platinum'] - 1 ##removing count of itself 
    return df_in

def get_days_diff(df_in_all, regions=['NORTH AMERICA','LATAM','EMEA']):
    df_in_all['first_release_date_ref'] = df_in_all['first_release_date'] - pd.DateOffset(days=60) 
    # df_in['last_release_date_ref'] = df_in['first_release_date'] + pd.DateOffset(days=60) #titles that ended before 28d post premiere
    df_in_all['days_diff'] = None
    df_s=[]
    for r in regions:
        df_in = df_in_all[df_in_all.region==r]
        for index, row in df_in.iterrows(): 
            for filter in [['Platinum']]: 
                ### Prev_title heuristics in the order of priority: 
                ### 1. Platinum titles that last aired within past 60d ago.
                ### 2. The most recent Gold or Platinum, released before current title.     
                ## days_diff = Difference between the last date of previous title and the release date of current title. If previous title ends after current, days_diff would be negative.                  
                eligible_rows = df_in[((df_in['medal'].isin(filter)) 
                    & (df_in['first_release_date'] < row['first_release_date']))]

                if not eligible_rows.empty: 
                    eligible_rows['days_diff_last'] =  (row['first_release_date']- eligible_rows['last_release_date']).abs() 
                    eligible_rows['days_diff_first'] = (row['first_release_date'] - eligible_rows['first_release_date']).abs()  
                    eligible_rows['days_diff_min'] = eligible_rows[['days_diff_last', 'days_diff_first']].min(axis=1)
                    closest_diff = eligible_rows.days_diff_min.min()
                    latest_date = eligible_rows.loc[eligible_rows['days_diff_min'] == closest_diff, 'first_release_date'].iloc[0]
                    # latest_date = eligible_rows['first_release_date'].max()
                    latest_title = eligible_rows.loc[eligible_rows['first_release_date'] == latest_date, 'title_name'].iloc[0]
                    latest_title_fv = eligible_rows.loc[eligible_rows['first_release_date'] == latest_date, 'rfv7'].iloc[0]
                    latest_last_date = eligible_rows.loc[eligible_rows['first_release_date'] == latest_date, 'last_release_date'].iloc[0]
                    df_in.at[index, 'days_diff'] = (row['first_release_date'] - latest_last_date).days
                    # df_in.at[index, 'days_diff'] = (row['first_release_date'] - latest_date).days
                    df_in.at[index, 'prev_title'] = latest_title
                    df_in.at[index, 'prev_title_fv'] = latest_title_fv
                else:
                    pass
        df_in = df_in.fillna(0)
        df_s.append(df_in)
    df_s = pd.concat(df_s)
    df_s.days_diff = df_s.days_diff.astype('int')

    return df_s

df_s=[]
for i in ['NORTH AMERICA','LATAM','EMEA']:
    df_r = get_count_medal(df_raw[df_raw.region==i], 30)
    df_r = get_days_diff(df_r)
    df_s.append(df_r)
    display(df_r[['region','title_name','medal','first_release_date','last_release_date','first_release_date_ref','days_diff', 'prev_title','count_platinum']].sort_values(by=['first_release_date']))

df_s = pd.concat(df_s)

## Change count_platinum to binary 
df_s['count_platinum_binary'] = 0
df_s.loc[df_s['count_platinum']>=1,'count_platinum_binary'] = 1 


# COMMAND ----------

## Pay 1 model training  
def transform_features(df_in, days_diff, features, target):
    df_in.loc[df_in.days_diff<0, 'days_diff'] = 0
    for d in days_diff:
        min_days_diff = 0  
        df_in[d] = df_in[d] + (-min_days_diff)
    df_in = log_features(df_in, features, target)

    # for d in days_diff:
    #     df_in[d] = df_in[d] + min_days_diff -1
    return df_in


# Define model 
target = 'vspct28'
medal = ['Gold','Platinum']
forecast_category = ['wb pay1']
region = ['NORTH AMERICA','LATAM','EMEA']
col_id = ['imdb_id','title_name','season_number','premiere_date','region','forecast_category','future_title']
features = ['bo_opening_imputed', 'gap', 'production_cost', 'medal', 'fv_pre_7','days_diff', 'prev_title', 'prev_title_fv',  'fv_pre_7_pay1','fv_pre_7_series','count_platinum_binary']


# Impute features 
df_s.loc[df_s.production_cost==0, 'production_cost'] = df_s['production_cost'].median()
df_s.loc[(df_s.title_name=='Fantastic Beasts: The Secrets of Dumbledore') & (df_s.region=='NORTH AMERICA'), 'medal'] = 'Platinum'
df_s.loc[(df_s.title_name=='The Batman') & (df_s.region=='NORTH AMERICA'), 'medal'] = 'Platinum'

df_s.loc[(df_s.title_name=='House Party') & (df_s.region=='LATAM'), 'gap'] = 50
df_s.loc[(df_s.title_name=="Magic Mike's Last Dance") & (df_s.region=='LATAM'), 'gap'] = 112
df_s.loc[(df_s.title_name=="Magic Mike's Last Dance") & (df_s.region=='LATAM'), 'bo_opening_imputed'] = 5556243.301106308 #removing negative

# df_in.loc[df_in.production_cost==0, 'production_cost'] = df_in['production_cost'].median()
# df_in.loc[(df_in.title_name=='The Batman') & (df_in.region=='NORTH AMERICA'), 'medal'] = 'Platinum'
# df_in.loc[(df_in.title_name=='House Party') & (df_in.region=='LATAM'), 'gap'] = 50
# df_in.loc[(df_in.title_name=="Magic Mike's Last Dance") & (df_in.region=='LATAM'), 'gap'] = 112
# df_in.loc[(df_in.title_name=="Magic Mike's Last Dance") & (df_in.region=='LATAM'), 'bo_opening_imputed'] = 5556243.301106308 #removing negative

# Clean data 
df_in = get_df(df_s[df_s.premiere_date<='2024-05-22'], medal, forecast_category, col_id, features, target)

# Transform features 
df_in = transform_features(df_in, ['days_diff'], features, target)
print(df_in.premiere_date.min())

# Remove outliers; 
# Batman: 150K fv; Magic Mike: 3K fv; House Party: ~0 , Shazam throws off pre-7d feature in EMEA
title_removed = ['The Color Purple', "Magic Mike's Last Dance"]#'The Batman',"Magic Mike's Last Dance"]#'The Batman','House Party', "Magic Mike's Last Dance"
title_removed_LATAM = ["Magic Mike's Last Dance"]#'House Party','The Batman','The Color Purple',"Magic Mike's Last Dance"]#,  "Magic Mike's Last Dance",'DC League of Super-Pets']#,'The Batman','Barbie']#, 'Blue Beetle']
title_removed_EMEA = ["Magic Mike's Last Dance"]#'House Party','The Batman','The Color Purple']#, "Shazam! Fury of the Gods"]#,  'DC League of Super-Pets','Evil Dead Rise', ]
df_in = df_in[((df_in.region=='NORTH AMERICA') & ~(df_in.title_name.isin(title_removed))) |
              ((df_in.region=='EMEA') & ~(df_in.title_name.isin(title_removed_EMEA))) | 
              ((df_in.region=='LATAM') & ~(df_in.title_name.isin(title_removed_LATAM)))] 

# COMMAND ----------

## Visualize input data 
display(df_in.sort_values(by=['region','premiere_date']))

import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from itertools import combinations

variables = list(df_in[[f'ln_{target}','count_platinum_binary','ln_days_diff','ln_gap','ln_bo_opening_imputed','ln_production_cost']])
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
    annot['font'] = dict(size=8)  # Customize font size and color here

fig.update_layout(height=1000, width=1000, title_font=dict(size=10), font=dict(size=9))
fig.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### train

# COMMAND ----------

## Define model and train 

model_list = {
    'pay_am': 'ln_bo_opening_imputed + medal',
    'pay_amg': 'ln_bo_opening_imputed + medal + ln_gap',
    'pay_amgd': 'ln_bo_opening_imputed + medal + ln_gap  + ln_days_diff',
    'pay_amgp': 'ln_bo_opening_imputed + medal + ln_gap + count_platinum_binary',
    'pay_amgpd': 'ln_bo_opening_imputed + medal + ln_gap + count_platinum_binary+ ln_days_diff',
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


## Run mixed model 
import statsmodels.formula.api as smf
for i in model_list:
    model = smf.mixedlm(f"ln_{target} ~" + model_list[i], df_in, groups=df_in["region"], re_formula="~1")
    result = model.fit()
    display(result.summary())
    # Extract fixed effects
    fixed_effects = result.fe_params
    fixed_effects_df = pd.DataFrame(fixed_effects, columns=["Coefficient"])
    fixed_effects_df["StdErr"] = result.bse
    fixed_effects_df["PValue"] = result.pvalues

    display("Fixed Effects:\n", fixed_effects_df)


# COMMAND ----------

display(df_summary)

# COMMAND ----------

df_coef[(df_coef.region=='NORTH AMERICA')] #new days_diff
# display(df_coef[(df_coef.region=='LATAM') & (df_coef.model=='pay_ogmdpc')]) 
# display(df_coef[(df_coef.region=='EMEA') & (df_coef.model=='pay_ogmdp')]) 

# COMMAND ----------

df_coef[(df_coef.region=='LATAM')]

# COMMAND ----------

df_coef[(df_coef.region=='EMEA')]

# COMMAND ----------

# display(df_coef[(df_coef.region=='NORTH AMERICA') & (df_coef.model=='pay_ogmdpc')]) #new days_diff
# display(df_coef[(df_coef.region=='LATAM') & (df_coef.model=='pay_ogmdpc')]) 
# display(df_coef[(df_coef.region=='EMEA') & (df_coef.model=='pay_ogmdp')]) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Slate level sensitivity

# COMMAND ----------

query = """
select 
f.region,
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

"""
    # and f.region='NORTH AMERICA'
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
df_ss.loc[df_ss.title_name=='Beetlejuice Beetlejuice','gap'] = 74
df_ss['gap']= df_ss['gap'].fillna(77)  
df_ss.loc[df_ss.gap==0, 'gap'] = 77                   

# COMMAND ----------

df_ss

# COMMAND ----------


df_fv = pd.read_sql("""
select 
    date, 
    region,
    sum(retail_fv) as fv_retail, 
    sum(retail_fv_series) as fv_series, 
    sum(retail_fv_pay1) as fv_pay1
from forecasting_fv_pacing_category a
left join forecasting_dates using (date)
where 
year(date) = 2024
group by 1, 2 order by 1, 2""", con)

df_fv_title = pd.read_sql("""
select 
    date, 
    region,
    retail_fv as fv_title
from max_dev.workspace.forecasting_fv_pacing a
where 
year(date) = 2024
-- and imdb_id='tt14539740' --gxk
and imdb_id = 'tt2049403' --Beetlejuice2
""", con) 

df_fv_tfr = pd.read_sql("""
select 
region,
date,
sum(RETAIL_FV) as sum_fv_tfr
from max_dev.workspace.forecasting_fv_pacing a
where 
year(date) = 2024
and title_name_clean='the front room'
group by ALL
order by date
""", con) 


def get_rolling_sum(df_fv_in_all, columns, regions=['NORTH AMERICA','EMEA','LATAM']):
    df_fv_in_all.columns=df_fv_in_all.columns.str.lower()
    df_fv_in_all['date'] = pd.to_datetime(df_fv_in_all['date'])

    df_list=[]
    for r in regions:
        df_fv_in = df_fv_in_all[df_fv_in_all.region==r]
        df_fv_in.set_index('date', inplace=True)
        rolling_sums = {}
        for col in columns:
            rolling_sums[col] = df_fv_in[col].rolling(window=7).sum()
        df_out = pd.DataFrame(rolling_sums).reset_index()
        df_out['region'] = r
        df_list.append(df_out)
    df_out = pd.concat(df_list)
    return df_out

def clean_fv_data(df_in):
    df_in = df_in.reset_index(drop=True)
    df_in.columns = df_in.columns.str.lower()
    df_in.date = pd.to_datetime(df_in.date)
    df_in = df_in.fillna(0)
    return df_in


#Clean data
df_fv = clean_fv_data(df_fv)
df_fv_title = clean_fv_data(df_fv_title)

# ## Correct for watcher date (to Jul 30)
# df_fv_title_watcher_new = df_fv_title_watcher.copy()
# df_fv_title_watcher_new['date'] = df_fv_title_watcher_new['date'] + pd.to_timedelta(7, unit='D')
# df_fv = df_fv.merge(df_fv_title_watcher, on=['date','region'],how='left').merge(df_fv_title_watcher_new, on=['date','region'],how='left', suffixes=('','_new'))
# df_fv = df_fv.fillna(0)
# df_fv['fv_pay1'] = df_fv['fv_pay1']-df_fv['fv_title_watcher'] + df_fv['fv_title_watcher_new']

# ## subtract The Front Room (A24pay11)
df_fv_tfr.columns = df_fv_tfr.columns.str.lower()
df_fv_tfr.date = pd.to_datetime(df_fv_tfr.date)

df_fv = df_fv.merge(df_fv_tfr, on=['date','region'],how='left')
df_fv = df_fv.fillna(0)
df_fv['fv_pay1'] = df_fv['fv_pay1']-df_fv['sum_fv_tfr'] 


## Merge title data
df_fv = df_fv.merge(df_fv_title, on=['date','region'],how='left')
df_fv = df_fv.fillna(0)

## Build scheduling scenarios 
title = 'Beetlejuice Beetlejuice'
delta_days_list= [0,7,14,21]
list_df=[]


for delta_days in delta_days_list:
    ## Get scenario premiere data
    df_ss_ = df_ss.copy()
    
    df_ss_.loc[df_ss_['title_name']==title, 'premiere_date'] = df_ss_['premiere_date'] + pd.to_timedelta(delta_days, unit='D')
    df_ss_.loc[df_ss_['title_name']==title, 'first_release_date'] = df_ss_['first_release_date'] + pd.to_timedelta(delta_days, unit='D')
    df_ss_.loc[df_ss_['title_name']==title, 'last_release_date'] = df_ss_['last_release_date'] + pd.to_timedelta(delta_days, unit='D')
    df_ss_.loc[df_ss_['title_name']==title, 'gap'] = df_ss_['gap'] + delta_days
    df_ss_ = get_days_diff(df_ss_)

    ## get pre_7d features 
    df_fv_title_new=  df_fv_title.copy()
    df_fv_title_new['date'] = df_fv_title_new['date'] + pd.to_timedelta(delta_days, unit='D')
    df_fv_ = df_fv.merge(df_fv_title_new, on=['region','date'],how='left', suffixes=('','_{}'.format(delta_days)))
    df_fv_ = df_fv_.fillna(0)
    df_fv_['fv_pay1_{}'.format(delta_days)] = df_fv_['fv_pay1'] - df_fv_['fv_title'] + df_fv_['fv_title_{}'.format(delta_days)]
    df_fv_['fv_retail_{}'.format(delta_days)] = df_fv_['fv_retail']  - df_fv_['fv_title'] + df_fv_['fv_title_{}'.format(delta_days)]
    df_fv_rs = get_rolling_sum(df_fv_, ['fv_retail','fv_series','fv_pay1']+['fv_pay1_{}'.format(delta_days)])
    df_fv_rs = df_fv_rs.rename(columns={'fv_pay1_{}'.format(delta_days):'fv_pre_7_pay1'})
    df_fv_rs['date'] =  df_fv_rs['date'] + pd.to_timedelta(1, unit='D')

    ## merge two fv_pay fv_pay1_7
    df_sc_fin = df_ss_.merge(df_fv_rs[['region','date','fv_pay1',	'fv_pre_7_pay1']], left_on =['region','premiere_date'], right_on=['region','date'], how='left')
    df_sc_fin = df_sc_fin.sort_values(by=['region','premiere_date'])
    df_sc_fin['fv_pre_7_pay1'] = df_sc_fin['fv_pre_7_pay1'].astype(int)

    ## Get platinum count 
    df_sc_fin = get_count_medal(df_sc_fin, 30)
    df_sc_fin = get_days_diff(df_sc_fin)
    df_sc_fin['count_platinum_binary'] = 0
    df_sc_fin.loc[df_sc_fin['count_platinum']>=1,'count_platinum_binary'] = 1 


    df_result=df_sc_fin[((df_sc_fin.forecast_category=='wb pay1')) |
                        (df_sc_fin.medal=='Platinum')][['region','title_name','medal','first_release_date','last_release_date','gap','days_diff','fv_pre_7_pay1','prev_title','count_platinum','count_platinum_binary', 'bo_opening_imputed', 'forecast_category']]
    
    df_result['delta_days'] = delta_days
    list_df.append(df_result)

df_result = pd.concat(list_df)


# COMMAND ----------

# def calc_change(df_in_all, target, delta_days, list_features, df_coef, model_dict, title, region_list=['NORTH AMERICA']):#,'EMEA','LATAM']):
#     df_1 = df_in_all[df_in_all.delta_days==0]
#     df_2 = df_in_all[df_in_all.delta_days==delta_days]
#     df_comp = df_1.merge(df_2, on=['title_name','region'],  suffixes=('','_2'))
#     df_comp[f'total_delta_{target}'] = 0
    
#     df_list=[]
#     for r in region_list:
#         df_comp_r = df_comp[df_comp.region==r]
#         coef = df_coef[(df_coef.model==model_dict[r]) & (df_coef.region==r)]
#         for i in list_features:
#             print(i)
#             try:
#                 coeff_val = coef.loc[coef.feature=='ln_'+i, 'coef'].values[0]
#             except:
#                 try:
#                     coeff_val = coef.loc[coef.feature==i, 'coef'].values[0]
#                 except:
#                     coeff_val = 0

#             df_comp_r['{}_coeff'.format(i)] = coeff_val
#             df_comp_r['{}_delta_{}'.format(i, target)]= np.exp(np.log(df_comp_r['{}_2'.format(i)])*coeff_val)/np.exp(np.log(df_comp_r['{}'.format(i)])*coeff_val)-1
#             df_comp_r = df_comp_r.fillna(0)
#             df_comp_r[f'total_delta_{target}'] = df_comp_r[f'total_delta_{target}']+df_comp_r['{}_delta_{}'.format(i,target)]
#         df_list.append(df_comp_r)
#     df_comp = pd.concat(df_list)
#     list_features_delta = ['region', 'first_release_date_2']+ [i + f'_delta_{target}' for i in list_features]
#     df_sum = df_comp[df_comp.title_name==title][list_features_delta]

#     return df_comp,df_sum


# COMMAND ----------

# df_result.head()

# df_f = get_df(df_result, medal, forecast_category, col_id, features, target)

# # Transform features 
# df_f = transform_features(df_result, ['days_diff'], features, target)
# print(df_in.premiere_date.min())

# print(df_f.columns)
# df_in.columns

# COMMAND ----------


# def calc_change(df_in_all, target, delta_days, list_features, df_coef, model_dict, title, region_list=['NORTH AMERICA']):#,'EMEA','LATAM']):
#     df_1 = df_in_all[df_in_all.delta_days==0]
#     df_2 = df_in_all[df_in_all.delta_days==delta_days]
#     df_comp = df_1.merge(df_2, on=['title_name','region'],  suffixes=('','_2'))
#     df_comp[f'total_delta_{target}'] = 0
    
#     coef = df_coef[(df_coef.model==model_dict[r]) & (df_coef.region==r)]
#     for i in list_features:
#         print(i)
#         try:
#             coeff_val = coef.loc[coef.feature=='ln_'+i, 'coef'].values[0]
#         except:
#             try:
#                 coeff_val = coef.loc[coef.feature==i, 'coef'].values[0]
#             except:
#                 coeff_val = 0

#         df_comp['{}_coeff'.format(i)] = coeff_val
#         df_comp['{}_delta_{}'.format(i, target)]= np.exp(np.log(df_comp['{}_2'.format(i)])*coeff_val)/np.exp(np.log(df_comp['{}'.format(i)])*coeff_val)-1
#         df_comp = df_comp.fillna(0)
#         df_comp[f'total_delta_{target}'] = df_comp[f'total_delta_{target}']+df_comp['{}_delta_{}'.format(i,target)]

#     list_features_delta = ['region', 'first_release_date_2']+ [i + f'_delta_{target}' for i in list_features]
#     df_sum = df_comp[df_comp.title_name==title][list_features_delta]

#     return df_comp,df_sum


# list_features = ['gap','days_diff','count_platinum_binary']
# model_dict={'NORTH AMERICA':'pay_amgpd',
#             'EMEA':'pay_amgd',
#             'LATAM':'pay_amgd'}

# # df__7, df__7_sum =calc_change(df_result, -7, list_features, df_coef, model_dict,'Beetlejuice Beetlejuice')

# target_var = 'vsr'

# df_0, df_0_sum =calc_change(df_result, target_var, 0, list_features, df_coef, model_dict,'Beetlejuice Beetlejuice')
# df_7, df_7_sum =calc_change(df_result, target_var, 7, list_features, df_coef, model_dict,'Beetlejuice Beetlejuice')
# df_14, df_14_sum =calc_change(df_result, target_var,14, list_features, df_coef, model_dict,'Beetlejuice Beetlejuice')
# df_21, df_21_sum =calc_change(df_result, target_var,21, list_features, df_coef, model_dict,'Beetlejuice Beetlejuice')

# df_final = pd.concat([df_7_sum, df_14_sum, df_21_sum], ignore_index=True)
# df_fin = df_final.T

# custom_order = ['NORTH AMERICA','EMEA','LATAM']  # Custom order values
# row_values = df_fin.iloc[0]
# custom_order_dict = {val: i for i, val in enumerate(custom_order)}
# ordered_columns = row_values.map(custom_order_dict).sort_values().index
# df_fin = df_fin[ordered_columns]

# df_fin = df_fin.reset_index()


# COMMAND ----------

## predict 
df_pred = transform_features(df_result, ['days_diff'], features, target)

predictions = result.predict(df_pred)
df_pred['ln_vspct28']=predictions
df_pred['vspct28'] = np.exp(df_pred['ln_vspct28'])
df_fin = df_pred.copy()

custom_order = ['NORTH AMERICA','EMEA','LATAM']  # Custom order values
row_values = df_fin.iloc[0]
custom_order_dict = {val: i for i, val in enumerate(custom_order)}
ordered_columns = row_values.map(custom_order_dict).sort_values().index
df_fin = df_fin[ordered_columns]

df_fin = df_fin.reset_index()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Eligibles

# COMMAND ----------

df_eligibles = pd.read_sql("""
    select 
    region,
    expire_date,
    tenure,
    sum(eligibles) as eligibles, 
    from MAX_DEV.WORKSPACE.FORECASTING_PROJ_CANCELS_FUTURE 
    where expire_date::date between '2024-11-22' and '2025-01-07'
    group by all 
    order by 1,2
    """, 
    con)

df_eligibles.columns=df_eligibles.columns.str.lower()
df_eligibles['tenure'] = df_eligibles['tenure'].astype(int)
mapping_dict = {1:'month_1', 2:'month_2', 3:'month_3', 4:'month_4_to_6', 5:'month_4_to_6', 6:'month_4_to_6', 7:'month_7_to_12', 8:'month_7_to_12', 9:'month_7_to_12', 10:'month_7_to_12', 11:'month_7_to_12', 12:'month_7_to_12'}
df_eligibles['tenure_bucket'] = df_eligibles['tenure'].map(mapping_dict).fillna('month_13+')

df_eligibles = df_eligibles.groupby(by=['region','expire_date','tenure_bucket']).sum().reset_index()
df_eligibles['expire_date'] = pd.to_datetime(df_eligibles['expire_date'] )
df_eligibles = df_eligibles[['region','expire_date','tenure_bucket','eligibles']]

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Engagement to churn conversion

# COMMAND ----------

def exponential_decay(x, a, b,c):
    return a * np.exp(b * x) + c
    
def exponential_decay_slope(x, a, b):
    return a * b*np.exp(b * x)

def fit_exponential(x_data, y_data, p0, param_bounds):
    x_fit = np.linspace(0, x_data.max(), 100)   
    params, _ = curve_fit(exponential_decay, np.array(x_data), y_data, p0, bounds=param_bounds)
    return x_fit, params

def get_simple_plot_multiple(df_plt, x, y, x_fit, y_fit, params, title=''):
    if title=='':
        
        title = f'{y} vs {x}'
       
    a_fit, b_fit, c_fit = params
    annotation_x_loc = 50
    annotation_y_loc = y_fit.min() +(y_fit.max()  - y_fit.min() )/2 
        
    fig = px.scatter(df_plt,
                  x=x, 
                  y=y, 
                  title=title,
                  width=500, height=400)
    fig.add_scatter( 
              x=x_fit, 
              y=y_fit)

    fig.update_layout(
        template='simple_white',
        showlegend=False,
        xaxis=dict(range=[0,50]),
        annotations=[
        dict(
            x=annotation_x_loc,  # x-coordinate for the text
            y=annotation_y_loc,  # y-coordinate for the text
            # text='y= {:.2f} * e^({:.2f} * title_viewed) + {:.2f}'.format(a_fit, b_fit, c_fit),  # the text to display
            showarrow=False,  # disable arrow for the annotation
            xanchor='right',
            font=dict(
                family='Arial',  # specify font family
                size=18,  # specify font size
                color='black'  # specify font color
            )
        )
    ]
) 
    fig.show()
    return 

def get_simple_plot_multiple_dot(df_plt, x, y, x_fit, y_fit, params, x_med, y_med, title=''):
    if title=='':
        
        title = f'{y} vs {x}'
       
    a_fit, b_fit, c_fit = params
    print('y= {:.2f} * e^({:.2f} * title_viewed) + {:.2f}'.format(a_fit, b_fit, c_fit))
    print('y= {:.3f} * e^({:.2f} * title_viewed)'.format(a_fit*b_fit,b_fit))
    annotation_x_loc = 50
    annotation_y_loc = y_fit.min() +(y_fit.max()  - y_fit.min() )/2 
        
    fig = px.scatter(df_plt,
                  x=x, 
                  y=y, 
                  title=title,
                  width=500, height=400)
    fig.add_scatter( 
              x=x_fit, 
              y=y_fit)
    
    fig.add_scatter( 
              x=x_med, 
              y=y_med,
                mode='markers',
            marker=dict(size=14, color='red', line=dict(color='black', width=2)))

    fig.update_layout(
        template='simple_white',
        showlegend=False,
        xaxis=dict(range=[0,15],
                   dtick=1),

) 
    fig.show()
    return fig



def get_churn_plot_simple(df_i, title, param_dic, x_med=0):
    df_i = df_i[df_i.is_cancel>=20]
#         display(df_i.tail(5))

    x_var = df_i.title_viewed_bin
    y_data = df_i.churn
    p0 = [0.5, -0.1, 0.01] 
    param_bounds = ([0, -0.8, 0.01], [np.inf, -0.1, np.inf])

    x_fit, params = fit_exponential(x_var, y_data, p0, param_bounds)
    a_fit, b_fit, c_fit = params
    y_fit = exponential_decay(x_fit, a_fit, b_fit, c_fit)
    
    if x_med==0:
        print(0)
        # fig = get_simple_plot_multiple(df_i, 'title_viewed_bin', 'churn', x_fit, y_fit, params, f'{title}')
    else:
        y_med = exponential_decay(x_med, a_fit, b_fit, c_fit)
        y_med_slope = exponential_decay_slope(x_med, a_fit, b_fit)
        print(x_med)
        print('average churn: ' + str('{:.3f}'.format(y_med)))
        print('slope: ' + str('{:.3f}'.format(y_med_slope*100))+'%')
        # fig = get_simple_plot_multiple_dot(df_i, 'title_viewed_bin', 'churn', x_fit, y_fit, params, x_med, np.array(y_med), f'{title}')
    # display(df_i.head())
    param_dic[title] = params
    return fig, params


def get_churn_slope_plot_simple(df_i, title, params, x_med=0):
    df_i = df_i[df_i.is_cancel>=20]
#         display(df_i.tail(5))

    x_var = df_i.hours_viewed_bin
    x_fit = np.linspace(0, x_var.max(), 100)   
    a_fit, b_fit, c_fit = params
    y_fit = exponential_decay_slope(x_fit, a_fit, b_fit)
    
    y_med = exponential_decay_slope(x_med, a_fit, b_fit)
    print(x_med)
    print(y_med)
    fig = get_simple_plot_dot(df_i, 'hours_viewed_bin', 'churn', x_fit, y_fit, params, x_med, np.array(y_med), f'{title}')
    display(df_i.head())
    param_dic['acquired'] = params
    return fig



def get_churn_bin(df_in, grpby, nbins = 100):
    df = df_in[df_in.monthly_hours_viewed<=60]
    df = df.groupby(by=['user_id','sub_month']+ grpby +['is_cancel']).sum().reset_index()
    df['title_viewed_bin'] = pd.qcut(df['monthly_title_viewed'], np.linspace(0,1,nbins), duplicates='drop')
    df['title_viewed_bin'] = df['title_viewed_bin'].apply(lambda x: (x.left+x.right)/2)
    df['title_viewed_bin'] = df['title_viewed_bin'].astype('float')
    df['churn'] = 1*df['is_cancel']  
    
    df_bin = df.groupby(['title_viewed_bin']+grpby).agg({'churn':'mean', 'user_id':'count',
                                                         'is_cancel':'sum','monthly_hours_viewed':'sum'}).reset_index()
    return(df_bin)

def get_churn_bin_mth1(df_in, grpby, nbins = 100):
    df = df_in[df_in.monthly_hours_viewed<=60]
    df = df.groupby(by=['user_id','sub_month']+ grpby +['is_cancel']).sum().reset_index()
    df['title_viewed_bin'] = df['monthly_title_viewed']
    # df['title_viewed_bin'] = df['title_viewed_bin'].apply(lambda x: (x.right))
    df['title_viewed_bin'] = df['title_viewed_bin'].astype('float')
    df['churn'] = 1*df['is_cancel']  
    
    df_bin = df.groupby(['title_viewed_bin']+grpby).agg({'churn':'mean', 'user_id':'count',
                                                         'is_cancel':'sum','monthly_hours_viewed':'sum'}).reset_index()
    return(df_bin)



# COMMAND ----------

# df_60_00 = pd.read_parquet('df_raw_tenure_churn_plt_0624.parquet', engine='pyarrow')

# ## Plot by tenure 
# param_dic={}
# med_dic = {} 

# for m in df_60_00.tenure_bucket.unique().tolist():
#     df_plt= df_60_00[df_60_00['tenure_bucket'] == m]
#     df_60_t = df_plt.groupby(by=['user_id','is_cancel_vol','sub_month'])[['monthly_title_viewed', 'monthly_hours_viewed']].sum().reset_index()
#     df_60_t['is_cancel'] = df_60_t['is_cancel_vol']
#     if m == 'month_1':
#         df_60_s = get_churn_bin_mth1(df_60_t, [])
#     else:
#         df_60_s = get_churn_bin(df_60_t, [])
#     med_x= df_60_t.monthly_title_viewed.median()
#     fig, params = get_churn_plot_simple(df_60_s[df_60_s['title_viewed_bin']<15], 
#                                         m, {}, np.array(med_x))
#     param_dic[m] = params
#     med_dic[m] = med_x


# df_param = pd.DataFrame.from_dict(param_dic, orient='index').reset_index()
# df_param.columns = ['tenure_bucket', 'a','b','c']
# df_median = pd.DataFrame.from_dict(med_dic, orient='index').reset_index()
# df_median.columns = ['tenure_bucket', 'median']

# df_param = df_param.merge(df_median, on=['tenure_bucket'])
# df_param

# df_param.to_csv('df_param.csv')
# df_median.to_csv('df_median.csv')

df_param = pd.read_csv('df_param.csv')
df_median = pd.read_csv('df_median.csv')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sensitivity w/ eligibles
# MAGIC
# MAGIC Per scenario: 
# MAGIC 1. Get viewing sub change, decay over 28d 
# MAGIC 2. For each date of viewing sub change over 28d, merge eligibles 
# MAGIC  

# COMMAND ----------

query_decay = """
select 
*
from bolt_cus_dev.bronze.cso_pct_vs_decay_pay1_platinum
"""

df_decay = spark.sql(query_decay)
df_decay = df_decay.toPandas()
df_decay.columns = df_decay.columns.str.lower()


# COMMAND ----------

df_fin_title

# COMMAND ----------

title = 'Beetlejuice Beetlejuice'
df_fin_title = df_fin[df_fin.title_name==title]
df_fin_title

df_vs_decay = pd.DataFrame()
for index, row in df_fin_title.iterrows():
    date_range = pd.date_range(start=row['first_release_date'], periods=28, freq='D')
    days_range = list(range(0,28))
    temp_df = pd.DataFrame({'region':row['region'],
                            'delta_days':row['delta_days'],
                            'title_name':row['title_name'],
                            'medal':row['medal'],
                            'first_release_date':row['first_release_date'],
                            'last_release_date':row['last_release_date'],
                            'gap':row['gap'],
                            'days_diff':row['days_diff'],
                            'date': date_range, 
                            'days_on_max':days_range,
                            'vspct28': row['vspct28']})
    df_vs_decay = pd.concat([df_vs_decay, temp_df], ignore_index=True)
df_vs_decay['date'] = pd.to_datetime(df_vs_decay['date'])

df_vs_decay= df_vs_decay.merge(df_decay, on=['days_on_max'], how='left')
df_vs_decay['vs_decay'] = df_vs_decay['vspct28'] * df_vs_decay['vs_pct_decay']/100

## Merge eligibles 
df_vs = df_vs_decay.merge(df_eligibles[['region','expire_date','tenure_bucket','eligibles']], left_on=['region','date'], right_on=['region','expire_date'], how='left')
df_vs = df_vs.merge(df_param, on=['tenure_bucket'], how='left')

## Convert %vs to churn, by tenure_bucket  
# dy/dx = a * b * np.exp(b * x_med) 
# dy = a * b * np.exp(b*x_med) * dx 
# dx = vs_decay
# dy =   a * b * np.exp(b*x_med) * vs_decay

df_vs['change_in_churn'] = df_vs['a'] * df_vs['b'] * np.exp(df_vs['b'] *df_vs['median']) * df_vs['vs_decay']
df_vs['change_in_churn_eligibles'] = df_vs['change_in_churn'] * df_vs['eligibles']

grpby= ['region','delta_days','tenure_bucket']
df_vs_tenure = df_vs[grpby+['eligibles','change_in_churn_eligibles']].groupby(by=grpby).sum()

grpby= ['region','delta_days']
df_vs_sum = df_vs[grpby+['eligibles','change_in_churn_eligibles']].groupby(by=grpby).sum().reset_index()

df_vs_sum['pct_change_in_churn'] = df_vs_sum['change_in_churn_eligibles']/df_vs_sum['eligibles']*100

# COMMAND ----------

## Churn reduction post 28d of release date 

df_vs_sum

# COMMAND ----------

19122.289/13051656.625	

# COMMAND ----------

df_vs_tenure

# COMMAND ----------

df_vs.sort_values(by=['region','delta_days', 'days_on_max']).head()
