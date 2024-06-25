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

query = """
select 
* 
from bolt_cus_dev.silver.forecasting_feature_space_test_tj f
left join bolt_cus_dev.gold.delphi_title_metadata d
on f.imdb_id=d.imdb_series_id 
and f.season_number= d.season_number
and f.region = d.geo_value 
where f.premiere_date>='2021-01-01'
    and f.premiere_date<'2024-05-22'
    and d.geo_level='REGION'"""


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
df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()]


## Clean data 
df_raw = df_raw[(df_raw.rfv7.notnull()) & (df_raw.rfv7>0)].sort_values(by=['premiere_date','rfv7'])
df_raw = df_raw.drop_duplicates(subset=['title_name','season_number','region'], keep='last')
df_raw['medal'] = np.where(df_raw['medal']=='Other', df_raw['predicted_medal_us'],df_raw['medal'])
print(df_raw.shape)

# # ## get ref feature 
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
df_raw.loc[(df_raw.content_category=='series') & (df_raw.region=='EMEA'), 'last_release_date'] = df_raw['first_release_date'] + pd.Timedelta(days=60)


## Fix for aquaman
df_raw.loc[(df_raw.title_name.str.contains('Aquaman')) & (df_raw.region=='NORTH AMERICA'), 'fv_pre_7_pay1'] = 16791
df_raw.gap = np.where(df_raw.forecast_category=='popcorn', 0, df_raw.gap)
df_raw = df_raw.sort_values(by=['region','first_release_date','last_release_date']) 

# COMMAND ----------

df_raw[df_raw.forecast_category=='wb pay1'].groupby(by=['medal']).count()

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

                # if filter==['Platinum']:
                #     eligible_rows = df_in[((df_in['medal'].isin(filter)) 
                #         & (df_in['first_release_date'] < row['first_release_date'])
                #         & (df_in['first_release_date'] >= row['first_release_date_ref']))]
                # else:
                
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

df_s = get_days_diff(df_raw)
df_s.head()



# COMMAND ----------

df_s[df_s.title_name.str.contains('Aquaman')]

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


df_in = get_df(df_s[df_s.premiere_date<='2024-05-22'], medal, forecast_category, col_id, features, target)

## data cleaning 
df_in.loc[df_in.production_cost==0, 'production_cost'] = df_in['production_cost'].median()
df_in.loc[(df_in.title_name=='House Party') & (df_in.region=='LATAM'), 'gap'] = 50
df_in.loc[(df_in.title_name=="Magic Mike's Last Dance") & (df_in.region=='LATAM'), 'gap'] = 112
df_in.loc[(df_in.title_name=="Magic Mike's Last Dance") & (df_in.region=='LATAM'), 'bo_opening_imputed'] = 5556243.301106308 #removing negative

def transform_features(df_in, days_diff, features, target):
    df_in.loc[df_in.days_diff<0, 'days_diff'] = 0
    for d in days_diff:
        min_days_diff = 0  
        df_in[d] = df_in[d] + (-min_days_diff)
    df_in = log_features(df_in, features, target)

    # for d in days_diff:
    #     df_in[d] = df_in[d] + min_days_diff -1
    return df_in

# df_in.loc[df_in.days_diff<-45, 'days_diff'] = -45
df_in = transform_features(df_in, ['days_diff'], features, target)


# Remove outliers; Batman: 150K fv; Magic Mike: 3K fv; House Party: ~0 , Shazam throws off pre-7d feature in EMEA
title_removed = ['The Batman',"Magic Mike's Last Dance"]#'The Batman','House Party', "Magic Mike's Last Dance"
title_removed_LATAM = ['House Party','The Batman','The Color Purple',"Magic Mike's Last Dance"]#,  "Magic Mike's Last Dance",'DC League of Super-Pets']#,'The Batman','Barbie']#, 'Blue Beetle']
title_removed_EMEA = ['House Party','The Batman','The Color Purple']#, "Shazam! Fury of the Gods"]#,  'DC League of Super-Pets','Evil Dead Rise', ]
df_in = df_in[((df_in.region=='NORTH AMERICA') & ~(df_in.title_name.isin(title_removed))) |
              ((df_in.region=='EMEA') & ~(df_in.title_name.isin(title_removed_EMEA))) | 
              ((df_in.region=='LATAM') & ~(df_in.title_name.isin(title_removed_LATAM)))]

# COMMAND ----------

## Check df 
display(df_in.sort_values(by=['region','premiere_date']))

import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from itertools import combinations

variables = list(df_in[['ln_rfv7','ln_days_diff','ln_gap','ln_fv_pre_7_pay1', 'ln_bo_opening_imputed','ln_production_cost']])
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

## NA: cgmdp w/ old days_diff def
## LATAM: cmg w/ new days_diff def, min_days_diff=-120


model_list = {
    'pay_a': 'ln_bo_opening_imputed',
    'pay_ag': 'ln_bo_opening_imputed + ln_gap',
    'pay_agd': 'ln_bo_opening_imputed +  ln_gap  + ln_days_diff',
    'pay_amgd': 'ln_bo_opening_imputed + ln_gap   + medal + ln_days_diff',
    # 'pay_amgp': 'ln_bo_opening_imputed + medal + ln_gap  + ln_fv_pre_7_pay1',
    # 'pay_amgdp': 'ln_bo_opening_imputed + medal + ln_gap  + ln_days_diff + ln_fv_pre_7_pay1',
    # 'pay_cm': 'ln_production_cost + medal',
    # # 'pay_cmg': 'ln_production_cost + medal + ln_gap',
    # 'pay_cmgd': 'ln_production_cost + medal + ln_gap + ln_days_diff',
    # 'pay_cmgp': 'ln_production_cost + medal + ln_gap + ln_fv_pre_7_pay1',
    # 'pay_cmgpd': 'ln_production_cost + medal + ln_gap + ln_days_diff + ln_fv_pre_7_pay1',
    # 'pay_dgpd': 'ln_bo_opening_imputed + ln_production_cost + ln_gap + ln_days_diff + ln_fv_pre_7_pay1',
    # 'pay_dmgd': 'ln_bo_opening_imputed + ln_production_cost + medal + ln_gap + ln_days_diff',
    'pay_dgd': 'ln_bo_opening_imputed + ln_gap + ln_days_diff + ln_production_cost ',
    'pay_dgpd': 'ln_bo_opening_imputed + ln_gap + ln_days_diff + ln_fv_pre_7_pay1 + ln_production_cost ',
    'pay_dmgd': 'ln_bo_opening_imputed + ln_gap + ln_days_diff + medal + ln_production_cost ',
    'pay_dmgpd': 'ln_bo_opening_imputed + ln_production_cost + medal + ln_gap + ln_days_diff + ln_fv_pre_7_pay1',
    # 'pay_ogmd': 'ln_bo_opening_imputed + ln_gap + medal + ln_days_diff',
    # 'pay_ogmdc': 'ln_bo_opening_imputed + ln_gap + medal + ln_days_diff  + ln_production_cost',
    # 'pay_ogmp': 'ln_bo_opening_imputed + ln_gap + medal + ln_fv_pre_7_pay1',
    # 'pay_ogmpc': 'ln_bo_opening_imputed + ln_gap + ln_fv_pre_7_pay1 + medal + ln_production_cost',
    # 'pay_ogmpc': 'ln_bo_opening_imputed + ln_gap + medal + ln_fv_pre_7_pay1  + ln_production_cost',
    # 'pay_ogmdp': 'ln_bo_opening_imputed + ln_gap + medal + ln_days_diff  + ln_fv_pre_7_pay1',
    # 'pay_ogmdpc': 'ln_bo_opening_imputed + ln_gap + medal + ln_days_diff  + ln_fv_pre_7_pay1 + ln_production_cost',
    # 'pay_ogmdpc_': 'ln_bo_opening_imputed + ln_gap + medal + days_diff  + ln_fv_pre_7_pay1 + ln_production_cost',
    # 'pay_cgmd': 'ln_production_cost + ln_gap + medal + ln_days_diff',
    # 'pay_cgmp': 'ln_production_cost + ln_gap + medal + ln_fv_pre_7_pay1',
    # 'pay_cgmdp': 'ln_production_cost + ln_gap + medal + ln_days_diff  + ln_fv_pre_7_pay1',
    # 'pay_cgmdp_': 'ln_bo_opening_imputed + ln_gap + medal + days_diff  + ln_fv_pre_7_pay1 + ln_production_cost',

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

import statsmodels.formula.api as smf

for i in model_list:
    model = smf.mixedlm(f"ln_{target} ~" + model_list[i], df_in, groups=df_in["region"], re_formula="~1")
    result = model.fit()
    print(result.summary())

# COMMAND ----------

display(df_summary)

# COMMAND ----------

df_coef[df_coef.region=='NORTH AMERICA'] #WITH MEDAL == PREDICTED MEDAL 

# COMMAND ----------



# COMMAND ----------

df_coef[df_coef.region=='EMEA']

# COMMAND ----------

df_coef[df_coef.region=='LATAM'] 

# COMMAND ----------

## 90d fv
model_dict={'NORTH AMERICA':'pay_dmgpd',
            'EMEA':'pay_amgd',
            'LATAM':'pay_amgd'}

# Define the custom order for the 'Priority' column
priority_order = ['ln_gap', 'ln_days_diff', 'ln_fv_pre_7_pay1', 'ln_bo_opening_imputed','medal[T.Platinum]','ln_production_cost', 'Intercept']
# Convert 'Priority' column to a categorical type with the custom order
df_coef['feature'] = pd.Categorical(df_coef['feature'], categories=priority_order, ordered=True)
df_coef = df_coef.sort_values(by=['feature'])

display(df_coef[(df_coef.region=='NORTH AMERICA') & (df_coef.model=='pay_dmgpd')][['feature','coef','pvalue']]) #new days_diff
display(df_coef[(df_coef.region=='EMEA') & (df_coef.model=='pay_amgd')][['feature','coef','pvalue']]) 
display(df_coef[(df_coef.region=='LATAM') & (df_coef.model=='pay_amgd')][['feature','coef','pvalue']]) 

# COMMAND ----------

model_dict={'NORTH AMERICA':'pay_dmgpd',
            'EMEA':'pay_amgd',
            'LATAM':'pay_amgd'}

# Define the custom order for the 'Priority' column
priority_order = ['ln_gap', 'ln_days_diff', 'ln_fv_pre_7_pay1', 'ln_bo_opening_imputed','medal[T.Platinum]','ln_production_cost', 'Intercept']
# Convert 'Priority' column to a categorical type with the custom order
df_coef['feature'] = pd.Categorical(df_coef['feature'], categories=priority_order, ordered=True)
df_coef = df_coef.sort_values(by=['feature'])

display(df_coef[(df_coef.region=='NORTH AMERICA') & (df_coef.model=='pay_dmgpd')][['feature','coef','pvalue']]) #new days_diff
display(df_coef[(df_coef.region=='EMEA') & (df_coef.model=='pay_amgd')][['feature','coef','pvalue']]) 
display(df_coef[(df_coef.region=='LATAM') & (df_coef.model=='pay_amgd')][['feature','coef','pvalue']]) 

# COMMAND ----------

df_coef[df_coef.feature=='ln_days_diff'].sort_values(by=['region','model'])

# COMMAND ----------

df_coef[df_coef.feature=='ln_fv_pre_7_pay1']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Figures

# COMMAND ----------


def compute_residuals(df, y_plt, coefficients, leave_out):
    vars_to_include = [var for var in coefficients if var not in leave_out]
    print(vars_to_include)
    prediction = sum(df[var] * coefficients[var] for var in vars_to_include) + coefficients['medal[T.Platinum]'] + coefficients['Intercept']
    residuals = df[y_plt] - prediction
    return residuals, prediction

# def get_avg_y(df, y_plt, coefficients, leave_out):


r = 'NORTH AMERICA'
model_dict={'NORTH AMERICA':'pay_dmgpd',
            'EMEA':'pay_amgd',
            'LATAM':'pay_amgd'}
df_plt = df_in[(df_in.region==r) & (df_in.medal=='Platinum')]
coef = df_coef[(df_coef.model==model_dict[r]) & (df_coef.region==r)].set_index('feature')['coef'].to_dict()

y_var = 'rfv7'
list_features = ['days_diff','gap','fv_pre_7_pay1']

for x_var in list_features:
    ln_x_var = 'ln_'+x_var
    ln_y_var = 'ln_'+y_var

    #### RESIDUALS 
    ## Get residuals
    residuals_var, pred_var = compute_residuals(df_plt, ln_y_var, coef, [ln_x_var, 'Intercept', 'medal[T.Platinum]'])
    x_fit = np.linspace(df_plt[ln_x_var].min(), df_plt[ln_x_var].max(), 100)
    y_fit = coef[ln_x_var]*x_fit + residuals_var.mean()

    ## plot residuals
    fig = px.scatter(df_plt, x=x_var, y=residuals_var, title='{} vs. {}'.format(plt_var, x_var), hover_data=['title_name','medal'], template='plotly_white')
    # fit_line = px.line(x=np.exp(x_fit), y=y_fit)
    # for trace in fit_line.data:
    #     fig.add_trace(trace)

    fig.update_layout(
    height=400,
    width=600,
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
    title_font=dict(size=12)
    )

    fig.show(showlegend=True)

    #### RAW DATA
    ## Get fitted line 
    avg_val = b*df_plt[ln_x_var].mean()
    avg_y = pred_var.mean()
    x_fit = np.linspace(df_plt[ln_x_var].min(), df_plt[ln_x_var].max(), 100)
    y_fit = coef[ln_x_var]*x_fit + avg_y #df_plt[ln_y_var].mean() - avg_val

    ##Plot fitted line
    fig = px.scatter(df_plt, x=x_var, y=y_var, title='', hover_data=['title_name','medal'], template='plotly_white', color='region')
    fit_line = px.line(x=np.exp(x_fit), y=np.exp(y_fit))
    for trace in fit_line.data:
        fig.add_trace(trace)

    fig.update_layout(
    height=400,
    width=600,
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
    title_font=dict(size=12)
    )

    fig.update_traces(showlegend=True)
    fig.show()



# COMMAND ----------

df_f = df_sc_fin[df_sc_fin.title_name=='Furiosa']
features= ['days_diff','gap','fv_pre_7_pay1']
features = ['bo_opening_imputed', 'gap', 'production_cost', 'medal', 'fv_pre_7','days_diff','fv_pre_7_pay1']

df_f = log_features(df_f, features, target)

# COMMAND ----------


def compute_residuals(df, y_plt, coefficients, leave_out):
    vars_to_include = [var for var in coefficients if var not in leave_out]
    prediction = sum(df[var] * coefficients[var] for var in vars_to_include) + coefficients['medal[T.Platinum]'] + coefficients['Intercept']
    residuals = df[y_plt] - prediction
    return residuals, prediction


model_dict={'NORTH AMERICA':'pay_dmgpd',
            'EMEA':'pay_amgd',
            'LATAM':'pay_amgd'}

y_var = 'rfv7'
dict_features = {'NORTH AMERICA':['days_diff','gap','fv_pre_7_pay1'],
                 'EMEA':['days_diff','gap'],
                 'LATAM':['days_diff','gap']}
color_dict = {'NORTH AMERICA':['blue'],'EMEA':['green'],'LATAM':['red']}
label_dict = {'days_diff':'# Days from previous Platinum','gap':'# Days from theatrical release','fv_pre_7_pay1':'Pay 1 FV from previous 7 days', 'rfv7':'7d First Views'}


for x_var in ['days_diff','gap','fv_pre_7_pay1']:
    fig = px.scatter(template='plotly_white')
    for r in ['NORTH AMERICA','EMEA','LATAM']:
        try:
            df_plt = df_f[(df_f.region==r)]
            coef = df_coef[(df_coef.model==model_dict[r]) & (df_coef.region==r)].set_index('feature')['coef'].to_dict()
            ln_x_var = 'ln_'+x_var
            ln_y_var = 'ln_'+y_var

            avg_dict =  {'days_diff':{'NORTH AMERICA':np.log(15367)-np.log(12)*coef[ln_x_var],
                                    'EMEA':np.log(6022)-np.log(12)*coef[ln_x_var],
                                    'LATAM':np.log(8558)-np.log(12)*coef[ln_x_var]},
                        'gap':{'NORTH AMERICA':np.log(15367)-np.log(84)*coef[ln_x_var],
                                    'EMEA':np.log(6022)-np.log(84)*coef[ln_x_var],
                                    'LATAM':np.log(8558)-np.log(84)*coef[ln_x_var]},
                        'fv_pre_7_pay1':{'NORTH AMERICA':np.log(15367)-np.log(7450)*coef[ln_x_var],
                                    'EMEA':np.log(6022)-np.log(1903)*coef[ln_x_var],
                                    'LATAM':np.log(8558)-np.log(3314)*coef[ln_x_var]}}



            #### RESIDUALS 
            ## Get residuals
            residuals_var, pred_var = compute_residuals(df_plt, ln_y_var, coef, [ln_x_var, 'Intercept', 'medal[T.Platinum]'])

            #### RAW DATA
            ## Get fitted line 
            avg_val = coef[ln_x_var]*df_plt[ln_x_var].mean()
            avg_y = avg_dict[x_var][r]
            print(avg_y)
            x_fit = np.linspace(df_in[ln_x_var].min(), df_in[ln_x_var].max(), 100)
            y_fit = coef[ln_x_var]*x_fit + avg_y #df_plt[ln_y_var].mean() - avg_val
            ##Plot fitted line
            # fit_markers = px.scatter(df_plt, x=x_var, y=y_var, title='',hover_data=['title_name','medal'], color_discrete_sequence=color_dict[r])
            # for trace in fit_markers.data:
            #     trace.name = r
            #     fig.add_trace(trace)
            fit_line = px.line(x=np.exp(x_fit), y=np.exp(y_fit),color_discrete_sequence=color_dict[r])
            for trace in fit_line.data:
                trace.name=r
                trace.update(line=dict(width=1.5))
                fig.add_trace(trace)
            
        except:
            pass

    fig.update_layout(
    height=400,
    width=600,
    yaxis_range =[2000,22000],
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
    xaxis_title= label_dict[x_var],
    yaxis_title = label_dict[y_var]
    )

    fig.update_traces()
    fig.show()




# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Sensitivity analysis

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

df_ss = spark.sql(query)
df_ss = df_ss.toPandas()
df_ss.columns = df_ss.columns.str.lower()

df_ss.loc[df_ss['title_name']=='The Watchers', 'premiere_date'] = '2024-08-23'
df_ss.loc[df_ss['title_name']=='The Watchers', 'first_release_date'] = '2024-08-23'
df_ss.loc[df_ss['title_name']=='The Watchers', 'last_release_date'] = '2024-08-23'

df_ss.premiere_date=pd.to_datetime(df_ss.premiere_date)
print(df_ss.premiere_date.max())

## Data cleaning 
df_ss['last_release_date'] = np.where(df_ss.forecast_category=='series',
                                      df_ss['premiere_date'] + pd.to_timedelta(49, unit='D'),
                                      df_ss['premiere_date'])
df_ss['first_release_date'] = df_ss['premiere_date']


df_ss.first_release_date =  pd.to_datetime(df_ss.first_release_date).dt.date    
df_ss.last_release_date =  pd.to_datetime(df_ss.last_release_date).dt.date    

display(df_ss[df_ss.title_name=='Furiosa']) 
display(df_ss[df_ss.title_name=='The Watchers']) 
df_ss.loc[df_ss.title_name=='Furiosa','gap'] = 77   




# COMMAND ----------

df_sc_fin[df_sc_fin.title_name=='Furiosa']

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
and imdb_id = 'tt12037194' --Furiosa
""", con) 

df_fv_title_watcher = pd.read_sql("""
select 
    date, 
    region,
    retail_fv as fv_title_watcher
from max_dev.workspace.forecasting_fv_pacing a
where year(date) = 2024
-- and imdb_id='tt14539740' --gxk
and imdb_id = 'tt26736843' --watchers
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
df_fv_title_watcher = clean_fv_data(df_fv_title_watcher)
df_fv_title = clean_fv_data(df_fv_title)

# ## Correct for watcher date (to Jul 30)
# df_fv_title_watcher_new = df_fv_title_watcher.copy()
# df_fv_title_watcher_new['date'] = df_fv_title_watcher_new['date'] + pd.to_timedelta(7, unit='D')
# df_fv = df_fv.merge(df_fv_title_watcher, on=['date','region'],how='left').merge(df_fv_title_watcher_new, on=['date','region'],how='left', suffixes=('','_new'))
# df_fv = df_fv.fillna(0)
# df_fv['fv_pay1'] = df_fv['fv_pay1']-df_fv['fv_title_watcher'] + df_fv['fv_title_watcher_new']



## Merge furiosa data
df_fv = df_fv.merge(df_fv_title, on=['date','region'],how='left')
df_fv = df_fv.fillna(0)

## Build scheduling scenarios 
title = 'Furiosa'
delta_days_list= [0,7,14,21]
list_df=[]

for delta_days in delta_days_list:
    ## Get scenario premiere data
    # Furiosa
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

    df_result=df_sc_fin[((df_sc_fin.forecast_category=='wb pay1')) |
                        (df_sc_fin.medal=='Platinum')][['region','title_name','first_release_date','last_release_date','gap','days_diff','fv_pre_7_pay1','prev_title',]]
    df_result['delta_days'] = delta_days


    display(df_result[df_result.title_name=='Furiosa'])
    list_df.append(df_result)

df_result = pd.concat(list_df)


# COMMAND ----------

def calc_change(df_in_all, delta_days, list_features, df_coef, model_dict, title, region_list=['NORTH AMERICA','EMEA','LATAM']):
    df_1 = df_in_all[df_in_all.delta_days==7]
    df_2 = df_in_all[df_in_all.delta_days==delta_days]
    df_comp = df_1.merge(df_2, on=['title_name','region'],  suffixes=('','_2'))
    df_comp['total_delta_fvr'] = 0
    
    df_list=[]
    for r in region_list:
        df_comp_r = df_comp[df_comp.region==r]
        coef = df_coef[(df_coef.model==model_dict[r]) & (df_coef.region==r)]
        for i in list_features:
            try:
                coeff_val = coef.loc[coef.feature=='ln_'+i, 'coef'].values[0]
                df_comp_r['{}_coeff'.format(i)] = coeff_val
                df_comp_r['{}_delta_fvr'.format(i)]= np.exp(np.log(df_comp_r['{}_2'.format(i)])*coeff_val)/np.exp(np.log(df_comp_r['{}'.format(i)])*coeff_val)-1
                df_comp_r['total_delta_fvr'] = df_comp_r['total_delta_fvr']+df_comp_r['{}_delta_fvr'.format(i)]
            except:
                df_comp_r['{}_coeff'.format(i)] = 0
                df_comp_r['{}_delta_fvr'.format(i)] = 0
                df_comp_r['total_delta_fvr'] = df_comp_r['total_delta_fvr']+df_comp_r['{}_delta_fvr'.format(i)]
        df_list.append(df_comp_r)
    df_comp = pd.concat(df_list)
    df_sum = df_comp[df_comp.title_name==title][['region','gap_delta_fvr','days_diff_delta_fvr','fv_pre_7_pay1_delta_fvr','total_delta_fvr']]
    return df_comp,df_sum

list_features = ['gap','days_diff','fv_pre_7_pay1']
model_dict={'NORTH AMERICA':'pay_dmgpd',
            'EMEA':'pay_amgd',
            'LATAM':'pay_amgd'}

df_0, df_0_sum =calc_change(df_result, 0, list_features, df_coef, model_dict,'Furiosa')
# df_7, df_7_sum =calc_change(df_result, 7, list_features, df_coef, model_dict,'Furiosa')
df_14, df_14_sum =calc_change(df_result, 14, list_features, df_coef, model_dict,'Furiosa')
df_21, df_21_sum =calc_change(df_result, 21, list_features, df_coef, model_dict,'Furiosa')

df_final = pd.concat([df_0_sum, df_14_sum, df_21_sum], ignore_index=True)
df_fin = df_final.T

custom_order = ['NORTH AMERICA','EMEA','LATAM']  # Custom order values
row_values = df_fin.iloc[0]
custom_order_dict = {val: i for i, val in enumerate(custom_order)}
ordered_columns = row_values.map(custom_order_dict).sort_values().index
df_fin = df_fin[ordered_columns]

df_fin = df_fin.reset_index()
display(df_fin)

# COMMAND ----------

query = """
select 
title_name,
region,
sum(prediction)
from bolt_cus_dev.gold.delphi_predictions
where title_name='Furiosa'
and model_name='mds_prelaunch_retail'
and DAYS_AFTER_PREMIERE <=7
group by title_name, region
"""

query = """
select 
*
from bolt_cus_dev.silver.forecasting_feature_space
where title_name='Furiosa'

"""

#from bolt_cus_dev.gold.delphi_predictions
df_ss = spark.sql(query)
df_ss = df_ss.toPandas()
df_ss

# COMMAND ----------

21616.354/34815.124	

# COMMAND ----------

df_test[df_test.title_name=='The Penguin'].head()

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


