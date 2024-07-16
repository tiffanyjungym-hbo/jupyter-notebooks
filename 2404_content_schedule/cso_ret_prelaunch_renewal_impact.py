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

for file in (utils_imports,  utils_import_data, utils_models):# utils_sf,
    with open(file, "r") as file:
        python_code = file.read()
    exec(python_code)

pd.options.display.float_format = '{:,.3f}'.format



# COMMAND ----------

# MAGIC %md
# MAGIC ## Check change in churn by days before launch of big title

# COMMAND ----------

## analyse change in renewal by upcoming titles in the next 2 weeks -> per daily eligible 
## Change in churn from title ~ # days prelaunch 
## Change in renewal after title A ~  # days till next title (after billing cycle). or platinum series after billing cycle.. 
## eligibles renewal rate time trend. 

# df = pd.read_sql("""
# select * 
# from  MAX_DEV.WORKSPACE.FORECASTING_PROJ_CANCELS_US 
# where expire_date>='2022-01-01'""", con)

# df.columns = df.columns.str.lower()
# display(df.head())

# df_sum = df.groupby(by=['expire_date']).sum().reset_index()
# df_sum[df_sum.expire_date=='2024-05-01']

# COMMAND ----------

## Not sure what the diff is from FORECASTING_PROJ_CANCELS_US ??
# df_eligibles = pd.read_sql("""
#     select 
#     region,
#     expire_date,
#     tenure,
#     sum(eligibles) as eligibles, 
#     sum(paid_cancels) as cancels
#     from MAX_DEV.WORKSPACE.FORECASTING_PROJ_CANCELS_FUTURE 
#     where expire_date::date between '2022-01-01' and '2024-07-07'
#     and payment_period='PERIOD_MONTH'
#     and ad_strategy = 'ad_free'
#     and payment_period = 'PERIOD_MONTH'
#     and signup_offer = 'none'
#     and dtc='DIRECT'
#     group by all 
#     order by 1,2
#     """, 
#     con)

df_eligibles = pd.read_csv('/Workspace/Repos/tiffany.jung@wbd.com/jupyter-notebooks/2404_content_schedule/data/df_eligibles.csv')
df_eligibles.columns = df_eligibles.columns.str.lower()

df_eligibles['expire_weekend'] = pd.to_datetime(df_eligibles['expire_date']).dt.to_period('W').dt.end_time
df_eligibles['expire_weekend'] = pd.to_datetime(df_eligibles['expire_weekend'])

df_eligibles['expire_mth'] = pd.to_datetime(df_eligibles['expire_date']).dt.to_period('M').dt.start_time
df_eligibles['expire_mth'] = pd.to_datetime(df_eligibles['expire_mth'])

df_eligibles['tenure'] = df_eligibles['tenure'].astype(int)
mapping_dict = {1:'month_1', 2:'month_2', 3:'month_3', 4:'month_4_to_6', 5:'month_4_to_6', 6:'month_4_to_6', 7:'month_7_to_12', 8:'month_7_to_12', 9:'month_7_to_12', 10:'month_7_to_12', 11:'month_7_to_12', 12:'month_7_to_12'}
df_eligibles['tenure_bucket'] = df_eligibles['tenure'].map(mapping_dict).fillna('month_13+')
# df_eligibles['cancel_rate'] = df_eligibles['cancels'] /df_eligibles['eligibles']

# df_eligibles[df_eligibles.expire_date=='2024-05-01'].groupby(by=['region']).sum()
df_eligibles.head()
# df_eligibles.expire_date.unique()

# COMMAND ----------

query = """
select 
title_name,
season_number,
premiere_date,
finale_date,
TFV7
from bolt_cus_dev.silver.forecasting_fv_summary a 
where region='NORTH AMERICA'
and premiere_date>='2022-01-05'
"""

df_title = spark.sql(query)
df_title = df_title.toPandas()

df_title.columns = df_title.columns.str.lower()
df_title.loc[(df_title.title_name=='House of the Dragon') & (df_title.season_number==2), 'title_name'] = 'House of the Dragon 2'
df_title[df_title.tfv7>=50000].shape
df_title['premiere_date'] = pd.to_datetime(df_title['premiere_date'])
df_title['finale_date'] = pd.to_datetime(df_title['finale_date'])

# COMMAND ----------


titles_to_exclude=['Black Adam','True Detective','The Last of Us','South Park','Euphoria','The Menu','Peacemaker','The Gilded Age','The Idol','Avatar: The Way of Water','A Christmas Story Christmas','The White Lotus','Wonka','Aquaman and the Lost Kingdom','Elvis','Succession'] ## removing january/december titles and holloween (whitelotus, christmas)  and avatar/idol wonka/aquaman/Elvis/Succession that overlap too closely 
# df_sum.loc[(df_sum.title_name=='House of the Dragon') & (df_sum.season_number==2), 'title_name'] = 'House of the Dragon 2'
df_title[(df_title.tfv7>=50000) & (~df_title.title_name.isin(titles_to_exclude))]

# COMMAND ----------

for date in df_title[(df_title.tfv7>=50000) & (~df_title.title_name.isin(titles_to_exclude))].premiere_date.tolist():
    print(date, df_title[(df_title.premiere_date==date)& (df_title.tfv7>50000)].title_name.tolist())
    date_pre = pd.to_datetime(date) + pd.DateOffset(days=-30)
    display(df_title[(df_title.finale_date>date_pre) & (df_title.finale_date<date) & (df_title.tfv7>=10000)].sort_values(by=['premiere_date']))

# COMMAND ----------

tenure_df_list =[]
fv_threshold=40000 #140000
for tenure in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]:

    df_test = df_eligibles[(df_eligibles.tenure.between(tenure,tenure)) & (df_eligibles.region=='NORTH AMERICA')]
    df_test.expire_date = pd.to_datetime(df_test.expire_date)
    df_test = df_test.groupby(by=['tenure','tenure_bucket','expire_date']).sum().reset_index()
    df_title.premiere_date = pd.to_datetime(df_title.premiere_date)

    df_merged = df_test.merge(df_title[(df_title.tfv7>=fv_threshold)], right_on=['premiere_date'], left_on=['expire_date'], how='left').fillna(0)

    df_list=[]
    df_tenure = df_merged.copy()
    for i in ['cancels','eligibles']:
        df_tenure[f'{i}_28_21'] = df_tenure[i].rolling(window=7, min_periods=1).sum().shift(21)
        df_tenure[f'{i}_21_14'] = df_tenure[i].rolling(window=7, min_periods=1).sum().shift(13)
        df_tenure[f'{i}_14_7'] = df_tenure[i].rolling(window=7, min_periods=1).sum().shift(6)
        df_tenure[f'{i}_7_0'] = df_tenure[i].rolling(window=7, min_periods=1).sum().shift(1)
        df_tenure[f'{i}_0_7'] = df_tenure[i].rolling(window=7, min_periods=1).sum().shift(-6)
        df_tenure[f'{i}_7_14'] = df_tenure[i].rolling(window=7, min_periods=1).sum().shift(-13)
    df_list.append(df_tenure)

    df_sum_ = pd.concat(df_list)
    df_sum = df_sum_[df_sum_.title_name!=0]


    for i in ['28_21','21_14','14_7','7_0','0_7','7_14']:
        df_sum[f'cancel_rate_{i}'] = df_sum[f'cancels_{i}']/df_sum[f'eligibles_{i}']
    df_sum['pct_diff_21_14'] = -(df_sum['cancel_rate_28_21']-df_sum['cancel_rate_21_14'])/df_sum['cancel_rate_28_21']
    df_sum['pct_diff_14_7'] = -(df_sum['cancel_rate_21_14']-df_sum['cancel_rate_14_7'])/df_sum['cancel_rate_21_14']
    df_sum['pct_diff_7_0'] = -(df_sum['cancel_rate_14_7']-df_sum['cancel_rate_7_0'])/df_sum['cancel_rate_14_7']
    df_sum['pct_diff_0_7'] = -(df_sum['cancel_rate_7_0']-df_sum['cancel_rate_0_7'])/df_sum['cancel_rate_7_0']
    df_sum['pct_diff_7_14'] = -(df_sum['cancel_rate_0_7']-df_sum['cancel_rate_7_14'])/df_sum['cancel_rate_0_7']

    titles_to_exclude=['Black Adam','True Detective','The Last of Us','South Park','Euphoria','The Menu','Peacemaker','The Gilded Age','The Idol','Avatar: The Way of Water','A Christmas Story Christmas','The White Lotus','Wonka','Aquaman and the Lost Kingdom','Elvis','Succession','Quiet on Set: The Dark Side of Kids TV'] ## removing january/december titles and holloween (whitelotus, christmas)  and avatar/idol wonka/aquaman/elvis/succession that overlap too closely 
    df_sum.loc[(df_sum.title_name=='House of the Dragon') & (df_sum.season_number==2), 'title_name'] = 'House of the Dragon 2'

    tenure_df_list.append(df_sum)


df_sum_tot = pd.concat(tenure_df_list)

## Title level plot 
df_melted = df_sum_tot.melt(id_vars=['title_name','season_number','tenure'], value_vars=['cancel_rate_21_14', 'cancel_rate_14_7', 'cancel_rate_7_0','cancel_rate_0_7','cancel_rate_7_14'], var_name='days', value_name='cancel_rate')
fig = px.line(df_melted[(~df_melted.title_name.isin(titles_to_exclude)) & (df_melted.tenure==5)], x='days', y='cancel_rate', color='title_name', markers=True, title='tenure=4')
fig.show()

## Title level average 
pct_vars = ['pct_diff_21_14','pct_diff_14_7', 'pct_diff_7_0', 'pct_diff_0_7','pct_diff_7_14']
df_melted_chg = df_sum_tot[~df_sum_tot.title_name.isin(titles_to_exclude)].melt(id_vars=['title_name','season_number','tenure','tenure_bucket'], value_vars=pct_vars, var_name='days', value_name='pct_change_cancel_rate')
df_melted_chg_grp = df_melted_chg.groupby(by=['days','tenure']).mean().reset_index()

df_melted_chg_grp['days'] = pd.Categorical(df_melted_chg_grp['days'], categories=pct_vars, ordered=True)
df_melted_chg_grp = df_melted_chg_grp.sort_values('days')
fig = px.line(df_melted_chg_grp, x='days', y='pct_change_cancel_rate', markers=True, title='% change in churn from previous week, 7-0 excluding day of launch', color='tenure')
fig.show()

df_melted_chg_grp = df_melted_chg.groupby(by=['days','tenure_bucket']).mean().reset_index()
df_melted_chg_grp['days'] = pd.Categorical(df_melted_chg_grp['days'], categories=pct_vars, ordered=True)
df_melted_chg_grp = df_melted_chg_grp.sort_values('days')
fig = px.line(df_melted_chg_grp, x='days', y='pct_change_cancel_rate', markers=True, title='% change in churn from previous week 7-0 excluding day of launch', color='tenure_bucket')
fig.show()

# custom_order = ['cancel_rate_21_14', 'cancel_rate_14_7', 'cancel_rate_7_0', 'cancel_rate_0_7']
# df_melted_sum['days'] = pd.Categorical(df_melted_sum['days'], categories=custom_order, ordered=True)
# df_melted_sum = df_melted_sum.sort_values('days')
# fig = px.line(df_melted_sum, x='days', y='cancel_rate', markers=True, title='')
# fig.show()



# COMMAND ----------

# MAGIC %md
# MAGIC ## diff-in-diff to find impact of new title launch on pre-launch churn

# COMMAND ----------

df_title[(df_title.tfv7>=fv_threshold) &~(df_title.title_name.isin(titles_to_exclude+[0]))]
## Remove Fantastic Beasts, Elvis,  Succession, Shazam (doesn't meet parallel assumption)

# COMMAND ----------

df_title[(df_title.tfv7>=40000)]

# COMMAND ----------

fv_threshold = 50000
def get_period_data(df_in, date_list, title_list=['ctrl']):
    df_date_list = []
    n=0
    
    if len(title_list)==1:
        title_list = title_list * len(date_list)

    for date, title in zip(date_list, title_list):
        df_test_date = df_in[df_in.expire_date.between(date[0],date[1])].reset_index()
        df_test_date['day'] = 0
        df_test_date['period'] = n
        df_test_date['title_name'] = title
        n+=1
        ## start counting days from 0
        d=0
        print()
        if title=='ctrl':
            df_test_date['day'] = [i + 1 for i, _ in enumerate(df_test_date.index)]
        else:
            # df_test_date['day'] = [i + 1 for i, _ in enumerate(df_test_date.index)]
            df_test_date['day'] = range(-14, -14 + len(df_test_date))

        df_date_list.append(df_test_date)

    df_out = pd.concat(df_date_list)
    return df_out

## Get control & treatment periods 
# date_start_ctr = ['2022-07-11','2022-09-17','2023-07-06','2024-04-01']
# date_end_ctr = ['2022-08-06','2022-10-15','2023-07-24','2024-05-06']
date_start_ctr= ['2022-07-13','2022-10-01','2023-07-03','2024-02-01']
date_end_ctr = ['2022-07-26','2022-10-15','2023-07-18','2024-05-14']
# date_start_ctr = ['2022-02-27','2022-10-03','2023-09-25']
# date_end_ctr = ['2022-03-18','2022-10-17','2023-10-17']
# date_start_ctr = ['2022-02-11','2022-06-14','2022-09-17','2023-02-01','2024-04-01']
# date_end_ctr = ['2022-03-18','2022-07-09','2022-10-02','2023-02-23','2024-04-20']

titles_to_exclude=['Black Adam','True Detective','The Last of Us','South Park','Euphoria','The Menu','Peacemaker','The Gilded Age','The Idol','Avatar: The Way of Water','A Christmas Story Christmas','The White Lotus','Elvis','Aquaman and the Lost Kingdom','Succession','Quiet on Set: The Dark Side of Kids TV','Barbie','A Christmas Story Christmas','Godzilla x Kong: The New Empire'] ## removing january/december titles and holloween (whitelotus, christmas)  and avatar/idol wonka/aquaman/elvis/succession that overlap too closely 

date_premiere = df_title[(df_title.tfv7>=fv_threshold) & (~df_title.title_name.isin(titles_to_exclude))].premiere_date.unique()
pre_30d = [dt + timedelta(days=-14) for dt in pd.to_datetime(date_premiere)]
post_30d = [dt + timedelta(days=14) for dt in pd.to_datetime(date_premiere)]

date_ctrl = list(zip(date_start_ctr, date_end_ctr))
date_trt = list(zip(pre_30d, post_30d))
title_list = df_title[(df_title.tfv7>=fv_threshold) & (~df_title.title_name.isin(titles_to_exclude))].title_name.unique()


df_tenure_list=[]
for tenure in [4]:
    df_test = df_eligibles[(df_eligibles.tenure.between(tenure,tenure)) & (df_eligibles.region=='NORTH AMERICA')]
    df_test = df_test.groupby(by=['tenure_bucket','expire_date']).sum().reset_index()
    df_test['cancel_rate'] = df_test['cancels']/df_test['eligibles']
    df_test['expire_date'] = pd.to_datetime(df_test.expire_date)

    df_ctrl = get_period_data(df_test, date_ctrl)
    df_trt = get_period_data(df_test, date_trt, title_list)

    df_ctrl['ctrl_trt'] = 'ctrl'
    df_trt['ctrl_trt'] = 'trt'

    df_plt = pd.concat([df_ctrl, df_trt])
    df_plt['ctrl_trt_period'] = df_plt['ctrl_trt'] + '_' + df_plt['period'].astype(str)
    df_tenure_list.append(df_plt)

df_plt = pd.concat(df_tenure_list)


## plt
fig = px.line(df_plt[(df_plt.ctrl_trt=='trt')], x='day', y='cancel_rate', markers=True, title='churn_ treatment', color='ctrl_trt_period')
fig.show()

## plt
fig = px.line(df_plt[df_plt.ctrl_trt=='ctrl'], x='day', y='cancel_rate', markers=True, title='churn_control', color='ctrl_trt_period')
fig.show()

# COMMAND ----------



# df_trt = get_period_data(df_test, date_trt, title_list)
df_plt[(df_plt.ctrl_trt=='trt') & (df_plt.day==0)]

# COMMAND ----------

## title by title regression
trt_grp = ['trt_0', 'trt_2','trt_3','trt_4','trt_5','trt_6','trt_7']
ctrl_grp = ['ctrl_1','ctrl_1','ctrl_2','ctrl_2','ctrl_2','ctrl_2','ctrl_2']
df_trt_dd = df_plt[(df_plt.ctrl_trt_period.isin(trt_grp))& (df_plt.day<=0)]

## ctrl 
df_ctrl_dd = df_plt[((df_plt.ctrl_trt_period.isin(['ctrl_0','ctrl_1','ctrl_2','ctrl_3'])) & (df_plt.day<=15))]

df_dd = pd.concat([df_trt_dd, df_ctrl_dd])
df_dd['day'] = df_dd.groupby('ctrl_trt_period').cumcount()
df_dd['day'] = -14 + df_dd['day']

df_dd[['treatment','post']] = 0
df_dd.loc[df_dd.ctrl_trt=='trt', 'treatment']=1
df_dd.loc[df_dd.day>=-7, 'post']=1
df_dd['treat_post'] = df_dd['treatment'] * df_dd['post']

import statsmodels.formula.api as smf

df_list = []
for trt,ctrl in zip(trt_grp, ctrl_grp):
    title = df_dd.loc[df_dd.ctrl_trt_period.isin([trt]), 'title_name'].values[0]
    df_plt_trt=df_dd[df_dd.ctrl_trt_period.isin([trt])].reset_index()
    df_plt_ctrl=df_dd[df_dd.ctrl_trt_period.isin([ctrl])].reset_index()
    # fig = px.line(df_dd[df_dd.ctrl_trt_period.isin([trt,ctrl])], x='day', y='cancel_rate', title=f'{title}', color='ctrl_trt_period',  markers=True)
    
    # fig = px.line(df_plt_trt, x='day', y='cancel_rate', labels={'y': 'Group 1'}, title=f'{title}',  markers=True, template='plotly_white')
    # fig.add_scatter(x=df_plt_ctrl['day'], y=df_plt_ctrl['cancel_rate'], mode='lines + markers', name='control', yaxis='y2')


    # Create the line plot for treatment group
    fig = px.line(df_plt_trt, x='day', y='cancel_rate', markers=True, title = f'{title} pre-launch churn', template='plotly_white')
    fig.update_traces(name='Prelaunch churn')
    fig_control = px.line(df_plt_ctrl, x='day', y='cancel_rate', markers=True)
    fig_control.update_traces(name='control', yaxis='y2',line=dict(color='red'), marker=dict(color='red'))

    # Combine the two plots
    fig.add_traces(fig_control.data)

    # diff_y1 = df_plt_trt['cancel_rate'].max()- df_plt_trt['cancel_rate'].min()
    # diff_y2 = df_plt_ctrl['cancel_rate'].max()- df_plt_ctrl['cancel_rate'].min()
    
    # diff = max(diff_y1, diff_y2) + 0.003
    y1_min = df_plt_trt.loc[0, 'cancel_rate']
    y2_min = df_plt_ctrl.loc[0,'cancel_rate'] 
    


    fig.update_layout(
        yaxis2=dict(
            title='cancel rate, control',
            overlaying='y',
            side='right',
            range = [y2_min-0.015, y2_min+0.01]
        ),
        yaxis=dict(
            title='cancel rate, treatment',
            range = [y1_min-0.015, y1_min+0.01]
        ))
    
    fig.update_layout(
        height=400,
        width=600,
        xaxis=dict(showline=True,
                   mirror=True,
            title=dict(font=dict(size=12)),   
            linecolor='black',
            ticks='outside',
            tickmode='linear',
            tick0=0,
            dtick=1,
            showgrid=False
        ),
        yaxis1=dict(
            title=dict(font=dict(size=12)),   
            linecolor='black',
            ticks='outside'
        ),
            yaxis2=dict(
            title=dict(font=dict(size=12)),   
            linecolor='black',
            ticks='outside',
            showgrid=False
        ),
        font=dict(size=12),  # Font size for the rest of the text elements
        title_font=dict(size=14),
        xaxis_title= 'days before premiere'
        )


    fig.show()

    # Run DiD regression
    model = smf.ols(formula='cancel_rate ~ treatment + post + treat_post', data=df_dd[df_dd.ctrl_trt_period.isin([trt,ctrl])])
    results = model.fit()
    # Combine coefficients and p-values into a DataFrame
    results_df = pd.DataFrame({
        'title':title,
        'coeff_name' : results.params.index,
        'coeff_value': results.params.values,
        'p_value': results.pvalues.values
    })

    df_list.append(results_df)

df_results= pd.concat(df_list).reset_index()

p_values = df_results[df_results.coeff_name=='treat_post']['p_value']
from scipy.stats import combine_pvalues
_, combined_p = combine_pvalues(p_values.tolist())
print(df_results[(df_results.coeff_name=='treat_post')].coeff_value.mean(),combined_p)
df_results['Churn reduction 7d prior to release'] = df_results['coeff_value']*100
df_results['p value'] = df_results['p_value']
df_results[(df_results.coeff_name=='treat_post')][['title','Churn reduction 7d prior to release','p value']]

# COMMAND ----------

trt = 'trt_5'
ctrl='ctrl_2'
title = df_dd.loc[df_dd.ctrl_trt_period.isin([trt]), 'title_name'].values[0]
df_plt_trt=df_dd[df_dd.ctrl_trt_period.isin([trt])].reset_index()
df_plt_ctrl=df_dd[df_dd.ctrl_trt_period.isin([ctrl])].reset_index()
# fig = px.line(df_dd[df_dd.ctrl_trt_period.isin([trt,ctrl])], x='day', y='cancel_rate', title=f'{title}', color='ctrl_trt_period',  markers=True)

fig = px.line(df_plt_trt, x='day', y='cancel_rate', labels={'y': 'Group 1'}, title=f'{title}',   markers=True)
fig.add_scatter(x=df_plt_ctrl['day'], y=df_plt_ctrl['cancel_rate'], mode='lines', name='control', yaxis='y2')

# diff_y1 = df_plt_trt['cancel_rate'].max()- df_plt_trt['cancel_rate'].min()
# diff_y2 = df_plt_ctrl['cancel_rate'].max()- df_plt_ctrl['cancel_rate'].min()

# diff = max(diff_y1, diff_y2) + 0.003
y1_min = df_plt_trt.loc[0, 'cancel_rate']
y2_min = df_plt_ctrl.loc[0,'cancel_rate'] 
    

# COMMAND ----------

# results.coefficients

# COMMAND ----------


# treat_post    -0.0009      0.002     -0.517      0.610      -0.004       0.003
# '''
#                                     coef        p val
# treat_post0 Batman                -0.0029       0.008     
# treat_post2 HOTD                  -0.0028       0.002      
# treat_post6 Love & Death,         -0.0036       0.013    
# treat_post8 And Just Like That,   -0.0006       0.575    
# treat_post9 The Flash             -0.0017       0.043    
# '''

# COMMAND ----------

## group regression
# trt_grp = ['trt_0', 'trt_2','trt_6','trt_8','trt_9']
# ctrl_grp = ['ctrl_1','ctrl_1','ctrl_2','ctrl_2','ctrl_2']

# trt_grp =  ['trt_0', 'trt_2','trt_4','trt_6','trt_7','trt_8','trt_9','trt_10','trt_11','trt_12','trt_13','trt_14']
# ctrl_grp = ['ctrl_1','ctrl_1','ctrl_1','ctrl_2','ctrl_2','ctrl_2','ctrl_2','ctrl_2','ctrl_2','ctrl_3','ctrl_3','ctrl_3']

trt_grp =  ['trt_0', 'trt_2','trt_4','trt_5','trt_6','trt_7','trt_8','trt_9','trt_10'] 
ctrl_grp = ['ctrl_1','ctrl_1','ctrl_1','ctrl_2','ctrl_2','ctrl_2','ctrl_2','ctrl_2','ctrl_2']

# df_trt_dd = df_plt[(df_plt.ctrl_trt_period.isin(trt_grp))& (df_plt.day>=15) & (df_plt.day<=29)]

# ## ctrl 
# df_ctrl_dd = df_plt[((df_plt.ctrl_trt_period.isin(['ctrl_0','ctrl_2','ctrl_3'])) & (df_plt.day<=15)) |
#                      ((df_plt.ctrl_trt_period.isin(['ctrl_1'])) & (df_plt.day>=15)& (df_plt.day<=29))]
# df_ctrl_dd.loc[(df_ctrl_dd.ctrl_trt_period=='ctrl_2') & (df_ctrl_dd.day==15),'cancel_rate']= df_ctrl_dd.loc[(df_ctrl_dd.ctrl_trt_period=='ctrl_2') & (df_ctrl_dd.day==14),'cancel_rate'].values[0]

df_list=[]
n=0
for trt,ctrl in zip(trt_grp, ctrl_grp):
    df_temp_trt = df_trt_dd[df_trt_dd.ctrl_trt_period.isin([trt])]
    df_temp_ctrl = df_ctrl_dd[df_ctrl_dd.ctrl_trt_period.isin([ctrl])]
    df_temp_trt['trt_grp'] = n
    df_temp_ctrl['trt_grp'] = n
    df_list.append(df_temp_trt)
    df_list.append(df_temp_ctrl)
    n+=1


df_dd = pd.concat(df_list)
df_dd['trt_grp'] = df_dd['trt_grp'].astype(int)
df_dd['day'] = df_dd.groupby(['trt_grp','ctrl_trt_period']).cumcount()
df_dd['day'] = -14 + df_dd['day']

fig = px.line(df_dd, x='day', y='cancel_rate', markers=True, title='diff in diff', color='ctrl_trt_period')
fig.show()


import statsmodels.formula.api as smf

## add grp features 
one_hot = pd.get_dummies(df_dd['trt_grp'], prefix='grp')
df_dd_fin = df_dd.join(one_hot)

# treatment and post features 
df_dd_fin[['treatment','post']] = 0
df_dd_fin.loc[df_dd_fin.ctrl_trt=='trt', 'treatment']=1
df_dd_fin.loc[df_dd_fin.day>=-7, 'post']=1

# interaction features 
for i in one_hot.columns.tolist():
    df_dd_fin[f'{i}_post'] = df_dd_fin[i] * df_dd_fin['post']
    # df_dd_fin['grp1_post'] = df_dd_fin['grp_1'] * df_dd_fin['post']
    # df_dd_fin['grp2_post'] = df_dd_fin['grp_2'] * df_dd_fin['post']
    # df_dd_fin['grp3_post'] = df_dd_fin['grp_3'] * df_dd_fin['post']
    # df_dd_fin['grp4_post'] = df_dd_fin['grp_4'] * df_dd_fin['post']

formula = 'cancel_rate ~ grp_0 + grp_1 + grp_2 + grp_3 + grp_4 + grp_5 + grp_6 + grp_7 + grp_8 + post + grp_0_post +grp_1_post + grp_2_post + grp_3_post + grp_4_post+ grp_5_post + grp_6_post+ grp_7_post + grp_8_post'
# formula = 'cancel_rate ~ grp_0 +  grp_2 +  post + grp0_post  + grp2_post '

model = smf.ols(formula=formula, data=df_dd_fin)
results = model.fit(cov_type='cluster', cov_kwds={'groups': df_dd_fin['trt_grp']})

# Print results
print(results.summary())



# df_dd[['treatment','post']] = 0
# df_dd.loc[df_dd.ctrl_trt=='trt', 'treatment']=1
# df_dd.loc[df_dd.day>=-7, 'post']=1
# df_dd['treat_post'] = df_dd['treatment'] * df_dd['post']

# import statsmodels.formula.api as smf

# for trt,ctrl in zip(trt_grp, ctrl_grp):
#     fig = px.line(df_dd[df_dd.ctrl_trt_period.isin([trt,ctrl])], x='day', y='cancel_rate', markers=True, title=f'diff in diff {trt} {ctrl}', color='ctrl_trt_period')
#     fig.show()

#     # Run DiD regression
#     model = smf.ols(formula='cancel_rate ~ treatment + post + treat_post', data=df_dd[df_dd.ctrl_trt_period.isin([trt,ctrl])])
#     results = model.fit()

#     # Print results
#     print(results.summary())


# COMMAND ----------

df_dd_fin.columns

# COMMAND ----------

trt_grp = ['trt_0', 'trt_2','trt_6','trt_8','trt_9'] #['trt_2','trt_6','trt_9','trt_12']
ctrl_grp_first_day:{'ctrl_0':0, 'ctrl_1':15, 'ctrl_2':0, 'ctrl_3':0}

ctrl_grp={'ctrl_0':0, 'ctrl_1':1, 'ctrl_2':2, 'ctrl_3':3}
trt_grp={'trt_2':0, 'trt_0':1, 'trt_9':2, 'trt_12':3}

## treatment 
df_trt_dd = df_plt[(df_plt.ctrl_trt_period.isin(trt_grp))& (df_plt.day>=15) & (df_plt.day<=29)]
df_trt_dd['trt_grp'] = df_trt_dd['ctrl_trt_period'].map(trt_grp)

## ctrl 
df_ctrl_dd = df_plt[((df_plt.ctrl_trt_period.isin(['ctrl_0','ctrl_2','ctrl_3'])) & (df_plt.day<=15)) |
                     ((df_plt.ctrl_trt_period.isin(['ctrl_1'])) & (df_plt.day>=15)& (df_plt.day<=29))]
df_ctrl_dd['trt_grp'] = df_ctrl_dd['ctrl_trt_period'].map(ctrl_grp)
df_ctrl_dd.loc[(df_ctrl_dd.ctrl_trt_period=='ctrl_2') & (df_ctrl_dd.day==15),'cancel_rate']= df_ctrl_dd.loc[(df_ctrl_dd.ctrl_trt_period=='ctrl_2') & (df_ctrl_dd.day==14),'cancel_rate'].values[0]

df_dd = pd.concat([df_trt_dd, df_ctrl_dd])
df_dd['day'] = df_dd.groupby('ctrl_trt_period').cumcount()
df_dd['day'] = -14 + df_dd['day']

fig = px.line(df_dd, x='day', y='cancel_rate', markers=True, title='diff in diff', color='ctrl_trt_period')
fig.show()



import statsmodels.formula.api as smf

## add grp features 
one_hot = pd.get_dummies(df_dd['trt_grp'], prefix='grp')
df_dd_fin = df_dd.join(one_hot)

# treatment and post features 
df_dd_fin[['treatment','post']] = 0
df_dd_fin.loc[df_dd_fin.ctrl_trt=='trt', 'treatment']=1
df_dd_fin.loc[df_dd_fin.day>=-7, 'post']=1

# interaction features 
df_dd_fin['grp0_post'] = df_dd_fin['grp_0'] * df_dd_fin['post']
# df_dd_fin['grp1_post'] = df_dd_fin['grp_1'] * df_dd_fin['post']
df_dd_fin['grp2_post'] = df_dd_fin['grp_2'] * df_dd_fin['post']
# df_dd_fin['grp3_post'] = df_dd_fin['grp_3'] * df_dd_fin['post']


# formula = 'cancel_rate ~ grp_0 + grp_1 + grp_2 + grp_3 + post + grp0_post +grp1_post + grp2_post + grp3_post'
formula = 'cancel_rate ~ grp_0 +  grp_2 +  post + grp0_post  + grp2_post '

model = smf.ols(formula=formula, data=df_dd_fin)
results = model.fit(cov_type='cluster', cov_kwds={'groups': df_dd_fin['trt_grp']})

# Print results
print(results.summary())



# COMMAND ----------

df_plt['expire_month']=pd.to_datetime(df_plt.expire_date).dt.to_period('M')
df_plt.groupby(by=['ctrl_trt_period','expire_month']).count()

## trt_0(batman): ctrl_0, trt_2 (hotds1):ctrl_0, trt_6 (love&death): ctrl_2, trt8: ctrl_2, trt_9: ctrl_2 (flash)


# COMMAND ----------

df_dd.groupby(by=['trt_grp','ctrl_trt_period']).sum()

# COMMAND ----------



# COMMAND ----------

date_premiere
date_start_ctr = ['2022-07-11','2022-09-17','2023-07-06','2024-04-01']
date_end_ctr = ['2022-08-06','2022-10-15','2023-07-24','2024-05-06']

# COMMAND ----------



df_plt['expire_year'] = pd.to_datetime(df_plt['expire_date']).dt.to_period('M')
df_plt.groupby(by=['expire_year','ctrl_trt_period']).count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1:1 title diff in diff

# COMMAND ----------

## trt_0(batman): ctrl_0, trt_2 (hotds1):ctrl_0, trt_6 (love&death): ctrl_2, trt8: ctrl_2, trt_9: ctrl_2 (flash)

# COMMAND ----------

## control: ## Diners, Drive-Ins, and Dives	: 2023-07-21	3K fv 

df_ctrl_dd = df_plt[(df_plt.ctrl_trt_period=='ctrl_3')& (df_plt.day>=0) & (df_plt.day<=15)]
df_ctrl_dd['day'] = df_ctrl_dd['day']-15
# df_ctrl_dd.loc[df_ctrl_dd.day==0, 'cancel_rate'] = 0.108
df_trt_dd = df_plt[(df_plt.ctrl_trt_period=='trt_9')& (df_plt.day>=16) & (df_plt.day<=30)]
df_trt_dd['day'] = df_trt_dd['day']-30

df_dd = pd.concat([df_ctrl_dd, df_trt_dd])
fig = px.line(df_dd, x='day', y='cancel_rate', markers=True, title='diff in diff', color='ctrl_trt_period')
fig.show()

import statsmodels.formula.api as smf

df_dd[['treatment','post']] = 0
df_dd.loc[df_dd.ctrl_trt=='trt', 'treatment']=1
df_dd.loc[df_dd.day>=-7, 'post']=1
df_dd['treat_post'] = df_dd['treatment'] * df_dd['post']

# Run DiD regression
model = smf.ols(formula='cancel_rate ~ treatment + post + treat_post', data=df_dd)
results = model.fit()

# Print results
print(results.summary())


df_dd.head()


# COMMAND ----------



# COMMAND ----------

import statsmodels.formula.api as smf

# Assuming your data is in a DataFrame called 'df'
# with columns: 'outcome', 'treatment', 'post', and any control variables

# Create interaction term
df['treat_post'] = df['treatment'] * df['post']

# Run DiD regression
model = smf.ols(formula='outcome ~ treatment + post + treat_post + control_var', data=df)
results = model.fit()

# Print results
print(results.summary())

# COMMAND ----------

trtgrp = [0,2,9,10,12,13] #removed groups with funky lines, and group with low churn (~7%. too low compared to contrl?)
ctrlgrp =[2] #['2022-05-19','2023-04-26', '2023-07-23']..  clean dates are ['2022-06-03','2023-05-01',2023-08-01'] 
##  ctrlgrp 3 has hardknocks in aug/8 which may have impact. 

# Find titles around 2022-06-19, 2023-0515, 2023-0815 
df_title[df_title.premiere_date>='2024-03-15'][50:150]
# ['Father of the Bride', 'Rick and Morty',	'90 Day: The Last Resort']
# fotb 2022-06-16, 2023-05-11, 2023-08-15		
## Diners, Drive-Ins, and Dives	: 2023-07-21	3K fv 



# COMMAND ----------

display(df_plt[(df_plt.ctrl_trt=='ctrl') & (df_plt.period.isin([3]))])

# COMMAND ----------

# tenure_df_list =[]

# ## control period 
# date_start_ctr = ['2022-02-09','2022-05-19','2023-04-26','2023-07-23','2024-04-27']
# date_end_ctr = ['2022-03-18','2022-07-12','2023-05-22','2023-08-29','2024-05-16']
# date_ctrl = list(zip(date_start_ctr, date_end_ctr))

# ## Plot churn for control & treatment periods 
# for tenure in [4]:
#     df_test = df_eligibles[(df_eligibles.tenure.between(tenure,tenure)) & (df_eligibles.region=='NORTH AMERICA')]
#     df_test = df_test.groupby(by=['tenure','tenure_bucket','expire_date']).sum().reset_index()

#     df_date_list = []
#     for date in date_ctr:
#         df_test_date = df_test[df_test.expire_date.between(date)]

# df_list = []
# for tenure in [4]:#[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]:
#     df_test = df_eligibles[(df_eligibles.tenure.between(tenure,tenure)) & (df_eligibles.region=='NORTH AMERICA')]
#     df_test.expire_date = pd.to_datetime(df_test.expire_date)
#     df_test = df_test.groupby(by=['tenure','tenure_bucket','expire_date']).sum().reset_index()
#     df_title.premiere_date = pd.to_datetime(df_title.premiere_date)

#     df_merged = df_test.merge(df_title[(df_title.tfv90>=200000)], right_on=['premiere_date'], left_on=['expire_date'], how='left').fillna(0)
    


# COMMAND ----------

# MAGIC %md
# MAGIC ### Find control dates

# COMMAND ----------

## Find periods for control, out side of +/- 30 days of big titles (fv>200000)
fv_threshold
df_dates = df_test.merge(df_title[(df_title.tfv7>=30000)], right_on=['premiere_date'], left_on=['expire_date'], how='left').fillna(0)
date_premiere = df_dates[df_dates.title_name!=0].expire_date.tolist()
pre_30d = [dt + timedelta(days=-30) for dt in date_premiere]
post_30d = [dt + timedelta(days=14) for dt in date_premiere]


full_range_start = pd.datetime(2022, 1, 1)
full_range_end = pd.datetime(2024, 7, 1)

# Generate all dates within the full range
all_dates = []
current_date = full_range_start
while current_date <= full_range_end:
    all_dates.append(current_date)
    current_date += timedelta(days=1)

# Mark dates that fall within any of the date ranges
dates_in_ranges = set()
for start, end in zip(pre_30d, post_30d):
    current_date = start
    while current_date <= end:
        dates_in_ranges.add(current_date)
        current_date += timedelta(days=1)

# Find dates that do not fall within any range
dates_not_in_ranges = [date for date in all_dates if date not in dates_in_ranges]

# date_start_ctr = ['2022-03-22','2022-07-11','2022-09-17','2023-07-08','2024-04-01']
# date_end_ctr = ['2022-04-03','2022-08-06','2022-10-15','2023-07-24','2024-05-06']

date_start_ctr = ['2022-02-11','2022-06-14','2022-09-17','2023-02-01','2024-04-01']
date_end_ctr = ['2022-03-18','2022-07-09','2022-10-02','2023-02-23','2024-04-20']

date_start_ctr_short = ['2022-07-11','2022-09-30','2023-07-06','2024-03-01']
date_end_ctr_short = ['2022-07-27','2022-10-15','2023-07-21','2024-04-15']

dates_not_in_ranges
## 2022-02-09 to 2022-03-18
## 2022-05-19 to 2022-07-21
## 2023-04-26 to 2023-05-22
## 2023-07-23 to 2023-08-29
## 2024-04-17 to 2024-05-16

# COMMAND ----------

df_title[(df_title.premiere_date>='2024-01-01')& (df_title.tfv7>=5000)].head(50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initial EDA

# COMMAND ----------

custom_order = ['cancel_rate_21_14', 'cancel_rate_14_7', 'cancel_rate_7_0', 'cancel_rate_0_7']
df = df_melted_sum.copy()
df['days'] = pd.Categorical(df['days'], categories=custom_order, ordered=True)

# Sort the DataFrame based on the custom order
df


# COMMAND ----------

df_sum[df_sum.title_name=='The Last of Us']

# COMMAND ----------

df_merged = df_test.merge(df_title[(df_title.tfv90>=200000)], right_on=['premiere_date'], left_on=['expire_date'], how='left')
df_merged.head()

# COMMAND ----------

df_tenure[df_tenure.expire_date>='2023-12-01']

# COMMAND ----------

df_merged[df_merged.premiere_date>='2023-12-01']

# COMMAND ----------

df_sum[df_sum.title_name=='Barbie']

# COMMAND ----------

df_test = df_eligibles[(df_eligibles.tenure.between(4,6))& (df_eligibles.region=='NORTH AMERICA')]

data = []
df_list=[]
#'2023-01-15', TLOU
#'True Detective', '2024-01-14',
for date,title in zip(['2022-04-18','2022-08-21','2023-05-23','2023-06-04','2023-06-22','2023-12-15','2024-02-27','2024-03-08','2024-05-21','2024-06-16'], ['Batman','HOTD','Shazam','The Idol','And Just Like That','Barbie','Aquaman','Wonka','Dune','HOTD2']):
    pre_21d = (pd.datetime.strptime(date, '%Y-%m-%d') - timedelta(days=21)).strftime('%Y-%m-%d')
    pre_14d = (pd.datetime.strptime(date, '%Y-%m-%d') - timedelta(days=14)).strftime('%Y-%m-%d')
    pre_7d = (pd.datetime.strptime(date, '%Y-%m-%d') - timedelta(days=7)).strftime('%Y-%m-%d')
    post_7d = (pd.datetime.strptime(date, '%Y-%m-%d') + timedelta(days=7)).strftime('%Y-%m-%d')


    df_test[(df_test.expire_date>=pre_21d) & (df_test.expire_date<pre_14d)]
    cancel_21_14= df_test[(df_test.expire_date>=pre_21d) & (df_test.expire_date<pre_14d)].cancel_rate.mean()
    cancel_14_7= df_test[(df_test.expire_date>=pre_14d) & (df_test.expire_date<pre_7d)].cancel_rate.mean()
    cancel_7_0= df_test[(df_test.expire_date>=pre_7d) & (df_test.expire_date<date)].cancel_rate.mean()
    cancel_0_7=df_test[(df_test.expire_date>=date) & (df_test.expire_date<post_7d)].cancel_rate.mean()

    data=[{'date':date,
                 'title':title, 
                 '21_14': cancel_21_14,
                 '14_7': cancel_14_7, 
                 '7_0': cancel_7_0, 
                 '0_7':cancel_0_7}]
    
    df_temp = pd.DataFrame(data)
    df_list.append(df_temp)

df_plt = pd.concat(df_list)


df_melted = df_plt.melt(id_vars=['title'], value_vars=['21_14', '14_7', '7_0','0_7'], var_name='days', value_name='avg_churn')

df_melted['days'] = df_melted['days'].astype(str)

display(df_melted.head())
# Create a line plot using plotly.express
fig = px.line(df_melted, x='days', y='avg_churn', color='title', markers=True, title='')
fig.show()
# fig = px.line(df_plt, x='index', y='0', color='tenure')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Weekly

# COMMAND ----------

#HOTD : 8.21- 10.23
#TLOU : 01/15- 03/12
#Dune : 2024-05-21

grpby=['expire_weekend','tenure']
df_sum= df_eligibles[df_eligibles.region=='NORTH AMERICA'].groupby(by=grpby).sum().reset_index()
df_sum['cancel_rate'] = df_sum['cancels']/df_sum['eligibles']

import plotly.express as px
fig = px.line(df_sum, x='expire_weekend', y='cancel_rate', color='tenure')
fig.add_shape(
    type="line",
    x0='2022-08-21', y0=df_sum['cancel_rate'].min(), x1='2022-08-21', y1=df_sum['cancel_rate'].max(),
    line=dict(color="Black", width=2)
)

fig.add_shape(
    type="line",
    x0='2023-01-15', y0=df_sum['cancel_rate'].min(), x1='2023-01-15', y1=df_sum['cancel_rate'].max(),
    line=dict(color="Black", width=2)
)

fig.add_shape(
    type="line",
    x0='2024-05-21', y0=df_sum['cancel_rate'].min(), x1='2024-05-21', y1=df_sum['cancel_rate'].max(),
    line=dict(color="Black", width=2)
)
fig.show()

# COMMAND ----------

## Dip on release weekend 
print('hotd 08/21')
display(df_sum[(df_sum.expire_weekend>='2022-08-01') & (df_sum.expire_weekend<='2022-08-30') & (df_sum.tenure.between(6,6))].sort_values(by=['tenure','expire_weekend']))

print('white lotus 10/30')
display(df_sum[(df_sum.expire_weekend>='2022-10-10') & (df_sum.expire_weekend<='2022-11-10') & (df_sum.tenure.between(6,6))].sort_values(by=['tenure','expire_weekend']))

print('and just like that 06/22')
display(df_sum[(df_sum.expire_weekend>='2023-06-10') & (df_sum.expire_weekend<='2023-07-10') & (df_sum.tenure.between(6,6))].sort_values(by=['tenure','expire_weekend']))

print('dune 05/21')
display(df_sum[(df_sum.expire_weekend>='2024-05-05') & (df_sum.expire_weekend<='2024-05-30') & (df_sum.tenure.between(6,6))].sort_values(by=['tenure','expire_weekend']))

print('hotd 2 06/16')
display(df_sum[(df_sum.expire_weekend>='2024-06-05') & (df_sum.expire_weekend<='2024-07-10') & (df_sum.tenure.between(6,6))].sort_values(by=['tenure','expire_weekend']))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Daily

# COMMAND ----------

#HOTD : 8.21- 10.23
#TLOU : 01/15- 03/12
# and just like that : 2023-0622 
# White lotus S2: 2022-10-30
#Dune : 2024-05-21
#hotd2 : 2024-06-16

## DAILY 
## Dip on release weekend 
col = ['expire_date','tenure','cancel_rate']


df_eligibles['cancel_rate'] = df_eligibles['cancels'] /df_eligibles['eligibles']


display(df_eligibles[(df_eligibles.expire_date.between('2022-08-01','2022-08-30')) & (df_eligibles.tenure.between(4,4))& (df_eligibles.region=='NORTH AMERICA')][col].sort_values(by=['tenure','expire_date']))

df_eligibles['cancel_rate'] = df_eligibles['cancels'] /df_eligibles['eligibles']
display(df_eligibles[(df_eligibles.expire_date.between('2022-06-01','2022-06-30')) & (df_eligibles.tenure.between(4,4))& (df_eligibles.region=='NORTH AMERICA')][col].sort_values(by=['tenure','expire_date']))

df_eligibles['cancel_rate'] = df_eligibles['cancels'] /df_eligibles['eligibles']
display(df_eligibles[(df_eligibles.expire_date.between('2022-10-10','2023-11-10')) & (df_eligibles.tenure.between(4,4))& (df_eligibles.region=='NORTH AMERICA')][col].sort_values(by=['tenure','expire_date']))


df_eligibles['cancel_rate'] = df_eligibles['cancels'] /df_eligibles['eligibles']
display(df_eligibles[(df_eligibles.expire_date.between('2024-05-10','2024-05-30')) & (df_eligibles.tenure.between(4,4))& (df_eligibles.region=='NORTH AMERICA')][col].sort_values(by=['tenure','expire_date']))

df_eligibles['cancel_rate'] = df_eligibles['cancels'] /df_eligibles['eligibles']
display(df_eligibles[(df_eligibles.expire_date.between('2024-06-05','2024-06-30')) & (df_eligibles.tenure.between(4,4))& (df_eligibles.region=='NORTH AMERICA')][col].sort_values(by=['tenure','expire_date']))

# COMMAND ----------

#HOTD : 8.21- 10.23
# White lotus S2: 2022-10-30
#TLOU : 01/15- 03/12
# and just like that : 2023-0622 
#Dune : 2024-05-21

grpby=['expire_date','tenure']
df_sum= df_eligibles[df_eligibles.region=='NORTH AMERICA'].groupby(by=grpby).sum().reset_index()
df_sum['cancel_rate'] = df_sum['cancels']/df_sum['eligibles']

import plotly.express as px
fig = px.line(df_sum, x='expire_date', y='cancel_rate', color='tenure')
fig.add_shape(
    type="line",
    x0='2022-08-21', y0=df_sum['cancel_rate'].min(), x1='2022-08-21', y1=df_sum['cancel_rate'].max(),
    line=dict(color="Black", width=2)
)

fig.add_shape(
    type="line",
    x0='2023-06-04', y0=df_sum['cancel_rate'].min(), x1='2023-06-04', y1=df_sum['cancel_rate'].max(),
    line=dict(color="Black", width=2)
)
fig.add_shape(
    type="line",
    x0='2023-06-22', y0=df_sum['cancel_rate'].min(), x1='2023-06-22', y1=df_sum['cancel_rate'].max(),
    line=dict(color="Black", width=2)
)

fig.add_shape(
    type="line",
    x0='2024-05-21', y0=df_sum['cancel_rate'].min(), x1='2024-05-21', y1=df_sum['cancel_rate'].max(),
    line=dict(color="Black", width=2)
)
fig.add_shape(
    type="line",
    x0='2024-06-16', y0=df_sum['cancel_rate'].min(), x1='2024-06-16', y1=df_sum['cancel_rate'].max(),
    line=dict(color="Black", width=2)
)

fig.show()

# COMMAND ----------

## MONTHLY
grpby=['expire_mth','tenure']
df_sum= df_eligibles[df_eligibles.region=='NORTH AMERICA'].groupby(by=grpby).sum().reset_index()
df_sum['cancel_rate'] = df_sum['cancels']/df_sum['eligibles']

import plotly.express as px
fig = px.line(df_sum, x='expire_mth', y='cancel_rate', color='tenure')
fig.show()

# COMMAND ----------

# M1 Churn
df_churn = pd.read_sql("""
select a.region, a.title_name, a.imdb_id, a.forecast_category, a.season_number, a.premiere_date, a.finale_date,
c.fv_post_7, c.fv_post_30,
c.fv_post_7_pay1, c.fv_post_30_pay1,
c.fv_post_7_series, c.fv_post_30_series,
b.lob,
sum(b.num_subs) as fv, 
sum(b.eligible_churn * b.churn_rate)/sum(b.eligible_churn) as churn_rate
from forecasting_fv_summary a
left join max_ltv_content_past b using (title_name, region, season_number)
left join forecasting_fv_pre_post c using (title_name, region, season_number)
where 
period_number = 3 
and provider = 'Direct'
and sub_region = 'US'
and region = 'NORTH AMERICA'
and ad_strategy = 'ad_free'
and payment_period = 'PERIOD_MONTH'
and signup_offer = 'none'
and b.paid_start_date between a.premiere_date and a.premiere_date + 30
and imdB_id in (select imdb_id from forecasting_premieres)
and a.premiere_date between '2022-01-01' and '2023-12-31'
group by all
                       """, con)



df_churn.columns = df_churn.columns.str.lower()
df_churn.head()

# COMMAND ----------

df_plt

# COMMAND ----------

df_plt = df_churn.query("fv>=1000 & forecast_category=='series' & lob=='HBO Originals'")
df_plt['ln_fv_post_7'] = np.log(df_plt['fv_post_7'])
df_plt['ln_fv_post_30'] = np.log(df_plt['fv_post_30'])
fig = px.scatter(df_plt, x='ln_fv_post_7', y='churn_rate', hover_name='title_name', trendline='ols')
fig.show()
fig = px.scatter(df_plt, x='ln_fv_post_30', y='churn_rate', trendline='ols', hover_name='title_name')
fig.show()

# COMMAND ----------

post='series'

df_plt = df_churn.query("fv>=1000 & forecast_category=='series'")
df_plt['ln_fv_post_7'] = np.log(df_plt[f'fv_post_7'])
df_plt['ln_fv_post_30'] = np.log(df_plt[f'fv_post_30'])
fig = px.scatter(df_plt, x='ln_fv_post_7', y='churn_rate',hover_name='title_name', trendline='ols')
fig.show()
fig = px.scatter(df_plt, x='ln_fv_post_30', y='churn_rate', hover_name='title_name',trendline='ols')
fig.show()

# COMMAND ----------

post='series'

df_plt = df_churn.query("fv>=1000")
df_plt['ln_fv_post_7'] = np.log(df_plt[f'fv_post_7_{post}'])
df_plt['ln_fv_post_30'] = np.log(df_plt[f'fv_post_30_{post}'])
fig = px.scatter(df_plt, x='ln_fv_post_7', y='churn_rate', trendline='ols')
fig.show()
fig = px.scatter(df_plt, x='ln_fv_post_30', y='churn_rate', trendline='ols')
fig.show()

# COMMAND ----------

df[df.title_name.str.contains('Dragon')]

# COMMAND ----------

# Succession Mar 26.. finale= may28   is this first view LTV? 
df = pd.read_sql("""
                 select 
                 * 
                 from max_dev.workspace.max_ltv_content_past 
                 where region='NORTH AMERICA'
                 and sub_region='US'
                 and title_name='Succession'
                 and season_number=4
                 """, con)
df.columns = df.columns.str.lower()
df.paid_start_date = pd.to_datetime(df.paid_start_date)
df_filter=df[(df.ad_strategy=='premium_ad_free') & (df.provider=='Direct') & (df.signup_offer=='none')]
display(df_filter)
display(df_filter[df_filter.paid_start_date=='2023-06-01'])
# df_sum = df.groupby(by=['title_name','finale_cohort','cohort_type','period_number']).mean().reset_index()
# df_sum[df_sum.title_name=='Winning Time: The Rise of the Lakers Dynasty'].head()


# paid start date 2023-05-21, period= 4. (2023-08-01..)..
#  people who watched succession as first view, and watched until finale  w/ paid start date= 20230521. their churn over different periods.  

# COMMAND ----------

df.paid_start_date.unique()

# COMMAND ----------

# M1 Churn
df_churn = pd.read_sql("""
select a.region, a.title_name, a.imdb_id, a.forecast_category, a.season_number, a.premiere_date, a.finale_date,
c.fv_post_7, c.fv_post_30,
b.lob,
sum(b.num_subs) as fv, 
sum(b.eligible_churn * b.churn_rate)/sum(b.eligible_churn) as churn_rate
from forecasting_fv_summary a
left join max_ltv_content_past b using (title_name, region, season_number)
left join forecasting_fv_pre_post c using (title_name, region, season_number)
where 
period_number = 1
and provider = 'Direct'
and sub_region = 'US'
and region = 'NORTH AMERICA'
and ad_strategy = 'ad_free'
and payment_period = 'PERIOD_MONTH'
and signup_offer = 'none'
and b.paid_start_date between a.premiere_date and a.premiere_date + 30
and imdB_id in (select imdb_id from forecasting_premieres)
and a.premiere_date between '2022-01-01' and '2023-12-31'
group by 1,2,3,4,5,6,7,8,9,10
                       """, con)
