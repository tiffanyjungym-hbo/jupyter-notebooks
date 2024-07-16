# Databricks notebook source
import boto3
import datetime as dt
import json
import numpy as np
import pandas as pd
import io

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.options.display.float_format = '{:,.2f}'.format

import plotly.express as px
import plotly.graph_objects as go


from scipy.stats import percentileofscore
from scipy.optimize import curve_fit

def get_df_test(df_test):
    df_test['tenure_months'] = df_test['sub_month']
    df_test['monthly_title_viewed'] = np.where(df_test['tenure_months']>1, df_test['titles_viewed']/2, df_test['titles_viewed'])
    df_test['monthly_hours_viewed'] = np.where(df_test['tenure_months']>1, df_test['hours_viewed']/2, df_test['hours_viewed'])
    user_total = df_test.groupby(['user_id'])['monthly_title_viewed'].transform('sum')
    df_test['frc'] = df_test['monthly_title_viewed'] / user_total
    
    df_test['program_type'] = np.where((df_test.program_type=='original') & (df_test.old_new=='library'), 'acquired', df_test.program_type)
    df_test = df_test[df_test.tenure_months>2]
    df_test = df_test.fillna(0)
    return(df_test)

def get_df_60_h(list_df):
    df_list=[]
    num=0
    for df_test in list_df:
        df_test['num_df'] = num
        df_list.append(df_test)
        num=num+1
    return(df_list)

def exponential_decay(x, a, b,c):
    return a * np.exp(b * x) + c

def exponential_decay_slope(x, a, b):
    return a * b*np.exp(b * x)

def fit_exponential(x_data, y_data, p0, param_bounds):
    x_fit = np.linspace(0, x_data.max(), 100)   
    params, _ = curve_fit(exponential_decay, np.array(x_data), y_data, p0, bounds=param_bounds)
    return x_fit, params


def get_equal_churn_bin(df_in, grpby):
    df = df_in[df_in.monthly_hours_viewed<=60]
    df = df.groupby(by=['user_id','sub_month']+ grpby +['is_cancel']).sum().reset_index()
    nbins = int(df.monthly_title_viewed.max())
    df['title_viewed_bin'] = pd.cut(df['monthly_title_viewed'], 
                                    np.linspace(0,nbins,2*nbins+1))
    df['title_viewed_bin'] = df['title_viewed_bin'].apply(lambda x: x.right)
    df['churn'] = 1*df['is_cancel']  
    
    df_bin = df.groupby(['title_viewed_bin']+grpby).agg({'churn':'mean', 'user_id':'count',
                                                         'is_cancel':'sum','monthly_title_viewed':'sum'}).reset_index()
    return(df_bin)

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


# COMMAND ----------

expiration_month = '2023-12-01'
df_60_00 = spark.sql('''
     WITH new_library AS (
      SELECT s.*
                     , CASE WHEN r.recency_window_end >= expiration_month
                        THEN 'current'
                        ELSE 'library'
                        END AS old_new
      FROM bolt_cus_dev.bronze.cip_churn_user_stream60d_genpop_savod s
      left join bolt_cus_dev.bronze.cip_recency_title_series_level_offering_table r
         on s.ckg_program_id = r.ckg_program_id
      where s.expiration_month = '{expiration_month}'
      )

      SELECT s.user_id ::STRING
            , s.profile_id ::STRING
            , is_cancel
            , is_voluntary
            , sku
            , mode(seg.entertainment_segment_lifetime) as entertainment_segment_lifetime
            , min(sub_month)::STRING as sub_month
            , sum(hours_viewed)::STRING as hours_viewed 
            , count(distinct ckg_series_id) as titles_viewed
            , count(distinct (CASE WHEN old_new = 'current' THEN ckg_series_id else null END))  as new_titles_viewed
            , count(distinct (CASE WHEN old_new = 'library' THEN ckg_series_id else null END))  as library_titles_viewed
            , case when is_voluntary = 1 and is_cancel = 1 then 1 else 0 end as is_cancel_vol
            , case when is_voluntary = 0 and is_cancel = 1 then 1 else 0 end as is_cancel_invol
      FROM new_library s
      LEFT JOIN bolt_growthml_int.gold.max_content_preference_v3_segment_assignments_360_landing_table seg
                        on seg.PROFILE_ID = s.profile_id
      GROUP BY ALL
'''.format(expiration_month = expiration_month)
                     ).toPandas()

df_60_00.head()

# COMMAND ----------

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
        # annotations=[
        # dict(
        #     x=x_med+0.2,  # x-coordinate for the text
        #     y=y_med+0.01,  # y-coordinate for the text
        #     # text='{:.2f}, {:.2f}'.format(x_med, y_med),  # the text to display
        #     showarrow=False,  # disable arrow for the annotation
        #     xanchor='left',
        #     font=dict(
        #         family='Arial',  # specify font family
        #         size=18,  # specify font size
        #         color='black'  # specify font color
        #     )
        # )
    # ]
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
        fig = get_simple_plot_multiple(df_i, 'title_viewed_bin', 'churn', x_fit, y_fit, params, f'{title}')
    else:
        y_med = exponential_decay(x_med, a_fit, b_fit, c_fit)
        y_med_slope = exponential_decay_slope(x_med, a_fit, b_fit)
        print(x_med)
        print('average churn: ' + str('{:.3f}'.format(y_med)))
        print('slope: ' + str('{:.3f}'.format(y_med_slope*100))+'%')
        fig = get_simple_plot_multiple_dot(df_i, 'title_viewed_bin', 'churn', x_fit, y_fit, params, x_med, np.array(y_med), f'{title}')
    # display(df_i.head())
    param_dic[title] = params
    return fig, params

# def get_equal_churn_bin(df_in, grpby):
#     df = df_in[df_in.monthly_hours_viewed<=60]
#     df = df.groupby(by=['user_id','sub_month']+ grpby +['is_cancel']).sum().reset_index()
#     nbins = int(df.monthly_title_viewed.max())
#     df['title_viewed_bin'] = pd.cut(df['monthly_title_viewed'], 
#                                     np.linspace(0,nbins,20))
#     df['title_viewed_bin'] = df['title_viewed_bin'].apply(lambda x: x.right)
#     df['churn'] = 1*df['is_cancel']  
    
#     df_bin = df.groupby(['title_viewed_bin']+grpby).agg({'churn':'mean', 'user_id':'count',
#                                                          'is_cancel':'sum','monthly_title_viewed':'sum'}).reset_index()
#     return(df_bin)





# COMMAND ----------

####
# df_60_00 = pd.read_parquet(file_path+'churn_user_stream60d_segmented_20240327.parquet')
df_60_00['tenure_months'] = df_60_00['sub_month']
df_60_00.tenure_months = df_60_00.tenure_months.astype('int')
df_60_00.titles_viewed = df_60_00.titles_viewed.astype('float')
df_60_00.hours_viewed = df_60_00.hours_viewed.astype('float')

df_60_00['monthly_title_viewed'] = np.where(df_60_00['tenure_months']>1, df_60_00['titles_viewed']/2, df_60_00['titles_viewed'])
df_60_00['monthly_title_viewed'] = np.where(df_60_00['tenure_months']>1, df_60_00['titles_viewed']/2, df_60_00['titles_viewed'])
df_60_00['monthly_hours_viewed'] = np.where(df_60_00['tenure_months']>1, df_60_00['hours_viewed']/2, df_60_00['hours_viewed'])

df_60_00['monthly_hours_viewed']  = df_60_00['monthly_hours_viewed'].astype(float)

## Bucketing to tenure mths 
df_60_00['sub_month'] = df_60_00['sub_month'].astype(int)
mapping_dict = {1:'month_1', 2:'month_2', 3:'month_3', 4:'month_4_to_6', 5:'month_4_to_6', 6:'month_4_to_6', 7:'month_7_to_12', 8:'month_7_to_12', 9:'month_7_to_12', 10:'month_7_to_12', 11:'month_7_to_12', 12:'month_7_to_12'}
df_60_00['tenure_bucket'] = df_60_00['sub_month'].map(mapping_dict).fillna('month_13+')


# COMMAND ----------

# !pip install tables
# !pip install pandas h5py tables
# df_60_00.to_hdf('df_raw_tenure_churn_plt_0624.h5', key='df', mode='w')
# df_60_00.to_parquet('df_raw_tenure_churn_plt_0624.parquet', engine='pyarrow', compression='snappy')
df_60_00 = pd.read_parquet('/Workspace/Repos/tiffany.jung@wbd.com/jupyter-notebooks/2404_content_schedule/data/df_raw_tenure_churn_plt_0624.parquet')


# COMMAND ----------


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

## Plot by tenure 

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
    # fig.show()
    # return fig

fig = px.scatter(width=500, height=400)

color=['red','blue','green']
for m in ['month_1','month_2','month_3']:#df_60_00.tenure_bucket.unique():
    df_plt= df_60_00[df_60_00['tenure_bucket'] == m]
    df_60_t = df_plt.groupby(by=['user_id','is_cancel_vol','sub_month'])[['monthly_title_viewed', 'monthly_hours_viewed']].sum().reset_index()
    df_60_t['is_cancel'] = df_60_t['is_cancel_vol']
    if m == 'month_1':
        df_60_s = get_churn_bin_mth1(df_60_t, [])
    else:
        df_60_s = get_churn_bin(df_60_t, [])
    med_x= df_60_t.monthly_title_viewed.median()
    # fig, params = get_churn_plot_simple(df_60_s[df_60_s['title_viewed_bin']<15], 
    #                                     m, {}, np.array(med_x))
    # slope = get_churn_slope_plot_simple(df_60_s, , params, np.array(med_x))

    ## get_churn_slope_plot_simple
    df_i = df_60_s[df_60_s['title_viewed_bin']<15].copy()
    title=m
    param_dic={}
    x_med=0
    df_i = df_i[df_i.is_cancel>=20].copy()

    x_var = df_i.title_viewed_bin
    y_data = df_i.churn
    p0 = [0.5, -0.1, 0.01] 
    param_bounds = ([0, -0.8, 0.01], [np.inf, -0.1, np.inf])

    x_fit, params = fit_exponential(x_var, y_data, p0, param_bounds)
    a_fit, b_fit, c_fit = params
    y_fit = exponential_decay(x_fit, a_fit, b_fit, c_fit)

    ###get simple_plot_multiple     
    df_plt = df_i.copy()
    x='title_viewed_bin'
    y='churn' 
       
    a_fit, b_fit, c_fit = params
    annotation_x_loc = 50
    annotation_y_loc = y_fit.min() +(y_fit.max()  - y_fit.min() )/2 
        # ig.add_scatter(x=df_region['value1'], y=df_region['value2'], mode='markers', name=f'Region {region}')

    fig.add_scatter(x=df_plt[x], 
                  y=df_plt[y], mode='markers',name=m)
    fig.add_scatter(x=x_fit, y=y_fit, name=m)

fig.update_layout(
    template='simple_white',
    showlegend=False,
    xaxis=dict(range=[0,50]),
    xaxis_title='Number of titles watched',
    yaxis_title='Churn'
    # annotations=[
    # dict(
    #     x=annotation_x_loc,  # x-coordinate for the text
    #     y=annotation_y_loc,  # y-coordinate for the text
    #     # text='y= {:.2f} * e^({:.2f} * title_viewed) + {:.2f}'.format(a_fit, b_fit, c_fit),  # the text to display
    #     showarrow=False,  # disable arrow for the annotation
    #     xanchor='right',
    #     font=dict(
    #         family='Arial',  # specify font family
    #         size=18,  # specify font size
    #         color='black'  # specify font color
    #     )
    # )

    ) 

    # display(df_i.head())
    # param_dic[title] = params
fig.show()


# COMMAND ----------

df_plt[x_fit]

# COMMAND ----------

## Plot by tenure 

for m in ['month_1','month_2','month_3']:#df_60_00.tenure_bucket.unique():
    df_plt= df_60_00[df_60_00['tenure_bucket'] == m]
    df_60_t = df_plt.groupby(by=['user_id','is_cancel_vol','sub_month'])[['monthly_title_viewed', 'monthly_hours_viewed']].sum().reset_index()
    df_60_t['is_cancel'] = df_60_t['is_cancel_vol']
    if m == 'month_1':
        df_60_s = get_churn_bin_mth1(df_60_t, [])
    else:
        df_60_s = get_churn_bin(df_60_t, [])
    med_x= df_60_t.monthly_title_viewed.median()
    fig, params = get_churn_plot_simple(df_60_s[df_60_s['title_viewed_bin']<15], 
                                        m, {}, np.array(med_x))
    # slope = get_churn_slope_plot_simple(df_60_s, , params, np.array(med_x))



# COMMAND ----------

import numpy as np
-0.014*np.exp(-0.33*2.5)*1

# COMMAND ----------

dy= 0.039 * np.exp(-0.42 * 4) ## slow per title %view
dy*0.25*100  

# COMMAND ----------

param_tenure_dict = {}
med_tenure_dict = {}

df_60_00['sub_month'] = df_60_00['sub_month'].astype(int)
mapping_dict = {1:'month_1', 2:'month_2', 3:'month_3', 4:'month_4_to_6', 5:'month_4_to_6', 6:'month_4_to_6', 7:'month_7_to_12', 8:'month_7_to_12', 9:'month_7_to_12', 10:'month_7_to_12', 11:'month_7_to_12', 12:'month_7_to_12'}
df_60_00['tenure_bucket'] = df_60_00['sub_month'].map(mapping_dict).fillna('month_13+')


for m in df_60_00.tenure_bucket.unique():
    df_seg_amw= df_60_00[df_60_00['tenure_bucket'] == m]
    df_60_t = df_seg_amw.groupby(by=['user_id','is_cancel','sub_month'])[['monthly_title_viewed', 'monthly_hours_viewed']].sum().reset_index()
    df_60_s = get_churn_bin(df_60_t, [])

    med_x= df_60_t.monthly_title_viewed.median()
    fig, params = get_churn_plot_simple(df_60_s[df_60_s['title_viewed_bin']<15], 
                                        m, param_tenure_dict, np.array(med_x))
    med_tenure_dict[m] = med_x
    # break

# COMMAND ----------


