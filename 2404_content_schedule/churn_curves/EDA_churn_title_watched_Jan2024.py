# Databricks notebook source
# !pip install snowflake-connector-python
# !pip install sympy
# dbutils.library.restartPython()

# COMMAND ----------

import boto3
import datetime as dt
import json
import numpy as np
import pandas as pd
import io
from scipy.optimize import curve_fit
# import snowflake.connector
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.options.display.float_format = '{:,.2f}'.format
file_path = '/Workspace/Users/jeni.lu@wbd.com/Retention/files/'

# COMMAND ----------

import plotly.express as px
import plotly.graph_objects as go

# COMMAND ----------


from scipy.stats import percentileofscore


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

# DBTITLE 1,Plot By Content Category
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
                  width=1000, height=400)
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
            text='y= {:.2f} * e^({:.2f} * title_viewed) + {:.2f}'.format(a_fit, b_fit, c_fit),  # the text to display
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
        xaxis=dict(range=[0,15]),
        annotations=[
        dict(
            x=x_med+0.2,  # x-coordinate for the text
            y=y_med+0.01,  # y-coordinate for the text
            text='{:.2f}, {:.2f}'.format(x_med, y_med),  # the text to display
            showarrow=False,  # disable arrow for the annotation
            xanchor='left',
            font=dict(
                family='Arial',  # specify font family
                size=18,  # specify font size
                color='black'  # specify font color
            )
        )
    ]
) 
    fig.show()
    return fig



def get_churn_plot_simple(df_i, title, param_dic, x_med=0):
    # df_i = df_i[df_i.is_cancel>=10]
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



def get_simple_plot_dot(df_plt, x, y, x_fit, y_fit, params, x_med, y_med, title=''):
    if title=='':
        
        title = f'{y} vs {x}'
       
    a_fit, b_fit, c_fit = params
    print('y= {:.2f} * e^({:.2f} * title_viewed) + {:.2f}'.format(a_fit, b_fit, c_fit))
    print('y= {:.3f} * e^({:.2f} * title_viewed)'.format(a_fit*b_fit,b_fit))
    annotation_x_loc = 50
    annotation_y_loc = y_fit.min() +(y_fit.max()  - y_fit.min() )/2 
        
    fig = px.line(x=x_fit, 
                  y=y_fit, 
                  title=title,
                  width=500, height=400)
    fig.add_scatter( 
              x=x_med, 
              y=y_med,
                mode='markers',
            marker=dict(size=14, color='red', line=dict(color='black', width=2)))

    fig.update_layout(
        template='simple_white',
        showlegend=False,
        xaxis=dict(range=[0,15]),
        xaxis_title= "title_viewed_bin",
        yaxis_title= "Change in churn rate (slope)",
        annotations=[
        dict(
            x=x_med+0.25,  # x-coordinate for the text
            y=y_med+0.0005,  # y-coordinate for the text
            text='{:.2f}, {:.4f}'.format(x_med, y_med),  # the text to display
            showarrow=False,  # disable arrow for the annotation
            xanchor='left',
            font=dict(
                family='Arial',  # specify font family
                size=18,  # specify font size
                color='black'  # specify font color
            )
        )
    ]
) 
    fig.show()
    return fig

def get_churn_slope_plot_simple(df_i, title, params, x_med=0):
    df_i = df_i[df_i.is_cancel>=20]
#         display(df_i.tail(5))

    x_var = df_i.title_viewed_bin
    x_fit = np.linspace(0, x_var.max(), 100)   
    a_fit, b_fit, c_fit = params
    y_fit = exponential_decay_slope(x_fit, a_fit, b_fit)
    
    y_med = exponential_decay_slope(x_med, a_fit, b_fit)
    print(x_med)
    print(y_med)
    fig = get_simple_plot_dot(df_i, 'title_viewed_bin', 'churn', x_fit, y_fit, params, x_med, np.array(y_med), f'{title}')
    display(df_i.head())
    param_dict[title] = params
    return fig


# COMMAND ----------

df_60_00 = pd.read_parquet(file_path+'churn_user_stream60d_segmented_20240103.parquet')

# COMMAND ----------

df_60_00=get_df_test(df_60_00)
df_list = get_df_60_h([df_60_00]) #, df_60_0, df_60_1, df_60_2])
df_60 = pd.concat(df_list)
display(df_60.head())

# COMMAND ----------

param_dict = {}
med_dict = {}

# COMMAND ----------

seg_name = 'gen_pop'
df_60['monthly_hours_viewed'] = df_60['monthly_hours_viewed'].astype('float')
df_60['monthly_title_viewed'] = df_60['monthly_title_viewed'].astype('float')
df_60_t = df_60.groupby(by=['user_id','is_cancel','sub_month'])[['monthly_hours_viewed', 'monthly_title_viewed']].sum().reset_index()

# COMMAND ----------

df_60_s = get_churn_bin(df_60_t, [])
## Get median 
med_x= df_60_t.monthly_title_viewed.median()

# Plot the Churve 
fig, params = get_churn_plot_simple(df_60_s[(df_60_s['title_viewed_bin']<15)], 
                                    seg_name, param_dict, np.array(med_x))
med_dict[seg_name] = med_x

# COMMAND ----------

df_60_s = get_equal_churn_bin(df_60_t, [])
## Get median 
med_x= df_60_t.monthly_title_viewed.median()

# Plot the Churve 
fig, params = get_churn_plot_simple(df_60_s[df_60_s['title_viewed_bin']<15], 
                                    seg_name, param_dict, np.array(med_x))
med_dict[seg_name] = med_x

# COMMAND ----------

# MAGIC %md
# MAGIC # BY Segment

# COMMAND ----------

segment_info = spark.sql('''
SELECT a.entertainment_segment_lifetime, count(distinct a.user_id) as user_count
FROM  bolt_growthml_int.gold.max_content_preference_v3_segment_assignments_360_landing_table a
join bolt_cus_dev.bronze.user_retain_churn_list_test_wbd_max b
ON a.user_id = b.user_id 
where cycle_expire_date between '2024-01-01' and '2024-01-31'
group by all
order by user_count desc        
                  ''').toPandas()

# COMMAND ----------

segment_info.tail(10)

# COMMAND ----------

for seg_name in segment_info['entertainment_segment_lifetime'].unique():
    df_seg_amw= df_60[df_60['entertainment_segment_lifetime'] == seg_name]
    df_60_t = df_seg_amw.groupby(by=['user_id','is_cancel','sub_month'])[['monthly_hours_viewed', 'monthly_title_viewed']].sum().reset_index()
    
    if df_60_t.user_id.nunique() > 40000:
        df_60_s = get_churn_bin(df_60_t,[], 100)
    else:
        df_60_s = get_churn_bin(df_60_t,[], 10)

    med_x= df_60_t.monthly_title_viewed.median()
    fig, params = get_churn_plot_simple(df_60_s[df_60_s['title_viewed_bin']<15], 
                                        seg_name, param_dict, np.array(med_x))
    # slope = get_churn_slope_plot_simple(df_60_s, , params, np.array(med_x))
    med_dict[seg_name] = med_x
    # break
        

# COMMAND ----------

# MAGIC %md
# MAGIC # Distribution of the audience 

# COMMAND ----------

segment_info_top = segment_info.head(6)

# COMMAND ----------

df_60_t = df_60.groupby(by=['user_id','is_cancel','sub_month', 'entertainment_segment_lifetime'])\
            [['monthly_hours_viewed', 'monthly_title_viewed']].sum().reset_index()
df_60_t = df_60_t[df_60_t['entertainment_segment_lifetime'].isin(segment_info_top.entertainment_segment_lifetime.unique())]
df_60_s = get_equal_churn_bin(df_60_t, ['entertainment_segment_lifetime'])

user_total = df_60_s.groupby(['entertainment_segment_lifetime'])['user_id'].transform('sum')
df_60_s['Composition'] = df_60_s['user_id']/user_total

df_60_s[df_60_s['title_viewed_bin']<10].Composition.sum()/6 ##  91% people watched < 10 titles

# COMMAND ----------

fig = px.bar(df_60_s[df_60_s['title_viewed_bin']<10], 
             x="title_viewed_bin", y="Composition",
             color='entertainment_segment_lifetime', barmode='group',
             height=400)
fig.layout.yaxis.tickformat = ',.0%'
fig.show()

# COMMAND ----------

for seg_name in segment_info['entertainment_segment_lifetime'].unique():
    df_seg_amw= df_60[df_60['entertainment_segment_lifetime'] == seg_name]

    df_60_t = df_seg_amw.groupby(by=['user_id','is_cancel','sub_month', 'entertainment_segment_lifetime'])\
            [['monthly_hours_viewed', 'monthly_title_viewed']].sum().reset_index()
    df_60_s = get_equal_churn_bin(df_60_t, ['entertainment_segment_lifetime'])
    user_total = df_60_s.groupby(['entertainment_segment_lifetime'])['user_id'].transform('sum')
    df_60_s['Composition'] = df_60_s['user_id']/user_total

    fig = px.bar(df_60_s[df_60_s['title_viewed_bin']<10], 
             x="title_viewed_bin", y="Composition",
             color='entertainment_segment_lifetime', barmode='group',
             width=800, height=400)
    fig.layout.yaxis.tickformat = ',.0%'
    fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Segments by Segments

# COMMAND ----------

fig = go.Figure()
i = 0
for seg_name in param_dict.keys():
    color_discrete_sequence=["red", "green", "blue", "goldenrod", "magenta", "orange", "purple"]+px.colors.qualitative.Plotly
    color_cur = color_discrete_sequence[i]
    if seg_name in segment_info_top.entertainment_segment_lifetime.unique():
        a_fit, b_fit, c_fit = param_dict[seg_name]
        x_med = med_dict[seg_name]
        x_fit = np.linspace(0, 15, 100)   

        y_med = exponential_decay(x_med, a_fit, b_fit, c_fit)
        y_fit = exponential_decay(x_fit, a_fit, b_fit, c_fit)

        # fig.add_scatter( 
        #         x=x_fit, 
        #         y=y_fit, name = seg_name, color_discrete_sequence = color_cur)
        fig.add_trace(
            go.Scatter(
                mode='lines',
                x=x_fit,
                y=y_fit,
                marker=dict(
                    color=color_cur,
                    size=20,
                    line=dict(
                        color=color_cur,
                        width=2
                    )
                ),
                showlegend=True,
                name = seg_name
            )
        )   
        
        fig.add_scatter( 
                x=np.array(x_med), 
                y=np.array(y_med),
                    mode='markers',
                marker=dict(size=14, color=color_cur, line=dict(color='black', width=2),
                ),
                showlegend=False
                )
        i = i+1

fig.update_layout(
    template='simple_white',
    # showlegend=True,
    xaxis=dict(range=[0,15]),
    
) 
fig.show()



# COMMAND ----------

# MAGIC %md
# MAGIC # BY TENURE

# COMMAND ----------

df_churn = pd.read_csv('churn_curve_20231201.csv')

# COMMAND ----------

####
# df_60_00 = pd.read_parquet(file_path+'churn_user_stream60d_segmented_20240327.parquet')
df_60_00['tenure_months'] = df_60_00['sub_month']
df_60_00['monthly_title_viewed'] = np.where(df_60_00['tenure_months']>1, df_60_00['titles_viewed']/2, df_60_00['titles_viewed'])
df_60_00['monthly_hours_viewed'] = np.where(df_60_00['tenure_months']>1, df_60_00['hours_viewed']/2, df_60_00['hours_viewed'])

df_60_00['monthly_hours_viewed']  = df_60_00['monthly_hours_viewed'].astype(float)

# COMMAND ----------

|def get_simple_plot_multiple(df_plt, x, y, x_fit, y_fit, params, title=''):
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
            text='y= {:.2f} * e^({:.2f} * title_viewed) + {:.2f}'.format(a_fit, b_fit, c_fit),  # the text to display
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
        xaxis=dict(range=[0,15]),
        annotations=[
        dict(
            x=x_med+0.2,  # x-coordinate for the text
            y=y_med+0.01,  # y-coordinate for the text
            text='{:.2f}, {:.2f}'.format(x_med, y_med),  # the text to display
            showarrow=False,  # disable arrow for the annotation
            xanchor='left',
            font=dict(
                family='Arial',  # specify font family
                size=18,  # specify font size
                color='black'  # specify font color
            )
        )
    ]
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







# COMMAND ----------

def get_equal_churn_bin(df_in, grpby):
    df = df_in[df_in.monthly_hours_viewed<=60]
    df = df.groupby(by=['user_id','sub_month']+ grpby +['is_cancel']).sum().reset_index()
    nbins = int(df.monthly_title_viewed.max())
    df['title_viewed_bin'] = pd.cut(df['monthly_title_viewed'], 
                                    np.linspace(0,nbins,20))
    df['title_viewed_bin'] = df['title_viewed_bin'].apply(lambda x: x.right)
    df['churn'] = 1*df['is_cancel']  
    
    df_bin = df.groupby(['title_viewed_bin']+grpby).agg({'churn':'mean', 'user_id':'count',
                                                         'is_cancel':'sum','monthly_title_viewed':'sum'}).reset_index()
    return(df_bin)

# COMMAND ----------

df_60_00.head()

# COMMAND ----------

param_tenure_dict = {}
med_tenure_dict = {}

df_60_00['sub_month'] = df_60_00['sub_month'].astype(int)
mapping_dict = {1:'month_1_to_3', 2:'month_1_to_3', 3:'month_1_to_3', 4:'month_4_to_6', 5:'month_4_to_6', 6:'month_4_to_6', 7:'month_7_to_12', 8:'month_7_to_12', 9:'month_7_to_12', 10:'month_7_to_12', 11:'month_7_to_12', 12:'month_7_to_12'}
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

pd.options.display.float_format = '{:,.8f}'.format
df_seg_amw= df_60_00[df_60_00['tenure_bucket'] == 'month_7_to_12']
df_60_t = df_seg_amw.groupby(by=['user_id','is_cancel','sub_month'])[['monthly_title_viewed', 'monthly_hours_viewed']].sum().reset_index()
df_60_s = get_churn_bin(df_60_t, [])
df_60_s[['churn']].agg(['mean', 'median', 'std'])

# COMMAND ----------

fig = go.Figure()
i = 0
for seg_name in param_tenure_dict.keys():
    color_discrete_sequence=px.colors.qualitative.Plotly+px.colors.qualitative.Vivid
    color_cur = color_discrete_sequence[i]
    a_fit, b_fit, c_fit = param_tenure_dict[seg_name]
    x_med = med_tenure_dict[seg_name]
    x_fit = np.linspace(0, 15, 100)   

    y_med = exponential_decay(x_med, a_fit, b_fit, c_fit)
    y_fit = exponential_decay(x_fit, a_fit, b_fit, c_fit)

    # fig.add_scatter( 
    #         x=x_fit, 
    #         y=y_fit, name = seg_name, color_discrete_sequence = color_cur)
    fig.add_trace(
        go.Scatter(
            mode='lines',
            x=x_fit,
            y=y_fit,
            marker=dict(
                color=color_cur,
                size=20,
                line=dict(
                    color=color_cur,
                    width=2
                )
            ),
            showlegend=True,
            name = seg_name
        )
    )   
    
    fig.add_scatter( 
            x=np.array(x_med), 
            y=np.array(y_med),
                mode='markers',
            marker=dict(size=14, color=color_cur, line=dict(color='black', width=2),
            ),
            showlegend=False
            )
    i = i+1

fig.update_layout(
    template='simple_white',
    # showlegend=True,
    xaxis=dict(range=[0,15]),
    
) 
fig.show()

# COMMAND ----------

param_tenure_dict.keys()

# COMMAND ----------

seg_name

# COMMAND ----------

fig = go.Figure()

for seg_name in param_tenure_dict.keys():
    print(seg_name)
    a_fit, b_fit, c_fit = param_tenure_dict[seg_name]
    x_med = med_tenure_dict[seg_name]
    x_fit = np.linspace(1, 10, 10)   

    y_med = exponential_decay(x_med, a_fit, b_fit, c_fit)
    y_fit = exponential_decay(x_fit, a_fit, b_fit, c_fit)
    y_slope = exponential_decay_slope(x_fit, a_fit, b_fit)

    x = Symbol('x')
    x = x.evalf()
    f = (a_fit * b_fit *sp.exp(b_fit * x))/x
    f_prime_func = f.diff(x)
    f_prime = lambdify(x, f)

    y_opt = f_prime(x_fit)


    fig.add_scatter( 
        x=x_fit, 
        y=y_opt)
    
    # print(f_prime(1)/100)
    i = 1
    while i<10:
        # print(f_prime(i))
        if f_prime(i) > f_prime(1)/50:
            print (i)
            break
        else:
            i = i+0.5
    

    # break
fig.show()

# COMMAND ----------

# !pip install sympy

# COMMAND ----------

from sympy import *
import sympy as sp
from sympy import solve

# COMMAND ----------

x = Symbol('x')
# x = x.evalf()
f = x/(x+2)**3
f_prime_func = f.diff(x)
f = lambdify(x, f)
f_prime = lambdify(x, f_prime_func)

# COMMAND ----------

f_prime_func

# COMMAND ----------

x_fit = np.linspace(0, 10, 100)
y_fit = f(x_fit)
y_prime = f_prime(x_fit)

fig.add_scatter( 
        x=x_fit, 
        y=y_opt)

# COMMAND ----------

solve(f_prime)

# COMMAND ----------

# MAGIC %md
# MAGIC # Tenure by Library/new Content

# COMMAND ----------

df_60_00.head()

# COMMAND ----------

df_60_00.groupby(['tenure_bucket', 'old_new']).count()/df_60_00.groupby(['tenure_bucket']).count()

# COMMAND ----------

df_60_t.head()

# COMMAND ----------

df_60_t = df_60_00.groupby(by=['user_id','tenure_bucket'])[['monthly_title_viewed', 'monthly_hours_viewed']].sum().reset_index()
df_60_t_content = df_60_00.groupby(by=['user_id','is_cancel', 'tenure_bucket', 'old_new'])[['monthly_title_viewed', 'monthly_hours_viewed']].sum().reset_index()

# COMMAND ----------

df_final.head()

# COMMAND ----------

df_final = df_60_t_content.merge(df_60_t.rename(columns = {'monthly_title_viewed':'total_title_viewed', 'monthly_hours_viewed':'total_hours_viewed'}), 
                                on = ['user_id','tenure_bucket'])
df_final['frac'] = df_final['monthly_title_viewed']/df_final['total_title_viewed']

# COMMAND ----------

df_test = df_final[df_final['total_title_viewed']==2]

# COMMAND ----------

df_final[df_final['frac'] == 1]['user_id'].nunique()/df_final['user_id'].nunique() # 51% users watched mixed content

# COMMAND ----------

df_final[(df_final['frac'] < 1) & (df_final['old_new'] == 'current')]['monthly_title_viewed'].sum()/df_final[df_final['frac'] < 1]['monthly_title_viewed'].sum()

# COMMAND ----------

df_final[(df_final['frac'] == 1) &(df_final['old_new'] == 'current')]['user_id'].nunique()\
/df_final[df_final['frac'] == 1]['user_id'].nunique() # 51% users watched mixed content

# COMMAND ----------

param_tenure_dict = {}
med_tenure_dict = {}
groupby_col = 'old_new'
p0 = [0.5, -0.1, 0.01] 
param_bounds = ([0, -0.8, 0.01], [np.inf, -0.1, np.inf])

df_60_00['sub_month'] = df_60_00['sub_month'].astype(int)
mapping_dict = {1:'month_1_to_3', 2:'month_1_to_3', 3:'month_1_to_3', 4:'month_4_to_6', 5:'month_4_to_6', 6:'month_4_to_6', 7:'month_7_to_12', 8:'month_7_to_12', 9:'month_7_to_12', 10:'month_7_to_12', 11:'month_7_to_12', 12:'month_7_to_12'}
df_60_00['tenure_bucket'] = df_60_00['sub_month'].map(mapping_dict).fillna('month_13+')


for m in df_60_00.tenure_bucket.unique():
    df_seg_amw= df_60_00[df_60_00['tenure_bucket'] == m]
    # add seg total
    df_60_t = df_seg_amw.groupby(by=['user_id','is_cancel','sub_month'])[['monthly_title_viewed', 'monthly_hours_viewed']].sum().reset_index()
    df_60_s = get_churn_bin(df_60_t, [])
    df_60_s['title_viewed_bin'] = df_60_s['title_viewed_bin'].astype(float)
    df_60_s = df_60_s[df_60_s['title_viewed_bin']<=15]
    med_x= df_60_t.monthly_title_viewed.median()

    x_fit, params = fit_exponential(df_60_s.title_viewed_bin, df_60_s.churn, p0, param_bounds)
    a_fit, b_fit, c_fit = params
    y_fit = exponential_decay(x_fit, a_fit, b_fit, c_fit)
    y_med = exponential_decay(med_x, a_fit, b_fit, c_fit)
    y_med_slope = exponential_decay_slope(med_x, a_fit, b_fit)

    print(med_x)
    print('average churn: ' + str('{:.3f}'.format(y_med)))
    print('slope: ' + str('{:.3f}'.format(y_med_slope*100))+'%')
    print('y= {:.2f} * e^({:.2f} * title_viewed) + {:.2f}'.format(a_fit, b_fit, c_fit))

    x = Symbol('x')
    f = (a_fit * b_fit *sp.exp(b_fit * x))/x
    f_prime_func = f.diff(x)
    f_prime = lambdify(x, f)
    counter = 1
    while counter<15:
        # print(f_prime(i))
        if f_prime(counter) > f_prime(1)/50:
            print ('the optimal count of content' + str(counter))
            break
        else:
            counter= counter+0.5
    y_optimal = exponential_decay(counter, a_fit, b_fit, c_fit)
    
    fig = px.scatter(title=m, width=600, height=400)
    # fig.add_scatter(x=df_60_s.title_viewed_bin, y=df_60_s.churn, showlegend=False)
    fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', showlegend=True, name = 'segment total'))
    fig.add_scatter(x=np.array(med_x), y=np.array(y_med), mode='markers', marker=dict(size=14, color='red', line=dict(color='black', width=2)), showlegend=False)
    fig.add_scatter(x=np.array(counter), y=np.array(y_optimal), mode='markers', marker=dict(size=14, color='green', line=dict(color='black', width=1)), showlegend=False)

    ##### ADD BY Category ######
    for i in df_60_00[groupby_col].unique():
        
        df_seg_amw= df_60_00[(df_60_00['tenure_bucket'] == m) & (df_60_00[groupby_col] == i)]
        df_60_t = df_seg_amw.groupby(by=['user_id','is_cancel','sub_month'])[['monthly_title_viewed', 'monthly_hours_viewed']].sum().reset_index()
        df_60_s = get_churn_bin(df_60_t, [])
        df_60_s['title_viewed_bin'] = df_60_s['title_viewed_bin'].astype(float)
        df_60_s = df_60_s[df_60_s['title_viewed_bin']<=15]
        med_x= df_60_t.monthly_title_viewed.median()

        x_fit, params = fit_exponential(df_60_s.title_viewed_bin, df_60_s.churn, p0, param_bounds)
        a_fit, b_fit, c_fit = params
        x_fit = np.linspace(0, 15, 100)
        y_fit = exponential_decay(x_fit, a_fit, b_fit, c_fit)
        y_med = exponential_decay(med_x, a_fit, b_fit, c_fit)
        y_med_slope = exponential_decay_slope(med_x, a_fit, b_fit)

        print(med_x)
        print(i+' average churn: ' + str('{:.3f}'.format(y_med)))
        print(i+' slope: ' + str('{:.3f}'.format(y_med_slope*100))+'%')
        print('y= {:.2f} * e^({:.2f} * title_viewed) + {:.2f}'.format(a_fit, b_fit, c_fit))

        x = Symbol('x')
        f = (a_fit * b_fit *sp.exp(b_fit * x))/x
        f_prime_func = f.diff(x)
        f_prime = lambdify(x, f)

        counter = 1
        while counter<15:
            # print(f_prime(i))
            if f_prime(counter) > f_prime(1)/50:
                print ('the optimal count of content' + str(counter))
                break
            else:
                counter= counter+0.5
        y_optimal = exponential_decay(counter, a_fit, b_fit, c_fit)
        
        if i == 'current': i = 'new'
        fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', showlegend=True, name = i+' content'))
        # fig.add_scatter(x=df_60_s.title_viewed_bin, y=df_60_s.churn, showlegend=False)
        fig.add_scatter(x=np.array(med_x), y=np.array(y_med), mode='markers', marker=dict(size=10, color='red', line=dict(color='black', width=1)), showlegend=False)
        fig.add_scatter(x=np.array(counter), y=np.array(y_optimal), mode='markers', marker=dict(size=10, color='green', line=dict(color='black', width=1)), showlegend=False)
    
    fig.update_layout(
    template='simple_white',
    # showlegend=True,
    xaxis=dict(range=[0,15]),
    ) 
    fig.show()

    # break

# COMMAND ----------

df_60_00.head()

# COMMAND ----------

df_test = df_60_00.groupby(by=['user_id'])[['monthly_title_viewed', 'monthly_hours_viewed']].sum().reset_index()

# COMMAND ----------

df_test = df_test[df_test['monthly_title_viewed'] == 1]

# COMMAND ----------

df_1 = df_60_00[df_60_00['user_id'].isin(df_test.user_id)]

# COMMAND ----------

df_1 = df_60_00.groupby(['user_id', 'old_new'])[['monthly_hours_viewed']].sum()/df_60_00.groupby(['user_id'])[['monthly_hours_viewed']].sum()

# COMMAND ----------

df_1 = df_1.reset_index()

# COMMAND ----------

df_1.head()

# COMMAND ----------

df_library = df_1[(df_1['old_new']=='library') & (df_1['monthly_hours_viewed']>=0.85)] #80% of the population

# COMMAND ----------

df_60_00[groupby_col].unique()

# COMMAND ----------

param_tenure_dict = {}
med_tenure_dict = {}
groupby_col = 'old_new'
p0 = [0.5, -0.1, 0.01] 
param_bounds = ([0, -0.8, 0.01], [np.inf, -0.1, np.inf])

df_60_00['sub_month'] = df_60_00['sub_month'].astype(int)
mapping_dict = {1:'month_1_to_3', 2:'month_1_to_3', 3:'month_1_to_3', 4:'month_4_to_6', 5:'month_4_to_6', 6:'month_4_to_6', 7:'month_7_to_12', 8:'month_7_to_12', 9:'month_7_to_12', 10:'month_7_to_12', 11:'month_7_to_12', 12:'month_7_to_12'}
df_60_00['tenure_bucket'] = df_60_00['sub_month'].map(mapping_dict).fillna('month_13+')


for m in df_60_00.tenure_bucket.unique():
    df_seg_amw= df_60_00[df_60_00['tenure_bucket'] == m]
    # add seg total
    df_60_t = df_seg_amw.groupby(by=['user_id','is_cancel','sub_month'])[['monthly_title_viewed', 'monthly_hours_viewed']].sum().reset_index()
    df_60_s = get_churn_bin(df_60_t, [])
    df_60_s['title_viewed_bin'] = df_60_s['title_viewed_bin'].astype(float)
    df_60_s = df_60_s[df_60_s['title_viewed_bin']<=15]
    med_x= df_60_t.monthly_title_viewed.median()

    x_fit, params = fit_exponential(df_60_s.title_viewed_bin, df_60_s.churn, p0, param_bounds)
    a_fit, b_fit, c_fit = params
    y_fit = exponential_decay(x_fit, a_fit, b_fit, c_fit)
    y_med = exponential_decay(med_x, a_fit, b_fit, c_fit)
    y_med_slope = exponential_decay_slope(med_x, a_fit, b_fit)

    print(med_x)
    print('average churn: ' + str('{:.3f}'.format(y_med)))
    print('slope: ' + str('{:.3f}'.format(y_med_slope*100))+'%')
    print('y= {:.2f} * e^({:.2f} * title_viewed) + {:.2f}'.format(a_fit, b_fit, c_fit))

    x = Symbol('x')
    f = (a_fit * b_fit *sp.exp(b_fit * x))/x
    f_prime_func = f.diff(x)
    f_prime = lambdify(x, f)
    counter = 1
    while counter<15:
        # print(f_prime(i))
        if f_prime(counter) > f_prime(1)/50:
            print ('the optimal count of content' + str(counter))
            break
        else:
            counter= counter+0.5
    y_optimal = exponential_decay(counter, a_fit, b_fit, c_fit)
    
    fig = px.scatter(title=m, width=600, height=400)
    # fig.add_scatter(x=df_60_s.title_viewed_bin, y=df_60_s.churn, showlegend=False)
    fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', showlegend=True, name = 'segment total'))
    fig.add_scatter(x=np.array(med_x), y=np.array(y_med), mode='markers', marker=dict(size=14, color='red', line=dict(color='black', width=2)), showlegend=False)
    fig.add_scatter(x=np.array(counter), y=np.array(y_optimal), mode='markers', marker=dict(size=14, color='green', line=dict(color='black', width=1)), showlegend=False)

    ##### ADD BY Category ######
    for i in df_60_00[groupby_col].unique():
        
        df_seg_amw= df_60_00[(df_60_00['tenure_bucket'] == m)]# & (df_60_00[groupby_col] == i)]

        if i == 'library':
            df_seg_amw = df_seg_amw[df_seg_amw['user_id'].isin(df_library.user_id)]
        else:
            df_seg_amw = df_seg_amw[~df_seg_amw['user_id'].isin(df_library.user_id)]

        df_60_t = df_seg_amw.groupby(by=['user_id','is_cancel','sub_month'])[['monthly_title_viewed', 'monthly_hours_viewed']].sum().reset_index()
        df_60_s = get_churn_bin(df_60_t, [])
        df_60_s['title_viewed_bin'] = df_60_s['title_viewed_bin'].astype(float)
        df_60_s = df_60_s[df_60_s['title_viewed_bin']<=15]
        med_x= df_60_t.monthly_title_viewed.median()

        x_fit, params = fit_exponential(df_60_s.title_viewed_bin, df_60_s.churn, p0, param_bounds)
        a_fit, b_fit, c_fit = params
        x_fit = np.linspace(0, 15, 100)
        y_fit = exponential_decay(x_fit, a_fit, b_fit, c_fit)
        y_med = exponential_decay(med_x, a_fit, b_fit, c_fit)
        y_med_slope = exponential_decay_slope(med_x, a_fit, b_fit)

        print(med_x)
        print(i+' average churn: ' + str('{:.3f}'.format(y_med)))
        print(i+' slope: ' + str('{:.3f}'.format(y_med_slope*100))+'%')
        print('y= {:.2f} * e^({:.2f} * title_viewed) + {:.2f}'.format(a_fit, b_fit, c_fit))

        x = Symbol('x')
        f = (a_fit * b_fit *sp.exp(b_fit * x))/x
        f_prime_func = f.diff(x)
        f_prime = lambdify(x, f)

        counter = 1
        while counter<15:
            # print(f_prime(i))
            if f_prime(counter) > f_prime(1)/50:
                print ('the optimal count of content' + str(counter))
                break
            else:
                counter= counter+0.5
        y_optimal = exponential_decay(counter, a_fit, b_fit, c_fit)
        
        if i == 'current': i = 'new'
        fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', showlegend=True, name = i+' content watcher'))
        # fig.add_scatter(x=df_60_s.title_viewed_bin, y=df_60_s.churn, showlegend=False)
        fig.add_scatter(x=np.array(med_x), y=np.array(y_med), mode='markers', marker=dict(size=10, color='red', line=dict(color='black', width=1)), showlegend=False)
        fig.add_scatter(x=np.array(counter), y=np.array(y_optimal), mode='markers', marker=dict(size=10, color='green', line=dict(color='black', width=1)), showlegend=False)
    
    fig.update_layout(
    template='simple_white',
    # showlegend=True,
    xaxis=dict(range=[0,15]),
    ) 
    fig.show()

    # break

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


