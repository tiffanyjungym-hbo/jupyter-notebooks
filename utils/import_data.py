
## update GPE estimate table for cancels -- June 27, 2023
## update category='retail' for paid_adds -- Aug 4, 2023
import sys
import pandas as pd

root_path = sys.argv[1]


# Launch dates
launch_dates = {'DOMESTIC':'2020-05-27',
                'LATAM':'2021-06-29',
                'EMEA_WAVE_1':'2021-10-26',
                'EMEA_WAVE_2':'2022-03-08'}

# Important promos
promo_select_us = []
promo_select_emea = ['emea_launch_50%off_2021_Oct','emea_launch_2022_Mar']
promo_select_latam = []

promo_select = promo_select_us + promo_select_emea + promo_select_latam
promo_select_str = "'" + "','".join(promo_select) + "'"
promo_select_str

colSelect = ['DATE','LEVEL1','LEVEL2','LEVEL3','LEVEL4','LEVEL5','LEVEL6','LEVEL7','LAUNCH_SERVICE_WAVE',
             'REGION','FORECAST_TERRITORY','PAID_ADDS']

q_latam = f"""
date,
sku as level1,
case when signup_offer is null then 'Non-Promo'
when signup_offer='latam_migration' then 'Migrated'
else 'Promo' end as level2,
coalesce(signup_offer,'none') as level3,
'none' as level4,
'none' as level5,
'none' as level6,
'none' as level7,
region,
forecast_territory,
launch_service_wave
"""

q_emea = f"""
date, 
case when sku = 'SVOD Annual' then 'Annual'
    when signup_offer in ({promo_select_str}) then 'Promo' else 'Non-Promo' end as level1,
case when signup_offer in ({promo_select_str}) then signup_offer else 'none' end as level2,
'none' as level3,
'none' as level4,
'none' as level5,
'none' as level6,
'none' as level7,
region,
forecast_territory,
launch_service_wave
"""

##### Check that data is ready #####
def data_check():
    q = '''
    select * from max_dev.workspace.forecasting_data_check

    '''
    return pd.read_sql(q, con)

##### Load paid adds
def load_adds(region, q_string, last_actual):
    
    q = f"""
    select {q_string}, sum(paid_adds) as paid_adds
    from max_dev.workspace.forecasting_paid_adds
    where 
    region = '{region}'
    
    and category='retail'
    and 
    (
    (launch_service_wave = 'EMEA_WAVE_1' and date >= '{launch_dates["EMEA_WAVE_1"]}')
    or (launch_service_wave = 'EMEA_WAVE_2' and date >= '{launch_dates["EMEA_WAVE_2"]}')
    or (launch_service_wave = 'LATAM' and date>= '{launch_dates["LATAM"]}'))
    and date <= '{last_actual}'
    group by 1,2,3,4,5,6,7,8,9,10,11
    """
    return q


##### Load paid cancels

def load_cancels(region, q_string, last_actual, grace_period):
    q = f"""
    select {q_string}, sum(paid_cancels) as paid_cancels_matured
    from max_dev.workspace.forecasting_paid_cancels
    where 
    region = '{region}'
    and
    ((launch_service_wave = 'EMEA_WAVE_1' and date >= '{launch_dates["EMEA_WAVE_1"]}')
    or (launch_service_wave = 'EMEA_WAVE_2' and date >= '{launch_dates["EMEA_WAVE_2"]}')
    or (launch_service_wave = 'LATAM' and date>= '{launch_dates["LATAM"]}')
    )
    and date <= '{last_actual}'::date - {grace_period}
    
    group by 1,2,3,4,5,6,7,8,9,10,11
    """
    return q

##### Load grace period estimates

def load_gpe(region):
    q = f'''
    SELECT
        region, date, SUM(eligible) AS total_eligibles,
        ROUND((SUM(gpe_eligible) + SUM(gpe_unmatured)) / 2, 0) AS total_estimated_paid_cancels,
        ROUND(total_estimated_paid_cancels / total_eligibles, 2) AS cancel_rate
    FROM max_dev.workspace.forecasting_paid_cancels_grace_period_estimate_sku_promo
    where region='{region}'
    GROUP BY 1, 2
    having sum(eligible)>0
    ORDER BY 1, 2;
    '''
    return q
#FROM max_dev.workspace.forecasting_paid_cancels_grace_period_estimate

##### Paid balance

def load_balance(region, q_string, last_actual, grace_period):
    q = f'''
    select {q_string}, sum(paid_balance) as paid_balance
    from max_dev.workspace.forecasting_paid_balance_legacy
    where date<='{last_actual}'
    and region = '{region}'
    and date::date <= '{last_actual}'::date - {grace_period}
    group by 1,2,3,4,5,6,7,8,9,10,11
    '''
    return q
    
##### First views
def load_fv(region):
    region_string = "'" + "','".join(region) + "'"
    q = f"""
    select region, date, title_id, imdb_id, title_name, season_number, forecast_category, week_end, last_day(date) as month, premiere_date, days_after_premiere, sum(retail_fv) as retail_first_views, sum(wholesale_fv) as wholesale_first_views, sum(pvc_fv) as pvc_first_views, sum(RETAIL_PVC_FV) as RETAIL_PVC_FV, sum(TOTAL_FV) as TOTAL_FV
    from max_dev.workspace.forecasting_fv_pacing
    right join max_dev.workspace.forecasting_top_tier using (imdb_id,season_number,launch_service_wave)
    where region in ({region_string})
    group by 1,2,3,4,5,6,7,8,9,10,11
    """
    return q
    
##### Live sports first views
def load_fv_sports(region):
    q = f"""
    select date, region, sum(fv) as fv_sports
    from max_dev.workspace.forecasting_fv_daily_premiere
    where content_category in ('sports','livesports')
    and region = '{region}'
    and category = 'retail'
    group by 1,2
    """
    return q

##### Load projections
def load_proj(region):
    q = f"""select * from max_dev.workspace.forecasting_proj where region='{region}'"""
    return q
    
##### Migrated subs
def load_migrated(region):
    if region=='EMEA':
        q = '''
        select date, 'Migrated' as level1,
        offer_cohort as level2,
        'none' as level3, 'none' as level4, 'none' as level5, 'none' as level6, 'none' as level7,
        'EMEA' as region,
        case when region_cohort in ('DENMARK','FINLAND','NORWAY','SWEDEN') then 'NORDICS'
        when region_cohort in ('NETHERLANDS','PORTUGAL','SPAIN') then region_cohort
        else 'CENTRAL EUROPE' end as forecast_territory,
        case when region_cohort in ('DENMARK','FINLAND','NORWAY','SWEDEN','SPAIN') then 'EMEA_WAVE_1' else 'EMEA_WAVE_2' end as launch_service_wave,
        sum(num_subs) * 1.25 as paid_adds
        from max_dev.workspace.LTV_DAILY_COHORT_RETENTION_EMEA
        where offer_cohort in ('emea_migration','emea_w2_migration')
        and BASELINE_COHORT = 'Baseline'
        group by 1,2,3,4,5,6,7,8,9,10,11
        '''
    else:
        q = '''
        select date, 'Migrated' as level1,
        offer_cohort as level2,
        'none' as level3, 'none' as level4, 'none' as level5, 'none' as level6, 'none' as level7,
        'EMEA' as region,
        case when region_cohort in ('DENMARK','FINLAND','NORWAY','SWEDEN') then 'NORDICS'
        when region_cohort in ('NETHERLANDS','PORTUGAL','SPAIN') then region_cohort
        else 'CENTRAL EUROPE' end as forecast_territory,
        case when region_cohort in ('DENMARK','FINLAND','NORWAY','SWEDEN','SPAIN') then 'EMEA_WAVE_1' else 'EMEA_WAVE_2' end as launch_service_wave,
        sum(num_subs) * 1.25 as paid_adds
        from max_dev.workspace.LTV_DAILY_COHORT_RETENTION_EMEA
        where offer_cohort in ('emea_migration','emea_w2_migration')
        and BASELINE_COHORT = 'Baseline'
        group by 1,2,3,4,5,6,7,8,9,10,11
        '''
        
    return q
    
##### Premiere dates
# def load_premieres():
#     q = 'select premiere_date, title_name from max_dev.workspace.forecasting_premieres where dummy_show = 0'
#     return q
##### Updated Premiere dates + Promo dates


#####===================
#####Supplementary Tabs
#####===================
##### Daily First Views Supplementary Tab
def generate_fv_daily(region,first_view_days, min_date, max_date):
    FVPacing = pd.read_sql(load_fv(region), con)
    FVPacing = FVPacing[['DATE','RETAIL_FIRST_VIEWS','IMDB_ID','PREMIERE_DATE','TITLE_NAME','SEASON_NUMBER', 'FORECAST_CATEGORY','DAYS_AFTER_PREMIERE']]
    FVPacing.columns = ['DATE', 'First Views', 'imdb_id', 'PREMIERE_DATE', 'TITLE_NAME','SEASON_NUMBER', 'CONTENT_CATEGORY','Days After Premiere']
    FVPacing.PREMIERE_DATE = d(FVPacing.PREMIERE_DATE)
    first_view_long = FVPacing[(FVPacing['Days After Premiere']<first_view_days)&(FVPacing.PREMIERE_DATE.between(d(min_date), d(max_date)))]
    
    # Check if there's any duplicated title record:
    #The check below should gives four duplicated imdb_id x season_number records for US
    #     - `tt12361974` (Zack Snyder), `tt8543208` (Tom and Jerry), `tt0446622` (Hard Knocks), and `tt8772296` (Euphoria Special)
    if (region == 'NORTH AMERICA') and (not (pd.Series(first_view_long.groupby(['imdb_id','SEASON_NUMBER']).TITLE_NAME.nunique().reset_index().query('TITLE_NAME>1').imdb_id.unique()).isin( ['tt12361974', 'tt8543208', 'tt8772296','tt0446622'])).all()):
        print('Error in First View Pacing: Duplicated Titles')
    # Duplicate Sanity Check Ends
    
    first_view_long = first_view_long.sort_values(by = ['PREMIERE_DATE','TITLE_NAME','SEASON_NUMBER', 'DATE'])
    #export_gs(sheet_name = report_sheet_name, tab_name = 'Daily First Views', data = first_view_long
    #     , date_columns = ['DATE','PREMIERE_DATE'])
    return first_view_long



##### Content Schedule Supplementary Tab
def load_premieres(region):
    pdate_mapping = {'NORTH AMERICA':'premiere_date','LATAM':'premiere_date_latam','EMEA':'premiere_date_emea'}
    pdate = pdate_mapping[region]

    promo = pd.DataFrame(import_gs('Forecast Imports', 'Promo'))
    promo.columns = promo.iloc[0]
    promo = promo.iloc[1:]
    promo = promo[promo.REGION == region]

    promo = pd.melt(promo[['PROMO_NAME','BEGIN_DATE','END_DATE']],id_vars=['PROMO_NAME'], value_vars=['BEGIN_DATE','END_DATE'])
    promo.columns = ['Promo','Begin_End','PREMIERE_DATE']
    promo['TITLE_NAME'] = np.where(promo.Begin_End == 'BEGIN_DATE', promo['Promo'] + ' Begins', promo['Promo'] + ' Ends')

    content_schedule_query = f'''select distinct {pdate}, TITLE_NAME from MAX_DEV.WORKSPACE.FORECASTING_PREMIERES
    where {pdate} >='2020-01-01' and exclude=0 order by 1'''
    content_schedule= pd.read_sql(content_schedule_query, con)
    content_schedule['TITLE_NAME'] = content_schedule['TITLE_NAME'].str.title()
    content_df = pd.concat([promo[['PREMIERE_DATE','TITLE_NAME']],pd.DataFrame({'PREMIERE_DATE':[''],'TITLE_NAME':['']}), content_schedule.rename(columns={pdate.upper():'PREMIERE_DATE'})],axis = 0)
    return content_df
    
##### FV Summary
def generate_fv_summary(region, min_date, max_date, include_pvc = False, include_wholesale = False):
    df_fv_title = pd.read_sql(load_fv(region), con)
    df_fv_title['DATE'] = d(df_fv_title['DATE'])
    df_fv_title['WEEK_END'] = d(df_fv_title['WEEK_END'])
    df_fv_title['PREMIERE_DATE'] = d(df_fv_title['PREMIERE_DATE'])
    df_fv_premiere_date = df_fv_title.query("REGION==@region & @min_date<=DATE<=@max_date").groupby(['SEASON_NUMBER','IMDB_ID','FORECAST_CATEGORY'])[['TITLE_NAME','PREMIERE_DATE']].first().reset_index()
    if include_pvc and not include_wholesale:
        fv_cols = ['RETAIL_PVC_FV']
    elif include_pvc and include_wholesale:
        fv_cols = ['TOTAL_FV']
    else:
        fv_cols = ['RETAIL_FIRST_VIEWS']
        
    df_fv_cumu = (df_fv_title
                  .query("REGION==@region & PREMIERE_DATE>=@min_date")
                  .groupby(['SEASON_NUMBER','IMDB_ID','FORECAST_CATEGORY','DAYS_AFTER_PREMIERE'])[fv_cols].sum().reset_index())

    df_fv_cumu['FV_cumu'] = (df_fv_cumu
                  .groupby(['IMDB_ID','SEASON_NUMBER'])[fv_cols].cumsum().sum(axis = 1))

    df_fv_7d = (df_fv_cumu
                  .query("DAYS_AFTER_PREMIERE==[0,3,6,30,89]").pivot_table(index=['IMDB_ID','SEASON_NUMBER','FORECAST_CATEGORY'],columns='DAYS_AFTER_PREMIERE',values='FV_cumu')
                  .reset_index()
                 )
    df_fv_7d.columns = ['IMDB_ID','SEASON_NUMBER','FORECAST_CATEGORY','FV1','FV4','FV7','FV31','FV90']
    df_fv_title = df_fv_title[df_fv_title.DAYS_AFTER_PREMIERE.between(0,89)]
    df_fv_monthly = df_fv_title.query("REGION==@region & @min_date<=DATE<=@max_date").groupby(['IMDB_ID','SEASON_NUMBER','MONTH'])[fv_cols].sum().sum(axis = 1).unstack().fillna(0)
    df_fv_weekly = df_fv_title.query("REGION==@region & @min_date<=WEEK_END<=@max_date").groupby(['IMDB_ID','SEASON_NUMBER','WEEK_END'])[fv_cols].sum().sum(axis = 1).unstack().fillna(0)
    df_fv_summary = (df_fv_premiere_date.merge(df_fv_7d, on=['SEASON_NUMBER','IMDB_ID','FORECAST_CATEGORY'])
                     .merge(df_fv_monthly, on=['IMDB_ID','SEASON_NUMBER'], how='right')
                     .merge(df_fv_weekly, on=['IMDB_ID','SEASON_NUMBER'], how='right')
                     .sort_values('FV90',ascending=False))
    df_fv_summary = df_fv_summary[['TITLE_NAME','SEASON_NUMBER','IMDB_ID','FORECAST_CATEGORY']+list(df_fv_summary.columns[4:])]
    return df_fv_summary

#####======================================
#####Global First Views Google Sheet Update
#####======================================
def generate_global_fv_summary_query(days = [6,27,89], regions = ['NORTH AMERICA','LATAM','EMEA'] ):
    return ", ".join([f"sum(case when days_after_premiere between 0 and {d} and region = '{w}'  then {var}_fv end) as {w.replace(' ','_')}_{var}_fv{str(d+1)}"
            for d in days for var in ['retail','pvc','total']  for w in regions])
def generate_global_fv_gsheet_data(min_premiere_date, max_premiere_date,regions = ['NORTH AMERICA','LATAM','EMEA']):
    query = f'''with premieres as (select IMDB_ID,SEASON_NUMBER,TRAILER_DATE,PREMIERE_DATE,PREMIERE_DATE_EMEA,PREMIERE_DATE_LATAM,TITLE_NAME,FORECAST_CATEGORY,MODEL
    from max_dev.workspace.forecasting_premieres
    where exclude!=1),
    post_release as (select imdb_id, season_number,
    {generate_global_fv_summary_query(days = [6], regions = regions )}
    from max_dev.workspace.forecasting_fv_pacing_postrelease 
    group by 1,2),
    pre_release as (select imdb_id, season_number,
    {generate_global_fv_summary_query(days = [6,27,89], regions = regions)}
    from max_dev.workspace.forecasting_fv_pacing_prerelease where title_name is not null
    group by 1,2)
    select CONCAT(premieres.imdb_id, premieres.SEASON_NUMBER) as key ,premieres.*, 
    post_release.NORTH_AMERICA_retail_fv7 post_release_NORTH_AMERICA_retail_fv7
    , post_release.NORTH_AMERICA_PVC_fv7 post_release_NORTH_AMERICA_PVC_fv7
    , post_release.LATAM_retail_fv7 post_release_LATAM_retail_fv7
    , post_release.EMEA_retail_fv7 post_release_EMEA_retail_fv7
    , pre_release.NORTH_AMERICA_retail_fv7 pre_release_NORTH_AMERICA_retail_fv7
    , pre_release.NORTH_AMERICA_PVC_fv7 pre_release_NORTH_AMERICA_PVC_fv7
    , pre_release.LATAM_retail_fv7 pre_release_LATAM_retail_fv7
    , pre_release.EMEA_retail_fv7 pre_release_EMEA_retail_fv7,
    pre_release.NORTH_AMERICA_retail_fv28, pre_release.NORTH_AMERICA_PVC_fv28, pre_release.LATAM_retail_fv28, pre_release.EMEA_retail_fv28, 
    pre_release.NORTH_AMERICA_total_fv28, pre_release.LATAM_total_fv28, pre_release.EMEA_total_fv28, 
    pre_release.NORTH_AMERICA_retail_fv90 , pre_release.NORTH_AMERICA_PVC_fv90, pre_release.LATAM_retail_fv90, pre_release.EMEA_retail_fv90, 
    pre_release.NORTH_AMERICA_total_fv90, pre_release.LATAM_total_fv90, pre_release.EMEA_total_fv90
    from premieres
    left join post_release using(imdb_id, season_number) 
    left join pre_release using(imdb_id,season_number)
    where coalesce( premiere_date,PREMIERE_DATE_LATAM,PREMIERE_DATE_EMEA) between '{min_premiere_date}' and '{max_premiere_date}'
    and coalesce(  {",".join([f"{p}_release.{r.replace(' ','_')}_RETAIL_FV7" for p in ['post','pre'] for r in regions])} ) is not null
    order by coalesce(premiere_date,PREMIERE_DATE_LATAM,PREMIERE_DATE_EMEA),title_name'''
    global_fv = pd.read_sql(query,con)
    global_fv = global_fv.replace(np.nan,'')
    return global_fv

def generate_global_fv_summary_query_2024(days = [6,27,89], regions = ['NORTH AMERICA','LATAM','EMEA'] ):
    return ", ".join([f"sum(case when days_after_premiere between 0 and {d} and region = '{w}'  then {var}_fv end) as {w.replace(' ','_')}_{var}_fv{str(d+1)}"
           for d in days for w in regions  for var in ['retail','pvc','wholesale']   ])
    
def generate_global_fv_gsheet_data_2024(min_premiere_date, max_premiere_date,regions = ['NORTH AMERICA','LATAM','EMEA']):
    query = f'''with premieres as (select IMDB_ID,SEASON_NUMBER,TRAILER_DATE,PREMIERE_DATE,PREMIERE_DATE_EMEA,PREMIERE_DATE_LATAM,TITLE_NAME,FORECAST_CATEGORY,MODEL
    from max_dev.workspace.forecasting_premieres
    where exclude!=1),
    post_release as (select imdb_id, season_number,
    {generate_global_fv_summary_query_2024(days = [6,27,89], regions = regions )}
    from max_dev.workspace.forecasting_fv_pacing_postrelease 
    group by 1,2),
    pre_release as (select imdb_id, season_number,
    {generate_global_fv_summary_query_2024(days = [6,27,89], regions = regions)}
    from max_dev.workspace.forecasting_fv_pacing_prerelease where title_name is not null
    group by 1,2)
    select CONCAT(premieres.imdb_id, premieres.SEASON_NUMBER) as key ,premieres.*, 
    {", ".join([f"{r}.{w.replace(' ','_')}_{var}_fv{d}"
            for d in [7,28,90] for r in ['post_release','pre_release']  for w in regions  for var in ['retail','pvc','wholesale']  ])}
    from premieres
    left join post_release using(imdb_id, season_number) 
    left join pre_release using(imdb_id,season_number)
    where coalesce( premiere_date,PREMIERE_DATE_LATAM,PREMIERE_DATE_EMEA) between '{min_premiere_date}' and '{max_premiere_date}'
    and coalesce(  {",".join([f"{p}_release.{r.replace(' ','_')}_RETAIL_FV7" for p in ['post','pre'] for r in regions])} ) is not null
    order by coalesce(premiere_date,PREMIERE_DATE_LATAM,PREMIERE_DATE_EMEA),title_name'''
    #print(query)
    global_fv = pd.read_sql(query,con)
    global_fv = global_fv.replace(np.nan,'')
    return global_fv

def summarize_adds(df, idx, output_cols, date_a, date_b):
    df_out = df.query("@date_a<=DATE<=@date_b").groupby([idx,'REGION'])[output_cols].sum().reset_index()

    for i in output_cols:
        #df_out[i] = np.round(df_out[i]/1000,decimals=1)
        df_out[i] = np.round(df_out[i])
    
    output_us = df_out.query("REGION=='NORTH AMERICA'")
    output_latam = df_out.query("REGION=='LATAM'")
    output_emea = df_out.query("REGION=='EMEA'")

    output_final = (output_us
        .merge(output_latam, on=idx, suffixes=['_US','_LATAM'], how='outer')
        .merge(output_emea, on=idx, how='outer')
        .sort_values(idx))
    
    return output_final