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


def train(df_in,target, features):
    model = sm.quantreg('ln_' + target + ' ~ ' + features, data=df_in.query("future_title==0")).fit()
    df_in['fv_predict'] = np.exp(model.predict(df_in)) + 1
    df_in['error'] = df_in['ln_' + target] - model.predict(df_in)
    df_in = df_in.loc[~df_in.error.isnull()]
    return model, df_in

def get_coef(model, model_name, target):
    # Model coefficients
        df_coef = (model.params.reset_index()
                .merge(model.tvalues.reset_index(), on='index')
                .merge(model.pvalues.reset_index(), on='index')
                )
        df_coef.columns = ['feature','coef','tstat','pvalue']
        df_coef['model'] = model_name
        df_coef['target'] = target
        return df_coef[['model','target','feature','coef','tstat','pvalue']]

# Return model RSQ and MAPE
def get_summary(df_in, model, model_name, features, target):
    df_summary = pd.DataFrame({
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

# MAGIC %md
# MAGIC # Series

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from bolt_cus_dev.gold.delphi_titles limit 10

# COMMAND ----------

## include longer term wiki & imdb features  
## 

# COMMAND ----------

query = """
select 
* 
from bolt_cus_dev.silver.forecasting_feature_space_test f
left join bolt_cus_dev.gold.delphi_title_metadata d
on f.imdb_id=d.imdb_series_id and f.season_number= d.season_number
left join bolt_cus_dev.silver.delphi_title_imdb_refs_future
where f.premiere_date>='2021-01-01'
    and f.premiere_date<'2024-04-01'
    and f.region='NORTH AMERICA'
    and d.geo_value='UNITED STATES'"""

df_raw = spark.sql(query)
df_raw = df_raw.toPandas()

df_raw.rename(columns={'season_number': 'season_number_'}, inplace=True)
df_raw.columns=df_raw.columns.str.lower()

## Clean data 
df_raw = df_raw[df_raw.rfv7.notnull()]
df_raw = df_raw.drop_duplicates(subset=['title_name','season_number','rfv7'])
df_raw['medal'] = np.where(df_raw['medal']=='Other', df_raw['predicted_medal_us'],df_raw['medal'])

df_raw.shape

# COMMAND ----------

## days_diff for series 
# df_s = df_raw[(df_raw.forecast_category=='series') & (df_raw.season_number<2)]
df_s = df_raw.copy()

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
       'runtime_in_mins', 'derived_genre']
df_s = df_s.sort_values(by=['first_release_date','last_release_date'])[col]
df_s['first_release_date'] = pd.to_datetime(df_s['first_release_date'])
df_s['last_release_date'] = pd.to_datetime(df_s['last_release_date'])


## Days_diff1: new release after last_release_date
## interval, allow +/- day overlap 
df_s['last_release_date_ref'] = df_s['first_release_date'] + pd.DateOffset(days=0)
df_s['days_diff'] = None
for index, row in df_s.iterrows():
    ## Filter titles that was released before the title, and last release date is within 28d after title's release date
    eligible_rows = df_s[(df_s['medal'].isin(['Gold', 'Platinum'])) 
                         & (df_s['last_release_date'] < row['last_release_date_ref']) 
                         & (df_s['first_release_date'] < row['first_release_date'])]
    if not eligible_rows.empty:
        latest_date = eligible_rows['last_release_date'].max()
        latest_title = eligible_rows.loc[eligible_rows['last_release_date'] == latest_date, 'title_name'].iloc[0]
        latest_title_fv = eligible_rows.loc[eligible_rows['last_release_date'] == latest_date, 'rfv7'].iloc[0]
        df_s.at[index, 'days_diff_1'] = (row['first_release_date'] - latest_date).days
        df_s.at[index, 'prev_title_1'] = latest_title
        df_s.at[index, 'prev_title_fv_1'] = latest_title_fv

## Days_diff2: new release > last_release_date-14days
## interval, allow +/- day overlap 
df_s['last_release_date_ref'] = df_s['first_release_date'] + pd.DateOffset(days=14)
df_s['days_diff'] = None
for index, row in df_s.iterrows():
    ## Filter titles that was released before the title, and last release date is within 28d after title's release date
    eligible_rows = df_s[(df_s['medal'].isin(['Gold', 'Platinum'])) 
                         & (df_s['last_release_date'] < row['last_release_date_ref']) 
                         & (df_s['first_release_date'] < row['first_release_date'])]
    if not eligible_rows.empty:
        latest_date = eligible_rows['last_release_date'].max()
        latest_title = eligible_rows.loc[eligible_rows['last_release_date'] == latest_date, 'title_name'].iloc[0]
        latest_title_fv = eligible_rows.loc[eligible_rows['last_release_date'] == latest_date, 'rfv7'].iloc[0]
        df_s.at[index, 'days_diff_2'] = (row['first_release_date'] - latest_date).days
        df_s.at[index, 'prev_title_2'] = latest_title
        df_s.at[index, 'prev_title_fv_2'] = latest_title_fv

## Days_diff3: series only, after last release date
## interval, allow +/- day overlap 
df_s['last_release_date_ref'] = df_s['first_release_date'] + pd.DateOffset(days=0)
df_s['days_diff'] = None
for index, row in df_s.iterrows():
    ## Filter titles that was released before the title, and last release date is within 28d after title's release date
    eligible_rows = df_s[(df_s['medal'].isin(['Gold', 'Platinum'])) 
                         & (df_s['forecast_category'].isin(['series'])) 
                         & (df_s['last_release_date'] < row['last_release_date_ref']) 
                         & (df_s['first_release_date'] < row['first_release_date'])]
    if not eligible_rows.empty:
        latest_date = eligible_rows['last_release_date'].max()
        latest_title = eligible_rows.loc[eligible_rows['last_release_date'] == latest_date, 'title_name'].iloc[0]
        latest_title_fv = eligible_rows.loc[eligible_rows['last_release_date'] == latest_date, 'rfv7'].iloc[0]
        df_s.at[index, 'days_diff_3'] = (row['first_release_date'] - latest_date).days
        df_s.at[index, 'prev_title_3'] = latest_title
        df_s.at[index, 'prev_title_fv_3'] = latest_title_fv

df_s[['title_name','medal','first_release_date','last_release_date','last_release_date_ref','days_diff_1','days_diff_2','days_diff_3','prev_title_1','prev_title_2','prev_title_3']].head(30)

# COMMAND ----------

col_id = ['imdb_id','title_name','season_number','premiere_date','region','forecast_category','future_title']
df_coef=[]
df_summary =[]

medal = ['Gold','Platinum']
forecast_category = ['series']
features = ['search_2wbp', 'wiki_2wbp', 'num_episodes', 'medal', 'fv_pre_7','days_diff_1','days_diff_2','days_diff_3', 'prev_title_1', 'prev_title_fv_1','prev_title_fv_2','prev_title_fv_3']
target = 'rfv7'
model_list = {
    '2wbp_d':'ln_days_diff_1',
    '2wbp_dm':'ln_days_diff_1 + medal',
    '2wbp_dmn':'ln_days_diff_1 + medal + ln_num_episodes',
    '2wbp_dmnp':'ln_days_diff_1 + medal  + ln_num_episodes + ln_fv_pre_7',
    '2wbp_dmnf':'ln_days_diff_1 + medal  + ln_num_episodes + ln_prev_title_fv_1',
    # '2wbp_dmnp2':'ln_days_diff_2 + medal  + ln_num_episodes + ln_fv_pre_7',
    # '2wbp_dmnf2':'ln_days_diff_2 + medal  + ln_num_episodes + ln_prev_title_fv_2',
    # '2wbp_dmnp3':'ln_days_diff_3 + medal  + ln_num_episodes + ln_fv_pre_7',
    # '2wbp_dmnf3':'ln_days_diff_3 + medal  + ln_num_episodes + ln_prev_title_fv_3',

}

df_in['days_diff_1'] = np.where(df_in['days_diff_1']>=40, 40, df_in['days_diff_1'])


df_in = get_df(df_s, medal, forecast_category, col_id, features, target)
df_in['days_diff_1'] = np.where(df_in['days_diff_1']>=30, 30, df_in['days_diff_1'])
df_in = transform_features(df_in, ['days_diff_1', 'days_diff_2','days_diff_3'], features, target)


for i in model_list:
    model, df_in = train(df_in, target, model_list[i] )
    df_coef.append(get_coef(model, i, target))
    df_summary.append(get_summary(df_in, model, i, model_list[i], target))

df_summary = pd.concat(df_summary)
df_coef = pd.concat(df_coef).sort_values(by=['model','feature'])



# COMMAND ----------

df_in = get_df(df_s, medal, forecast_category, col_id, features, target)
df_in['days_diff_1'] = np.where(df_in['days_diff_1']>=40, 40, df_in['days_diff_1'])
df_in = transform_features(df_in, ['days_diff_1', 'days_diff_2','days_diff_3'], features, target)
display(df_in.ln_days_diff_1.describe())

df_in = get_df(df_s, medal, forecast_category, col_id, features, target)
df_in['days_diff_1'] = np.where(df_in['days_diff_1']>=20, 20, df_in['days_diff_1'])
df_in = transform_features(df_in, ['days_diff_1', 'days_diff_2','days_diff_3'], features, target)
display(df_in.ln_days_diff_1.describe())

# COMMAND ----------

df_summary

# COMMAND ----------

df_coef

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

df_in.describe()

# COMMAND ----------

corr_matrix = df_in[['ln_rfv7','ln_days_diff','ln_search_2wbp','ln_wiki_2wbp','ln_num_episodes','ln_fv_pre_7']].corr()

fig = px.imshow(corr_matrix,
                text_auto=True, # This will automatically add the correlation values as text on the cells
                labels=dict(x="Variable", y="Variable", color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                title="Correlation Matrix Heatmap")
fig.show()

# COMMAND ----------

corr_matrix = df_s[df_s.forecast_category=='wb pay1'][['rfv7','days_diff','search_2wbp','wiki_2wbp','num_episodes','fv_pre_7','gap']].corr()

fig = px.imshow(corr_matrix,
                text_auto=True, # This will automatically add the correlation values as text on the cells
                labels=dict(x="Variable", y="Variable", color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                title="Correlation Matrix Heatmap")
fig.show()

# COMMAND ----------

df_p= df_s[df_s.forecast_category=]
corr_matrix = df_[['ln_rfv7','ln_days_diff','ln_search_2wbp','ln_wiki_2wbp','ln_num_episodes','ln_fv_pre_7']].corr()

fig = px.imshow(corr_matrix,
                text_auto=True, # This will automatically add the correlation values as text on the cells
                labels=dict(x="Variable", y="Variable", color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                title="Correlation Matrix Heatmap")
fig.show()

# COMMAND ----------

display(df_in.groupby(by=['medal']).agg({'days_diff':'mean', 
                                 'num_episodes':'mean',
                                 'search_2wbp':'mean'}))
display(df_in.groupby(by=['medal']).count())

# COMMAND ----------

df_in.head()

# COMMAND ----------

df_s.premiere_date = pd.to_datetime(df_s.premiere_date)
df_s[df_s.premiere_date>='2023-05-04'][['title_name','medal','first_release_date','last_release_date','last_release_date_ref','days_diff','prev_title']].head(20)

# COMMAND ----------

df_s.premiere_date = pd.to_datetime(df_s.premiere_date)
df_s[df_s.premiere_date>='2023-01-01'][['title_name','medal','first_release_date','last_release_date','last_release_date_ref','days_diff','prev_title']].head(20)

# COMMAND ----------

df_in.premiere_date = pd.to_datetime(df_in.premiere_date)
fig = px.scatter(df_in[(df_in.rfv7>=10000) & (df_in.premiere_date>='2021-01-01')], x='days_diff_1', y='rfv7', title='fv_7 vs. days_diff Gold',
                 hover_data=['title_name', 'medal','premiere_date','prev_title_1'], color='ln_search_2wbp')  

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


