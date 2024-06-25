import pandas as pd
import numpy as np
import statsmodels.formula.api as sm  


# Single model
class fv_model:
    def __init__(self, model_name, target, features, region, df):
        self.model_name = model_name
        self.target = target
        self.region = region
        self.features = features
        self.df = df
        self.df['model'] = self.model_name
        self.df['target'] = self.target
        self.model = None

    # Runs quantile regression
    def train(self):
        # try:
        self.model = sm.quantreg('ln_' + self.target + ' ~ ' + self.features, data=self.df.query("future_title==0")).fit()
        self.df['fv_predict'] = np.exp(self.model.predict(self.df)) + 1
        self.df['error'] = self.df['ln_' + self.target] - self.model.predict(self.df)
        self.df = self.df.loc[~self.df.error.isnull()]
        # except:
        # self.df['fv_predict'] = np.nan
        # self.df['error'] = np.nan
        
    # Returns coefficients and t stats
    def get_coef(self):
        try:
        # Model coefficients
            df_coef = (self.model.params.reset_index()
                    .merge(self.model.tvalues.reset_index(), on='index')
                    .merge(self.model.pvalues.reset_index(), on='index')
                    )
            df_coef.columns = ['feature','coef','tstat','pvalue']
            df_coef['model'] = self.model_name
            df_coef['region'] = self.region
            df_coef['target'] = self.target
            return df_coef
        except:
            return

    # Return model RSQ and MAPE
    def get_summary(self):
        # try:
        df_summary = pd.DataFrame({'region':[self.region],
                        'model':[self.model_name],
                        'target':[self.target], 
                        'features':[self.features],
                        'rsquared':[self.model.prsquared],
                        'n_obs':[len(self.df.query("future_title==0 & error==error"))],
                        'med_ape':[np.abs(self.df.query("future_title==0").error).median()],
                        'mape':[np.abs(self.df.query("future_title==0").error).mean()]
                        })
        return df_summary
        # except:
        #     return

    # Returns predictions
    def get_pred(self, col_id):
        return self.df[col_id + ['model','target',self.target,'fv_predict']]
    
    # Returns model
    def get_model(self):
        return self.model
    
    # Returns data frame
    def get_df(self):
        return self.df
    

# Run a list of model specifications
class fv_model_set:
    def __init__(self, 
                 df, forecast_category, model_list, features_log, 
                 medal = [],
                 features_other = [],
                 target = 'tvf90',
                 titles_exclude = [],
                 col_id = ['imdb_id','title_name','full_title','premiere_date','season_number','region','forecast_category','medal','future_title'],
                 region_list = ['north america','latam','emea']):
        self.forecast_category = forecast_category
        self.model_list = model_list
        self.df = df
        self.region_list = region_list
        self.col_id = col_id
        self.features_log = features_log
        self.features_other = features_other
        self.target = target

        self.df_reg = df.query("forecast_category==@forecast_category & title_name!=@titles_exclude & medal==@medal")[col_id + features_log + features_other]

        for i in features_log:
            # Log transformations
            self.df_reg['ln_'+i] = np.log(self.df_reg[i].astype(float)+1)

    # Calculate correlations
    def calc_corr(self):
        return self.df_reg.query("future_title==0").groupby('region')[['ln_'+f for f in self.features_log]].corr()['ln_tfv90'].unstack()

    # Train all models
    def train_all(self):
        self.models = [fv_model(m,self.target,self.model_list[m],r,self.df_reg.query("region==@r")) 
                for m in self.model_list.keys() 
                for r in self.region_list]

        for m in self.models:
            m.train()
        self.df_coef = pd.concat([m.get_coef() for m in self.models]).sort_values('region')
        self.df_summary = pd.concat([m.get_summary() for m in self.models]).sort_values('region')
        self.df_pred = pd.concat([m.get_pred(self.col_id) for m in self.models])

    # Return single model
    def get_model(self, region, m):
        model_return = fv_model(m, self.target, self.model_list[m], region, self.df_reg.query("region==@region"))
        model_return.train()
        return model_return
    
    # Returns dataframe
    def get_df(self, region):
        return self.df_reg.query("region==@region")
    
    # Returns all predictions
    def get_pred(self):
        return self.df_pred
    
    # Returns single prediction for each title
    def get_pred_weighted(self):
        df_pred_weighted = self.df_pred.loc[df_pred.fv_predict>0].merge(self.df_summary[['region','model','rsquared']], left_on=['region','model'], right_on=['region','model'])
        df_pred_weighted['weight'] = df_pred_weighted.rsquared**2
        df_pred_weighted['pred_w'] = df_pred_weighted.fv_predict * df_pred_weighted.weight
        df_pred_weighted_avg = df_pred_weighted.groupby(['region','imdb_id','season_number','title_name'])[['pred_w','weight']].sum().reset_index()
        df_pred_weighted_avg['fv_predict'] = df_pred_weighted_avg.pred_w / df_pred_weighted_avg.weight
        df_pred_weighted_avg['model'] = 'weighted_avg'
        return df_pred_weighted_avg

    # Returns summary
    def get_summary(self):
        return self.df_summary.sort_values(by=['region','model'])
    
    # Returns coefficients
    def get_coef(self,region):
        return self.df_coef.query("region==@region").pivot_table(index='feature',columns='model',values='coef').reset_index()
    
    # Returns t stats
    def get_tstat(self,region):
        return self.df_coef.query("region==@region").pivot_table(index='feature',columns='model',values='tstat').reset_index()

    # Returns pvalue
    def get_pvalue(self,region):
        return self.df_coef.query("region==@region").pivot_table(index='feature',columns='model',values='pvalue').reset_index()

    # Returns coeff summary
    def get_coeff_summary(self,region):
        return self.df_coef.query("region==@region").reset_index().sort_values(by=['model','feature'])