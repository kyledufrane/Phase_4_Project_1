# --------------------------------------------------------------
# Define library for all functions within this notebook
# --------------------------------------------------------------

# Import libaries needed for functions

import pickle
import plotly
import ipywidgets
import pystan
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from fbprophet import Prophet 
from sklearn.model_selection import ParameterGrid

# from pmdarima import auto_arima

sns.set_style('darkgrid')
pd.set_option('display.max_columns', 999)

import warnings
warnings.filterwarnings('ignore')


##############################################

def diff_stationary_check(df):
    '''
       args = df
       
       This function takes in a time series dataframe
       and checks for stationarity with two diffs
    '''
        
    #Plot dataframe
    df.plot()
    
    # Apply diff's
    diff = df.diff().diff().dropna()
    
    # Plot diffs
    diff.plot()
    
    # Plot ACF 
    plot_acf(diff)
    
    #Plot PACF
    plot_pacf(diff)
    
    return adfuller(diff)


#############################################

def train_sarimax(df, zipcode, p, d, q):
    
    '''
    args = (df, zipcode, p, d, q)
    
    This function takes in a dataframe, dataframe column (zipcode) as a string, & p,d,q values.
    The output will return a trained SARIMAX model and dump a pickle file.'''
    
    # Filter dataframe
    df = df[zipcode]

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    pdqs = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    # Run a grid with pdq and seasonal pdq parameters calculated above and get the best AIC value
    ans = []
    for comb in pdq:
        for combs in pdqs:
            try:
                mod = SARIMAX(df,
                                order=comb,
                                seasonal_order=combs,
                                enforce_stationarity=False,
                                enforce_invertibility=False)

                output = mod.fit()
                ans.append([comb, combs, output.aic])

            except:
                continue
    
    # Create dataframe to allow us to grab the lowest AIC value
    ans_df = pd.DataFrame(ans, columns=['pdq', 'pdqs', 'aic'])
            
    # Create model            
    ARIMA_MODEL = SARIMAX(df, 
                          order=ans_df.loc[ans_df['aic'].idxmin()]['pdq'], 
                          seasonal_order=ans_df.loc[ans_df['aic'].idxmin()]['pdqs'], 
                          enforce_stationarity=False, 
                          enforce_invertibility=False)

    # Fit the model and print results
    output = ARIMA_MODEL.fit()

    with open(f'arima_model:{zipcode}', 'wb') as f:
        pickle.dump(f'arima_model:{zipcode}', f)
        
    return output

#########################################################

def melt_data(df):
    """
    Takes the zillow_data dataset in wide form or a subset of the zillow_dataset.  
    Returns a long-form datetime dataframe 
    with the datetime column names as the index and the values as the 'values' column.
    
    If more than one row is passes in the wide-form dataset, the values column
    will be the mean of the values from the datetime columns in all of the rows.
    """
    
    melted = pd.melt(df, id_vars=['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName', 'State', 'City', 'Metro',                                        'CountyName', '10_yr_Profit'], var_name='time')
    melted['time'] = pd.to_datetime(melted['time'], infer_datetime_format=True)
    melted = melted.dropna(subset=['value'])
    return melted.groupby('time').aggregate({'value':'mean'})


###########################################################

def get_baseline(train): 

    '''args = df_train
    
    This function takes in a time series training set
    and returns a baseline RMSE score with a window of 12
    '''
    
    
    yhat = train.rolling(window=12).mean().dropna()

    ytrue = train[11:]

    rmse = np.sqrt(mean_squared_error(ytrue, yhat))

    return rmse
    
###########################################################

def prophet_rename(df):
    '''
    args = df
    
    This function takes in a df and renames the df 
    to fbprophet specs 
    
    column 1 = 'ds'
    colum 2 = 'y'
    
    '''
    df = df.reset_index()
    df.rename(columns={'time':'ds'}, inplace = True)
    df.rename(columns={'value': 'y'}, inplace = True)
    return df

###########################################################


def best_prophet_params(df_train, df_test):
    '''
    args = df_train, df_test
    
    This function takes in the time series training
    and testing set. It then performs a grid search 
    on the data and finds the lowest RMSE value. After
    which it returns a list with the RMSE values and the
    hyperparameters'''
    
    # Create an empty dictionary
    param_grid_results = {}
    
    # Create param_grid
    param_grid = {  
                    'changepoint_prior_scale': [0.005, 0.05, 0.5],
                    'changepoint_range': [0.4, 0.8],
                    'n_changepoints': [5,10,20],
                    'seasonality_prior_scale':[0.1, 1],
                    'holidays_prior_scale':[0.1, 1, 10.0],
                    'seasonality_mode': ['multiplicative', 'additive'],
                    'yearly_seasonality': [5, 10, 20]
                  }
    
    # Establish param grid with sklearns ParameterGrid
    grid = ParameterGrid(param_grid)
    
    # Instantiate model and loop through param grid
    for p in grid:
        m = Prophet(**p)
        m.fit(df_train)
        future = m.make_future_dataframe(periods=24, freq = 'm')
        forecast = m.predict(future)
        metric_df = forecast.set_index('ds')[['yhat']].join(df_test.set_index('ds')).reset_index()
        metric_df.dropna(inplace=True)
        param_grid_results[f'{p}'] = np.sqrt(mean_squared_error(metric_df.y, metric_df.yhat))
     
    # Find the lowest RMSE value, find key,value pair
    # in dictionary, append it to the list, and output
    # from function
    k_val = []
    min_value = min(param_grid_results.values())

    for k,v in param_grid_results.items():
        if v == min_value:
            k_val.append(k)
            k_val.append(v)
        
    return k_val


###########################################################


def rmse(prediction, ytrue):
    '''args = forecasted, ytrue
    This function takes in the prediction
    and ytrue values and returns the RMSE'''
    

    forcasted = prediction.predicted_mean
    
    x = pd.DataFrame(forcasted)
    
    x['true'] = ytrue['value']
    
    x = x.dropna()
    
    rmse = np.sqrt(mean_squared_error(x['true'], x['predicted_mean']))
    
    return rmse

