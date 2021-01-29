#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import sys
import os
import time

#Xprize predictor
# Suppress noisy Tensorflow debug logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.constraints import Constraint
from keras.layers import Dense
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Lambda
from keras.models import Model
import pdb

########REMEMBER!!! All paths will have to be absoulate in the sandbox########

###FUNCTIONS###
# Construct model
class Positive(Constraint):

    def __call__(self, w):
        return K.abs(w)

# Functions to be used for lambda layers in model
def _combine_r_and_d(x):
    r, d = x
    return r * (1. - d)

def construct_model(nb_context, nb_action, lstm_size=32, nb_lookback_days=21):
    '''Create the std predictor
    '''
    # Create context encoder
    context_input = Input(shape=(nb_lookback_days, nb_context),
                          name='context_input')
    x = LSTM(lstm_size, name='context_lstm')(context_input)
    context_output = Dense(units=1,
                           activation='softplus',
                           name='context_dense')(x)

    # Create action encoder
    # Every aspect is monotonic and nonnegative except final bias
    action_input = Input(shape=(nb_lookback_days, nb_action),
                         name='action_input')
    x = LSTM(units=lstm_size,
             kernel_constraint=Positive(),
             recurrent_constraint=Positive(),
             bias_constraint=Positive(),
             return_sequences=False,
             name='action_lstm')(action_input)
    action_output = Dense(units=1,
                          activation='sigmoid',
                          kernel_constraint=Positive(),
                          name='action_dense')(x)

    # Create prediction model
    model_output = Lambda(_combine_r_and_d, name='prediction')(
        [context_output, action_output])
    model = Model(inputs=[context_input, action_input],
                  outputs=[model_output])
    model.compile(loss='mae', optimizer='adam')

    return model

def load_model():
    '''Load the standard predictor
    '''
    # Load model weights
    nb_context = 1  # Only time series of new cases rate is used as context (PredictionRatio)
    nb_action = 12 #the NPI columns
    LSTM_SIZE = 32
    NB_LOOKBACK_DAYS = 21
    predictor = construct_model(nb_context=nb_context,
                                nb_action=nb_action,
                                lstm_size=LSTM_SIZE,
                                nb_lookback_days=NB_LOOKBACK_DAYS)
    # Fixed weights for the standard predictor.
    MODEL_WEIGHTS_FILE = './trained_model_weights.h5'
    predictor.load_weights(MODEL_WEIGHTS_FILE)

    #Save
    prescriptor_weights = np.load('./prescr_weights/selected_population.npy',allow_pickle=True)

    return predictor, prescriptor_weights

def load_inp_data(start_date,lookback_days):
    '''Load the input data for the standard predictor and prescriptor
    '''

    DATA_COLUMNS = ['CountryName',
                    'RegionName',
                    'GeoID',
                    'Date',
                    'smoothed_cases',
                    'ConfirmedCases',
                    'ConfirmedDeaths',
                    'population']

    IP_MAX_VALUES = {'C1_School closing': 3,
                    'C2_Workplace closing': 3,
                    'C3_Cancel public events': 2,
                    'C4_Restrictions on gatherings': 4,
                    'C5_Close public transport': 2,
                    'C6_Stay at home requirements': 3,
                    'C7_Restrictions on internal movement': 2,
                    'C8_International travel controls': 4,
                    'H1_Public information campaigns': 2,
                    'H2_Testing policy': 3,
                    'H3_Contact tracing': 2,
                    'H6_Facial Coverings': 4
                    }

    #Set ip maxvals

    ip_maxvals = np.array([*IP_MAX_VALUES.values()])
    #Preprocessed data
    DATA_FILE = '../../data/adjusted_data.csv'
    data = pd.read_csv(DATA_FILE,
                        parse_dates=['Date'],
                        encoding="ISO-8859-1",
                        dtype={"RegionName": str,
                        "RegionCode": str},
                        error_bad_lines=False)

    data["RegionName"]= data["RegionName"].replace('0',np.nan)
    data["GeoID"] = np.where(data["RegionName"].isnull(),
                                  data["CountryName"],
                                  data["CountryName"] + ' / ' + data["RegionName"])

    #Get inp data for prescriptor
    data = data[DATA_COLUMNS]
    data = data[(data.Date <= start_date)]

    #They predict percent change in new cases
    #This prediction ration is also the input to the next step
    # Compute percent change in new cases and deaths each day
    # Add column for proportion of population infected
    data['ProportionInfected'] = data['ConfirmedCases'] / data['population']
    # Compute number of new cases and deaths each day
    data['NewCases'] = data.groupby('GeoID').ConfirmedCases.diff().fillna(0)
    # Replace negative values (which do not make sense for these columns) with 0
    data['NewCases'] = data['NewCases'].clip(lower=0)
    # Compute smoothed versions of new cases and deaths each day using a 7 day window
    data['SmoothNewCases'] = data.groupby('GeoID')['NewCases'].rolling(7, center=False).mean().fillna(0).reset_index(0, drop=True)
    # Compute percent change in new cases and deaths each day
    data['CaseRatio'] = data.groupby('GeoID').SmoothNewCases.pct_change().fillna(0).replace(np.inf, 0) + 1
    # Create column of value to predict
    data['PredictionRatio'] = data['CaseRatio'] / (1 - data['ProportionInfected'])
    #Normalize cases for prescriptor
    data['smoothed_cases']=data['smoothed_cases']/(data['population']/100000)

    if data[data.columns[4:]].isnull().any().sum()>0:
        pdb.set_trace()
    return data, ip_maxvals

def prescribe(start_date_str, end_date_str, path_to_prior_ips_file, path_to_cost_file, output_file_path):
    '''Prescribe using the pretrained prescriptor
    The output df should contain Date, CountryName, RegionName, intervention plan, intervention index (up to 10 in total)
    '''

    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')

    # Load the past IPs data
    past_ips_df =  pd.read_csv(path_to_prior_ips_file,
                            parse_dates=['Date'],
                            encoding="ISO-8859-1",
                            error_bad_lines=False)

    # Restrict it to dates before the start_date
    past_ips_df = past_ips_df[past_ips_df['Date'] <= start_date]
    past_ips_df['GeoID'] = np.where(past_ips_df["RegionName"].isnull(),
                                  past_ips_df["CountryName"],
                                  past_ips_df["CountryName"] + ' / ' + past_ips_df["RegionName"])
    #Fill NaNs
    past_ips_df[past_ips_df.columns[3:]] = past_ips_df[past_ips_df.columns[3:]].fillna(0)
    #Load the IP costs
    ip_costs = pd.read_csv(path_to_cost_file,
                            encoding="ISO-8859-1",
                            error_bad_lines=False)

    ip_costs['GeoID'] = np.where(ip_costs["RegionName"].isnull(),
                                  ip_costs["CountryName"],
                                  ip_costs["CountryName"] + ' / ' + ip_costs["RegionName"])


    #Prescriptor parameters
    lookback_days = 21
    forecast_days = 21
    NUM_WEIGHTS=2
    NUM_LAYERS=2
    #Load the model input data
    case_data, ip_maxvals = load_inp_data(start_date,lookback_days)
    #Load model for case prediction and the predcriptor
    predictor, prescriptor_weights = load_model()

    #Columns for prescriptor
    prescriptor_cols = ['C1_School closing', 'C2_Workplace closing', 'C3_Cancel public events',
                        'C4_Restrictions on gatherings', 'C5_Close public transport',
                        'C6_Stay at home requirements', 'C7_Restrictions on internal movement',
                        'C8_International travel controls', 'H1_Public information campaigns',
                        'H2_Testing policy', 'H3_Contact tracing', 'H6_Facial Coverings', 'smoothed_cases']
    #Create the output df
    OUTPUT_COLS = ['CountryName', 'RegionName','C1_School closing', 'C2_Workplace closing',
                    'C3_Cancel public events', 'C4_Restrictions on gatherings', 'C5_Close public transport',
                    'C6_Stay at home requirements', 'C7_Restrictions on internal movement', 'C8_International travel controls',
                    'H1_Public information campaigns', 'H2_Testing policy', 'H3_Contact tracing', 'H6_Facial Coverings']
    out_df = pd.DataFrame()
    out_df['Date']=np.arange(start_date,end_date+np.timedelta64(1, 'D'),np.timedelta64(1, 'D'))
    for out_col in OUTPUT_COLS:
        out_df[out_col]=''

    ########NOTE - need to fix so that prescriptions can be made for dates before start date if dates are missing
    #Go through each region and prescribe
    prescr_dfs = []
    for g in case_data.GeoID.unique():
        print('Predicting for', g)
        #Get data for g
        g_case_data = case_data[case_data.GeoID == g]
        g_ips = past_ips_df[past_ips_df.GeoID == g]
        #Drop cols to avoid duplicates
        g_case_data = g_case_data.drop(columns=['CountryName', 'RegionName', 'GeoID'])
        gdf = pd.merge(g_case_data,g_ips,on='Date',how='left')
        #Get ip costs for g
        g_ip_costs = ip_costs[ip_costs.GeoID == g][prescriptor_cols[:-1]].values[0]

        #Check the timelag to the last known date
        last_known_date = gdf.Date.max()
        #It may be that the start date is much ahead of the last known date, where input will have to be predicted
        # Start predicting from start_date, unless there's a gap since last known date
        current_date = min(last_known_date + np.timedelta64(1, 'D'), start_date)
        #Select everything from df up tp current date
        gdf = gdf[gdf['Date']<current_date]
        gdf = gdf.reset_index()

        #Check if enough data to predict
        if len(gdf)<lookback_days:
            print('Not enough data for',g)
            #Should add zeros here
            continue

        #Number of models
        n_inds = 10
        #Prescribe while end date is not passed
        #This consists of the 12 NPIs averaged over the past 21 days and the median smoothed cases per 100'000 population in that period
        X_g = gdf[prescriptor_cols].values
        X_ind = np.average(X_g[-lookback_days:,:],axis=0)
        X_ind[-1] = np.median(X_g[-lookback_days:,12],axis=0)
        #Get the input for the xprize predictor
        X_context_ind = gdf['PredictionRatio'].values[-lookback_days:]
        X_total_cases_ind = gdf['ConfirmedCases'].values[-lookback_days:]
        X_new_cases_ind = gdf['SmoothNewCases'].values[-lookback_days:]
        g_population = gdf['population'].values[0]

        #Tile
        X_ind = np.tile(X_ind,[n_inds,1])
        X_context_ind = np.tile(X_context_ind,[n_inds,1])
        X_total_cases_ind = np.tile(X_total_cases_ind,[n_inds,1])
        X_new_cases_ind = np.tile(X_new_cases_ind,[n_inds,1])
        g_population = np.tile(g_population,n_inds)

        #Should do this for all 10 prescriptor models - can run simultaneously
        reshaped_inds = []
        for ind in range(n_inds):
            reshaped_inds.append(np.reshape(prescriptor_weights[ind],(12,NUM_WEIGHTS,NUM_LAYERS)))

        individual = np.array(reshaped_inds)

        #Create prescr
        #Num prescriptors, num pred days, num NPIs
        prescr_g = np.zeros((n_inds,(end_date-start_date).days+1,12))
        while current_date <= end_date:
            #Get prescriptions and scale with weights
            prev_ip = X_ind[:,:12].copy()
            #Get cases in last period
            prev_cases = X_ind[:,12]
            #Multiply prev ip with the 2 prescr weight layers of the individual
            prescr = prev_ip*g_ip_costs*individual[:,:,0,0]*individual[:,:,0,1]
            #Add the case focus
            prescr += np.array([prev_cases]).T*individual[:,:,1,0]*individual[:,:,1,1]
            #Now the prescr can't really increase based only on the prescr
            #The cases will have to be high for the ips to increase
            #Perhaps this is not such a bad thing
            #Make sure the prescr don't exceeed the npi maxvals
            for pi in range(prescr.shape[0]):
                prescr[pi] = np.minimum(prescr[pi,:],ip_maxvals)
            X_ind[:,:12]=prescr
            #Distribute the prescriptions in the forecast period
            #I choose to keep these stable over a three week period as changing them
            #on e.g. a daily or weekly basis in various degrees will not only make them
            #hard to follow but also confuse the public
            #Generate the predictions
            #Repeat the array for each prescritor (individual)
            future_action_sequence = []
            previous_action_sequence = []
            for pi in range(individual.shape[0]):
                future_action_sequence.append(np.tile(prescr[pi,:],[forecast_days,1]))
                previous_action_sequence.append(np.tile(prev_ip[pi,:],[forecast_days,1]))
            #time
            #tic = time.clock()

            pred_new_cases, pred_output = roll_out_predictions(predictor, X_context_ind, np.array(previous_action_sequence), np.array(future_action_sequence),X_total_cases_ind,X_new_cases_ind,g_population)

            #toc = time.clock()
            #print(np.round(toc-tic,2))
            #preds have shape n_regions x forecast_days
            #Update X_context_ind
            X_context_ind = np.copy(pred_output)
            #Update new cases and toatl cases
            X_new_cases_ind = np.copy(pred_new_cases)
            #Get cumulative cases
            X_total_cases_ind = np.cumsum(np.concatenate((np.expand_dims(X_total_cases_ind[:,-1],axis=1),X_new_cases_ind),axis=1),axis=1)[:,1:]
            median_case_preds = np.median(pred_new_cases,axis=1)/(g_population/100000)

            #Update X_ind with preds
            X_ind[:,12]=median_case_preds

            # Add if it's a requested date
            if current_date+ np.timedelta64(forecast_days, 'D') >= start_date:
                #Append the predicted dates
                days_for_pred = min(current_date+ np.timedelta64(forecast_days, 'D'),end_date)-start_date
                #Days in
                days_in = (current_date-start_date).days
                #Days to add
                days_to_add = days_for_pred.days-days_in
                for pi in range(n_inds):
                    if days_in<0: #Have to assign start from 0
                        prescr_g[pi,0:days_in+days_to_add,:]=np.tile(prescr[pi,:],[days_in+days_to_add,1])
                    else:
                        prescr_g[pi,days_in:days_in+days_to_add,:]=np.tile(prescr[pi,:],[days_to_add,1])


            else:
                print(current_date.strftime('%Y-%m-%d'),"- Skipped (intermediate missing daily cases)")

            #Increase date
            # Move to next period
            current_date = current_date + np.timedelta64(forecast_days, 'D')

        #Create a df of prescriptions
        #Add the last prescription
        prescr_g[:,-1,:] = prescr_g[:,-2,:]
        for pi in range(n_inds):
            prescr_g_df = out_df.copy()
            prescr_g_df['CountryName']=gdf.CountryName.values[0]
            prescr_g_df['RegionName']=gdf.RegionName.values[0]
            prescr_g_df[OUTPUT_COLS[2:]]=prescr_g[pi,:,:] #The first two cols are country and region names
            prescr_g_df['PrescriptionIndex']=pi+1
            #Check if nans
            if prescr_g_df[prescr_g_df.columns[3:-1]].isna().any().sum()>0:
                pdb.set_trace()
            #save
            prescr_dfs.append(prescr_g_df)

    prescription_df = pd.concat(prescr_dfs)
    #Save to a csv file
    prescription_df.to_csv(output_file_path, index=False)
    print('Prescriptions saved to', output_file_path)

    return None



def convert_ratios_to_total_cases(ratios, window_size, prev_new_cases, initial_total_cases,pop_sizes):
    '''Convert the ratios to get the case number output
    '''
    new_new_cases = []
    curr_total_cases = initial_total_cases

    for days_ahead in range(ratios.shape[1]): #each ratio will contain predictions for a region
        prev_pct_infected = curr_total_cases / pop_sizes

        new_cases = (ratios[:,days_ahead] * (1 - prev_pct_infected) - 1) * \
                    (window_size * np.mean(prev_new_cases[:,-window_size:],axis=1)) \
                    + prev_new_cases[:,-window_size]
        # new_cases can't be negative!
        new_cases[new_cases<0]=0
        # Which means total cases can't go down
        curr_total_cases += new_cases
        # Update prev_new_cases_list for next iteration of the loop
        prev_new_cases = np.concatenate((prev_new_cases,np.expand_dims(new_cases,axis=1)),axis=1)


    return prev_new_cases[:,-ratios.shape[1]:]

def roll_out_predictions(predictor, context_input, action_input, future_action_sequence, prev_confirmed_cases, prev_new_cases, pop_sizes):
    '''The predictions happen in steps of one day, why they have to be rolled out day by day.
    They also have to be converted to daily cases as some kind of ratios are predicted
    '''


    #Expand inp dims for NN
    context_input = np.expand_dims(context_input,axis=2)
    WINDOW_SIZE = 7
    nb_roll_out_days = future_action_sequence.shape[1]
    pred_output = np.zeros((future_action_sequence.shape[0],nb_roll_out_days))

    for d in range(nb_roll_out_days):

        #context input: (None, 21, 1)
        #action input  (None, 21, 12)

        pred = predictor.predict([context_input, action_input])
        if pred[np.isnan(pred)].shape[0]>0:
            pdb.set_trace()
        pred_output[:,d] = pred[:,0]
        #Add the new action input according to the predictions
        action_input[:,-d+1,:] = future_action_sequence[:,d,:]
        context_input[:,-d+1,:] = pred

    #Convert to daily new cases
    # Compute number of new cases and deaths each day
    # Gather info to convert to total cases
    initial_total_cases = prev_confirmed_cases[:,-1] #Initial total cases
    pred_new_cases = convert_ratios_to_total_cases(pred_output,WINDOW_SIZE,prev_new_cases, initial_total_cases, pop_sizes)

    return pred_new_cases, pred_output



##########MAIN###########
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date",
                        dest="start_date",
                        type=str,
                        required=True,
                        help="Start date from which to prescribe, included, as YYYY-MM-DD."
                             "For example 2020-08-01")
    parser.add_argument("-e", "--end_date",
                        dest="end_date",
                        type=str,
                        required=True,
                        help="End date for the last prescription, included, as YYYY-MM-DD."
                             "For example 2020-08-31")
    parser.add_argument("-ip", "--interventions_past",
                        dest="prev_file",
                        type=str,
                        required=True,
                        help="The path to a .csv file of previous intervention plans")
    parser.add_argument("-c", "--intervention_costs",
                        dest="cost_file",
                        type=str,
                        required=True,
                        help="Path to a .csv file containing the cost of each IP for each geo")
    parser.add_argument("-o", "--output_file",
                        dest="output_file",
                        type=str,
                        required=True,
                        help="The path to an intervention plan .csv file")
    args = parser.parse_args()
    prescribe(args.start_date, args.end_date, args.prev_file, args.cost_file, args.output_file)
    print("Done!")
