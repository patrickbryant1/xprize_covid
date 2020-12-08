#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import numpy as np
import os
import sys
import glob


from tensorflow.keras.models import model_from_json
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import pdb
def load_model():
    '''Load the model
    '''
    intercepts = []
    coefs = []
    #Fetch model weights
    all_weights = glob.glob('./model/fold*')

    #Add weights to model
    all_models = []
    for weights in all_weights:
        #Load model
        json_file = open('./model/model.json', 'r')
        model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights(weights)
        all_models.append(model)

    return all_models


def predict(start_date, end_date, path_to_ips_file, output_file_path):
    """
    Will be called like:
    python predict.py -s start_date -e end_date -ip path_to_ip_file -o path_to_output_file

    Generates and saves a file with daily new cases predictions for the given countries, regions and intervention
    plans, between start_date and end_date, included.
    :param start_date: day from which to start making predictions, as a string, format YYYY-MM-DDD
    :param end_date: day on which to stop making predictions, as a string, format YYYY-MM-DDD
    :param path_to_ips_file: path to a csv file containing the intervention plans between inception date (Jan 1 2020)
     and end_date, for the countries and regions for which a prediction is needed
    :param output_file_path: path to file to save the predictions to
    :return: Nothing. Saves the generated predictions to an output_file_path CSV file
    with columns "CountryName,RegionName,Date,PredictedDailyNewCases"
    """
    # !!! YOUR CODE HERE !!!
    ID_COLS = ['CountryName',
               'RegionName',
               'GeoID',
               'Date']
    NPI_COLS = ['C1_School closing',
                    'C2_Workplace closing',
                    'C3_Cancel public events',
                    'C4_Restrictions on gatherings',
                    'C5_Close public transport',
                    'C6_Stay at home requirements',
                    'C7_Restrictions on internal movement',
                    'C8_International travel controls',
                    'H1_Public information campaigns',
                    'H2_Testing policy',
                    'H3_Contact tracing',
                    'H6_Facial Coverings']

    #1. Select the wanted dates from the ips file
    start_date = pd.to_datetime(start_date, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date, format='%Y-%m-%d')

    # Load historical intervention plans, since inception
    hist_ips_df = pd.read_csv(path_to_ips_file,
                              parse_dates=['Date'],
                              encoding="ISO-8859-1",
                              dtype={"RegionName": str},
                              error_bad_lines=True)

    # Add GeoID column that combines CountryName and RegionName for easier manipulation of data",
    #hist_ips_df['RegionName'] = hist_ips_df['RegionName'].fillna(0)
    hist_ips_df['GeoID'] = hist_ips_df['CountryName'] + '__' + hist_ips_df['RegionName'].astype(str)
    # Fill any missing NPIs by assuming they are the same as previous day
    for npi_col in NPI_COLS:
        hist_ips_df.update(hist_ips_df.groupby(['CountryName', 'RegionName'])[npi_col].ffill().fillna(0))

    # Intervention plans to forecast for: those between start_date and end_date
    ips_df = hist_ips_df[(hist_ips_df.Date >= start_date) & (hist_ips_df.Date <= end_date)]
    ips_df.GeoID.unique()
    #2. Load the model
    all_models = load_model()

    #3. Load the additional data
    data_path = '../../data/adjusted_data.csv'
    adjusted_data = pd.read_csv(data_path,
                     parse_dates=['Date'],
                     encoding="ISO-8859-1",
                     dtype={"RegionName": str,
                            "RegionCode": str,
                            "Country_index":int,
                            "Region_index":int},
                     error_bad_lines=False)
    # Add RegionID column that combines CountryName and RegionName for easier manipulation of data

    adjusted_data['RegionName'] = adjusted_data['RegionName'].replace('0', np.nan)
    adjusted_data['GeoID'] = adjusted_data['CountryName'] + '__' + adjusted_data['RegionName'].astype(str)
    adjusted_data = adjusted_data.fillna(0)

    #4. Run the predictor
    additional_features = ['smoothed_cases',
                            'cumulative_smoothed_cases',
                            'rescaled_cases',
                            'cumulative_rescaled_cases',
                            'rescaled_cases_daily_change',
                            'smoothed_cases_daily_change',
                            'monthly_temperature',
                            'retail_and_recreation',
                            'grocery_and_pharmacy',
                            'parks',
                            'transit_stations',
                            'workplaces',
                            'residential',
                            'gross_net_income',
                            'population_density',
                            'pdi', 'idv', 'mas', 'uai', 'ltowvs', 'ivr']

    #NB_LOOKBACK_DAYS=21
    # Make predictions for each country,region pair
    geo_pred_dfs = []
    for g in ips_df.GeoID.unique():
        print('Predicting for', g)
        #Get intervention plan for g
        ips_gdf = ips_df[ips_df.GeoID == g]
         # Pull out all relevant data for g
        adjusted_data_gdf = adjusted_data[adjusted_data.GeoID == g]


        #Check the timelag to the last known date
        last_known_date = adjusted_data_gdf.Date.max()
        #It may be that the start date is much ahead of the last known date, where input will have to be predicted
        # Start predicting from start_date, unless there's a gap since last known date
        current_date = min(last_known_date + np.timedelta64(1, 'D'), start_date)
        #Select everything from df up tp current date
        adjusted_data_gdf = adjusted_data_gdf[adjusted_data_gdf['Date']<current_date]
        adjusted_data_gdf = adjusted_data_gdf.reset_index()
        #Check if enough data to predict
        if len(adjusted_data_gdf)<21:
            print('Not enough data for',g)
            pdb.set_trace()
            continue

        #Get population
        population = adjusted_data_gdf['population'].values[0]
        #Normalize the cases by 100'000 population - remember to scale back for predictions as well
        adjusted_data_gdf['rescaled_cases']=adjusted_data_gdf['rescaled_cases']/(population/100000)
        adjusted_data_gdf['cumulative_rescaled_cases']=adjusted_data_gdf['cumulative_rescaled_cases']/(population/100000)
        adjusted_data_gdf['smoothed_cases']=adjusted_data_gdf['smoothed_cases']/(population/100000)
        adjusted_data_gdf['cumulative_smoothed_cases']=adjusted_data_gdf['cumulative_smoothed_cases']/(population/100000)
        #Add daily change
        adjusted_data_gdf['rescaled_cases_daily_change']=np.append(np.zeros(1),np.array(adjusted_data_gdf['rescaled_cases'])[1:]-np.array(adjusted_data_gdf['rescaled_cases'])[:-1])
        adjusted_data_gdf['smoothed_cases_daily_change']=np.append(np.zeros(1),np.array(adjusted_data_gdf['smoothed_cases'])[1:]-np.array(adjusted_data_gdf['smoothed_cases'])[:-1])

        #Get historical NPIs
        historical_npis_g = np.array(adjusted_data_gdf[NPI_COLS])
        #Get other daily features
        adjusted_additional_g = np.array(adjusted_data_gdf[additional_features])
        #Get future NPIs
        future_npis = np.array(ips_gdf[NPI_COLS])
        # Prepare data
        #Additional features
        X_additional = adjusted_additional_g
        #Get NPIS
        X_npis = historical_npis_g
        # Make prediction for each requested day
        geo_preds = []
        days_ahead = 0
        while current_date <= end_date:
            X = np.concatenate([X_additional,X_npis],axis=1)
            # Make the prediction
            pred = []
            for m in all_models:
                pred.append(m.predict(np.array([X[-7:,:]]))[0])
            pred = np.array(pred)

            # Do not allow predicting negative cases
            pred[pred<0]=0
            #Do not allow predicting more cases than 20 % of population at a given day
            pred[pred>(0.2*population/100000)]=0.2*population/100000
            std_pred = np.std(pred,axis=0)
            pred = np.average(pred,axis=0)

            # Add if it's a requested date
            if current_date+ np.timedelta64(21, 'D') >= start_date:

                #Append the predicted dates
                days_for_pred =  current_date+ np.timedelta64(21, 'D')-start_date
                geo_preds.extend(pred[-days_for_pred.days:])
                #print(current_date.strftime('%Y-%m-%d'), pred)
            else:
                print(current_date.strftime('%Y-%m-%d'), pred, "- Skipped (intermediate missing daily cases)")

            # Append the prediction and npi's for the next x predicted days
            # in order to rollout predictions for further days.
            future_additional = np.repeat(np.array([adjusted_additional_g[-1,:]]),len(pred),axis=0)
            future_additional[:,0]=pred #add predicted cases
            future_additional[:,1]=np.cumsum(pred) #add predicted cumulative cases
            #!!!!!!!!!!!!!!!
            #Look up monthly temperature for predicted dates: 'monthly_temperature'
            #!!!!!!!!!!!!!!!
            adjusted_additional_g = np.append(adjusted_additional_g, future_additional,axis=0)
            historical_npis_g = np.append(historical_npis_g, future_npis[days_ahead:days_ahead + 21], axis=0)

            # Move to next period
            current_date = current_date + np.timedelta64(21, 'D')
            days_ahead += 21


        # Create geo_pred_df with pred column
        geo_pred_df = ips_gdf[ID_COLS].copy()
        geo_pred_df['PredictedDailyNewCases'] = np.array(geo_preds[:len(geo_pred_df)]) #*(population/100000)

        #Check
        adjusted_data_gdf = adjusted_data[adjusted_data.GeoID == g]
        adjusted_data_gdf['smoothed_cases']=adjusted_data_gdf['smoothed_cases'] #*(population/100000)
        geo_pred_df = pd.merge(geo_pred_df,adjusted_data_gdf[['Date','smoothed_cases']],on='Date',how='left')
        geo_pred_df['population']=population

        #Save
        geo_pred_dfs.append(geo_pred_df)


    #4. Obtain output
    # Combine all predictions into a single dataframe - remember to only select the requied columns later
    pred_df = pd.concat(geo_pred_dfs)
    # Save to a csv file
    #All
    pred_df.to_csv('all_'+output_file_path, index=False)
    #Only the required columns
    pred_df.drop(columns={'GeoID','smoothed_cases','population'}).to_csv(output_file_path, index=False)
    print("Saved predictions to", output_file_path)


    return None



# !!! PLEASE DO NOT EDIT. THIS IS THE OFFICIAL COMPETITION API !!!
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date",
                        dest="start_date",
                        type=str,
                        required=True,
                        help="Start date from which to predict, included, as YYYY-MM-DD. For example 2020-08-01")
    parser.add_argument("-e", "--end_date",
                        dest="end_date",
                        type=str,
                        required=True,
                        help="End date for the last prediction, included, as YYYY-MM-DD. For example 2020-08-31")
    parser.add_argument("-ip", "--interventions_plan",
                        dest="ip_file",
                        type=str,
                        required=True,
                        help="The path to an intervention plan .csv file")
    parser.add_argument("-o", "--output_file",
                        dest="output_file",
                        type=str,
                        required=True,
                        help="The path to the CSV file where predictions should be written")
    args = parser.parse_args()
    print("Generating predictions from", args.start_date, "to", args.end_date,"...")
    predict(args.start_date, args.end_date, args.ip_file, args.output_file)
    print("Done!")
