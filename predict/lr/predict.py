#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from math import e
import pdb

def load_model():
    '''Load the model
    '''
    low_intercepts = []
    low_coefs = []
    high_intercepts = []
    high_coefs = []
    #Fetch intercepts and coefficients
    for i in range(1,6):
        try:
            low_intercepts.append(np.load('./model/median/low/intercepts'+str(i)+'.npy', allow_pickle=True))
            low_coefs.append(np.load('./model/median/low/coefs'+str(i)+'.npy', allow_pickle=True))
            high_intercepts.append(np.load('./model/median/high/intercepts'+str(i)+'.npy', allow_pickle=True))
            high_coefs.append(np.load('./model/median/high/coefs'+str(i)+'.npy', allow_pickle=True))
        except:
            print('Missing fold',i)

    return np.array(low_intercepts), np.array(low_coefs),np.array(high_intercepts), np.array(high_coefs)

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

    #2. Load the model
    low_intercepts, low_coefs, high_intercepts, high_coefs = load_model()

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
    #Select only world area data
    #world_areas = {1:"Europe & Central Asia"}
    #adjusted_data = adjusted_data[adjusted_data['world_area']==world_areas[1]]
    adjusted_data['RegionName'] = adjusted_data['RegionName'].replace('0', np.nan)
    adjusted_data['GeoID'] = adjusted_data['CountryName'] + '__' + adjusted_data['RegionName'].astype(str)
    adjusted_data = adjusted_data.fillna(0)

    #4. Run the predictor
    additional_features = ['smoothed_cases',
                            'cumulative_smoothed_cases',
                            'monthly_temperature',
                            'retail_and_recreation',
                            'grocery_and_pharmacy',
                            'parks',
                            'transit_stations',
                            'workplaces',
                            'residential', #These 9 features are used as daily features
                            'death_to_case_scale', #The rest are only used once
                            'case_death_delay',
                            'gross_net_income',
                            'population_density',
                            'pdi', 'idv', 'mas', 'uai', 'ltowvs', 'ivr',
                            'population']

    NB_LOOKBACK_DAYS=21
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
        #Get no-repeat features
        death_to_case_scale = adjusted_data_gdf.loc[0,'death_to_case_scale']
        case_death_delay = adjusted_data_gdf.loc[0,'case_death_delay']
        gross_net_income = adjusted_data_gdf.loc[0,'gross_net_income']
        population_density = adjusted_data_gdf.loc[0,'population_density']
        pdi = adjusted_data_gdf.loc[0,'pdi'] #Power distance
        idv = adjusted_data_gdf.loc[0, 'idv'] #Individualism
        mas = adjusted_data_gdf.loc[0,'mas'] #Masculinity
        uai = adjusted_data_gdf.loc[0,'uai'] #Uncertainty
        ltowvs = adjusted_data_gdf.loc[0,'ltowvs'] #Long term orientation,  describes how every society has to maintain some links with its own past while dealing with the challenges of the present and future
        ivr = adjusted_data_gdf.loc[0,'ivr'] #Indulgence, Relatively weak control is called “Indulgence” and relatively strong control is called “Restraint”.
        population = adjusted_data_gdf.loc[0,'population']

        #Normalize cases to per 100'000 pop
        adjusted_data_gdf['smoothed_cases']=adjusted_data_gdf['smoothed_cases']/(population/100000)
        adjusted_data_gdf['cumulative_smoothed_cases']=adjusted_data_gdf['cumulative_smoothed_cases']/(population/100000)

        #Get historical NPIs
        historical_npis_g = np.array(adjusted_data_gdf[NPI_COLS])
        #Get other daily features
        adjusted_additional_g = np.array(adjusted_data_gdf[additional_features[:9]])
        #Get future NPIs
        future_npis = np.array(ips_gdf[NPI_COLS])

        # Make prediction for each requested day
        geo_preds = []
        days_ahead = 0
        while current_date <= end_date:
            # Prepare data - make check so that enough previous data exists
            #The np array has to be copied!!!!!!!!
            #Otherwise there is a direct link to the adjusted_additional_g which means
            #that both arrays are updated simultaneously
            X_additional = adjusted_additional_g[-NB_LOOKBACK_DAYS:].copy() #The first col is 'smoothed_cases', then 'cumulative_smoothed_cases',
            #Get change over the past NB_LOOKBACK_DAYS
            case_in_end = X_additional[-1,0]
            period_change = X_additional[-1,1]-X_additional[0,1]
            case_medians = np.median(X_additional[:,:2],axis=0)
            X_additional = np.average(X_additional,axis=0)
            X_additional[:2]=case_medians


            #Get NPIS
            X_npis = historical_npis_g[-NB_LOOKBACK_DAYS:].copy()
            X_npis = np.average(X_npis,axis=0)

            X = np.concatenate([X_additional,X_npis])
            #Add
            X = np.append(X,[death_to_case_scale,case_death_delay,gross_net_income,population_density,period_change,pdi, idv, mas, uai, ltowvs, ivr, population])

            # Make the prediction
            if np.average(adjusted_additional_g[-NB_LOOKBACK_DAYS:,0])>1.8:
                preds = np.dot(high_coefs,X)+high_intercepts
            else:
                preds = np.dot(low_coefs,X)+high_intercepts

            #Order the predictions to run through the predicted mean
            pred = np.average(np.power(e,preds))
            pred = np.arange(case_in_end,pred,(pred-case_in_end)/21) #

            #Do not allow predicting more cases than 20 % of population
            pred[pred>((1/21*population)/(population/100000))]=(1/21*population)/(population/100000)

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
            future_additional = np.repeat(np.array([adjusted_additional_g[-1,:].copy()]),len(pred),axis=0)
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
        geo_pred_df['PredictedDailyNewCases'] = np.array(geo_preds[:len(geo_pred_df)])#/(population/100000)

        #Check
        adjusted_data_gdf = adjusted_data[adjusted_data.GeoID == g]
        adjusted_data_gdf['smoothed_cases']=adjusted_data_gdf['smoothed_cases']/(population/100000)
        geo_pred_df = pd.merge(geo_pred_df,adjusted_data_gdf[['Date','smoothed_cases']],on='Date',how='left')
        geo_pred_df['population']=population
        #Vis
        fig,ax = plt.subplots(figsize=(4.5/2.54,4.5/2.54))
        plt.plot(np.arange(24),geo_pred_df['PredictedDailyNewCases'],color='grey')
        plt.bar(np.arange(24),geo_pred_df['smoothed_cases'],color='g',alpha=0.5)
        plt.title(g)
        plt.tight_layout()
        plt.savefig('./plots/'+g+'.png',format='png')
        plt.close()
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
