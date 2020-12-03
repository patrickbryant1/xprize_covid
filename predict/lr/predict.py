# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import argparse

def load_model():
    '''Load the model
    '''
    intercepts = np.load('./intercepts', allow_pickle=True)
    coefficients = np.load('./coefficients', allow_pickle=True)

def predict(start_date: str,
            end_date: str,
            path_to_ips_file: str,
            output_file_path) -> None:
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
    #1. Select the wanted dates from the ips file
    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')

    # Load historical intervention plans, since inception
    hist_ips_df = pd.read_csv(path_to_ips_file,
                              parse_dates=['Date'],
                              encoding="ISO-8859-1",
                              dtype={"RegionName": str},
                              error_bad_lines=True)

    # Add GeoID column that combines CountryName and RegionName for easier manipulation of data",
    hist_ips_df['GeoID'] = hist_ips_df['CountryName'] + '__' + hist_ips_df['RegionName'].astype(str)
    # Fill any missing NPIs by assuming they are the same as previous day
    for npi_col in NPI_COLS:
        hist_ips_df.update(hist_ips_df.groupby(['CountryName', 'RegionName'])[npi_col].ffill().fillna(0))

    # Intervention plans to forecast for: those between start_date and end_date
    ips_df = hist_ips_df[(hist_ips_df.Date >= start_date) & (hist_ips_df.Date <= end_date)]

    #1. Load the model
    model = load_model()

    #2. Load the additional data
    data_path = '../../data/adjusted_data.csv'
    adjusted_data = pd.read_csv(,
                     parse_dates=['Date'],
                     encoding="ISO-8859-1",
                     dtype={"RegionName": str,
                            "RegionCode": str,
                            "Country_index":int,
                            "Region_index":int},
                     error_bad_lines=False)
    adjusted_data = adjusted_data.fillna(0)
    # Add RegionID column that combines CountryName and RegionName for easier manipulation of data
    adjusted_data['GeoID'] = adjusted_data['CountryName'] + '__' + adjusted_data['RegionName'].astype(str)

    #3. Run the predictor
    ip_features = ['C1_School closing',
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

    additional_features = ['smoothed_cases',
                        'cumulative_smoothed_cases',
                        'rescaled_cases',
                        'cumulative_rescaled_cases',
                        'death_to_case_scale',
                        'case_death_delay',
                        'gross_net_income',
                        'population_density',
                        'monthly_temperature',
                        'retail_and_recreation',
                        'grocery_and_pharmacy',
                        'parks',
                        'transit_stations',
                        'workplaces',
                        'residential',
                        'pdi', 'idv', 'mas', 'uai', 'ltowvs', 'ivr',
                        'population']
    # Make predictions for each country,region pair
    geo_pred_dfs = []
    for g in ips_df.GeoID.unique():
        print('\nPredicting for', g)

         # Pull out all relevant data for country c
        adjusted_data_gdf = adjusted_data[adjusted_data.GeoID == g]
        last_known_date = adjusted_data_gdf.Date.max()
        ips_gdf = ips_df[ips_df.GeoID == g]
        

        # Make prediction for each day
        geo_preds = []


    #4. Obtain output

    # Create the output path
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    # Save to a csv file
    preds_df.to_csv(output_file_path, index=False)
    print(f"Saved predictions to {output_file_path}")
    raise NotImplementedError


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
    print(f"Generating predictions from {args.start_date} to {args.end_date}...")
    predict(args.start_date, args.end_date, args.ip_file, args.output_file)
    print("Done!")
